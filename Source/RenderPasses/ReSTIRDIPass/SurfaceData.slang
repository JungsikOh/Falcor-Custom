#include "Utils/Math/MathConstants.slangh"

import Rendering.Materials.IBSDF;
import Utils.Sampling.TinyUniformSampleGenerator;
import Scene.RaytracingInline;
import Rendering.Materials.Microfacet;
import Rendering.Materials.Fresnel;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import PathVertex;
import LightSample;
import Reservoir;
import StaticParams;

struct PackedSurfaceData
{
    uint3 posW;     ///< asuint(world position)
    uint packedWeights; ///< packUnorm4x8(float4(diffuse, specular, roughness, diffuseProb))
    uint packedNormal;      ///< encodeNormal2x16
    float depthW;    ///< linear depth in view space (negative sentinel for invalid)
    float2 motionVec;   ///< prev->curr in NDC or pixel
    uint primMatId;     ///< (primId << 16) | materialId
};

struct SurfaceData
{
    float3 posW;
    float diffuseWeight;
    float specularWeight;
    float roughness;
    float diffuseProb;
    float3 normal;
    float depthW;
    float2 motionVec;
    uint primMatId;

    __init()
    {
        depthW = -FLT_MAX;
    }

    static SurfaceData create( const ShadingData sd, const IMaterialInstance mi, const float3 cameraPos, const HitInfo hit, const float2 motionVec )
    {
        SurfaceData s = {};
        BSDFProperties p = mi.getProperties(sd);
        uint primId;
        uint materialId;

        if (hit.getType() == HitType::Triangle)
        {
            TriangleHit tri = hit.getTriangleHit();
            primId = tri.primitiveIndex;
            materialId = gScene.getMaterialID(tri.instanceID);
        }
        else
        {
            primId = 0;
            materialId = 0;
        }

        s.posW = sd.posW;
        s.normal = sd.getOrientedFaceNormal();
        s.diffuseWeight = luminance(p.diffuseReflectionAlbedo);
        s.specularWeight = luminance(p.specularReflectionAlbedo);
        s.roughness = p.roughness;
        s.diffuseProb = s.diffuseWeight / max(s.diffuseWeight + s.specularWeight, 1e-4);
        s.depthW = distance(sd.posW, cameraPos);
        s.motionVec = motionVec;
        s.primMatId = (primId << 16) | (materialId & 0xFFFF);
        return s;
    }

    static SurfaceData unpackSurfaceData(const PackedSurfaceData p)
    {
        SurfaceData s = {};
        s.posW = asfloat(p.posW);
        s.normal = decodeNormal2x16(p.packedNormal);
        float4 w = unpackUnorm4x8(p.packedWeights);
        s.diffuseWeight = w.x;
        s.specularWeight = w.y;
        s.roughness = w.z;
        s.diffuseProb = w.w;
        s.depthW = p.depthW;
        s.motionVec = p.motionVec;
        s.primMatId = p.primMatId;
        return s;
    }

    PackedSurfaceData pack()
    {
        PackedSurfaceData p = {};
        p.posW = asuint(posW);
        p.packedNormal = encodeNormal2x16(normal);
        p.packedWeights = packUnorm4x8(float4(diffuseWeight, specularWeight, roughness, diffuseProb));
        p.depthW = depthW;
        p.motionVec = motionVec;
        p.primMatId = primMatId;
        return p;
    }

    /** Convert SurfaceData to PathVertex for light sampling.
        Creates a minimal PathVertex compatible with generateLightSample().
        \return PathVertex for use with light sampling functions.
    */
    PathVertex toPathVertex()
    {
        PathVertex vertex;
        vertex.pos = posW;
        vertex.faceNormal = normal;
        vertex.frontFacing = true;
        return vertex;
    }

    /** Get material ID from packed data.
     */
    uint getMaterialID()
    {
        return primMatId & 0xFFFF;
    }

    /** Get primitive ID from packed data.
     */
    uint getPrimitiveID()
    {
        return primMatId >> 16;
    }

    bool isValid()
    {
        return depthW != -FLT_MAX;
    }

    /** Evaluates the visibility of a light sample from the surface point.
        Performs ray tracing to check if the light is occluded.
        \param[in] lightSample Light sample to test visibility for.
        \return 1.0 if visible (no occlusion), 0.0 if occluded.
    */
    bool evalVisibility(const LightSample lightSample)
    {
        if (!lightSample.isValid())
            return false;

        // Use the pre-computed ray origin and direction from light sample
        // Light sample already has proper offsetting to avoid self-intersection
        float3 rayOrigin = lightSample.origin;
        float3 rayDir = lightSample.dir;
        float rayDistance = lightSample.distance;

        // Create visibility ray
        Ray visibilityRay = Ray(rayOrigin, rayDir, 0.f, rayDistance);

        // Perform inline ray query for visibility
        SceneRayQuery<1> rayQuery;
        return rayQuery.traceVisibilityRay(visibilityRay, RAY_FLAG_NONE, 0xff);
    }

    /** Evaluates the BRDF (Bidirectional Reflectance Distribution Function), giving the proportion of light reflected along an outgoing direction (wo), given an incoming light direction (wi).
        \param[in] wi Incoming light direction (normalized).
        \param[in] wo Outgoing view direction (normalized).
        \return BRDF value weighted by cosine term (wiDotN).
    */
    float evalBRDF(const float3 wi, const float3 wo)
    {
        float wiDotN = saturate(dot(wi, normal));
        float woDotN = saturate(dot(wo, normal));

        // Early out if either direction is below the surface
        if (wiDotN <= 0.f) //.//|| woDotN <= 0.f)
            return 0.f;

        // Compute half vector
        float3 H = normalize(wi + wo);
        float NDotH = saturate(dot(normal, H));
        float wiDotH = saturate(dot(wi, H));

        // Compute GGX microfacet BRDF for specular component
        float alpha = roughness * roughness;
        // GGX Normal Distribution Function (D term)
        // D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)
        float alpha2 = alpha * alpha;
        float NDotH2 = NDotH * NDotH;
        float denom = NDotH2 * (alpha2 - 1.f) + 1.f;
        float D = alpha2 / (M_PI * denom * denom);

        // Smith GGX Masking-Shadowing Function (G term) - Separable form
        // G(wi, wo) = G1(wi) * G1(wo)
        // G1(v) = 2 * (n·v) / ((n·v) + sqrt(α² + (1 - α²) * (n·v)²))
        float wiDotN2 = wiDotN * wiDotN;
        float woDotN2 = woDotN * woDotN;
        float G1_wi = 2.f * wiDotN / (wiDotN + sqrt(alpha2 + (1.f - alpha2) * wiDotN2));
        float G1_wo = 2.f * woDotN / (woDotN + sqrt(alpha2 + (1.f - alpha2) * woDotN2));
        float G = G1_wi * G1_wo;

        // Fresnel for specular reflection (assuming F0 = specularWeight as reflectance at normal incidence)
        // For physically plausible results, we normalize by specularWeight to get proper ratio
        float F = specularWeight < 1e-8f ? 0.f : evalFresnelSchlick(specularWeight, 1.f, wiDotH) / specularWeight;

        // Diffuse component: Lambertian BRDF with cosine term
        float diffuse = M_1_PI * wiDotN;

        // Specular component: Cook-Torrance microfacet BRDF with cosine term
        // Formula: D * G * F / (4 * wiDotN * woDotN) * wiDotN = D * G * F / (4 * woDotN)
        float specular = max(0.f, D * G * F * 0.25f / max(woDotN, 1e-6f));

        // Mix diffuse and specular based on their weights
        float weightSum = diffuseWeight + specularWeight;
        float diffuseSpecularMix = weightSum < 1e-7f ? 1.f : diffuseWeight / weightSum;

        return lerp(specular, diffuse, diffuseSpecularMix);
    }

    /** Evaluates the target PDF (Probability Density Function) for sampling a specific direction of light given the outgoing direction (wo).
        This function computes the unnormalized target function for ReSTIR, which is proportional to the contribution of the light sample.
        Different cases handle different types of light sources.
        \param[in] sample Light sample containing light type, position/direction, normal, and radiance.
        \param[in] wo Outgoing view direction (normalized).
        \return Unnormalized target PDF value (Le * BRDF * geometry term).
    */
    float evalTargetPDF(inout LightSample sample, const float3 wo)
    {
        if (!sample.isValid())
            return 0.f;

        float3 result = float3(0.f);

        sample.pdf = (sample.lightType == (uint)LightType::Analytic ? 1.f / gScene.getLightCount() : sample.pdf);
        sample.Li *= sample.pdf;

        float bsdfVal = evalBRDF(sample.dir, wo);
        result = bsdfVal * sample.Li;

        // switch (sample.lightType)
        // {
        // //────────────────────────────────────────────────────────────
        // // Analytic Lights (Point, Spot, Directional)
        // //────────────────────────────────────────────────────────────
        // case (uint)LightType::Analytic:
        // {
        //     if (sample.distance == FLT_MAX)
        //     {
        //         // Directional Light: 거리 감쇠 없음
        //         // p̂ = Le × BRDF × cosθ_i
        //         // (cosθ_i는 이미 brdfWeight에 포함되어 있다면 제외)
        //         result = sample.Li * bsdfVal;
        //     }
        //     else
        //     {
        //         // Point/Spot Light: 거리² 감쇠
        //         // p̂ = Le × BRDF × cosθ_i / d²
        //         float distSq = sample.distance * sample.distance;
        //         result = sample.Li * bsdfVal / max(distSq, 1e-6f);
        //     }
        //     break;
        // }

        // //────────────────────────────────────────────────────────────
        // // Emissive Lights (Area Lights / Mesh Emitters)
        // //────────────────────────────────────────────────────────────
        // case (uint)LightType::Emissive:
        // {
        //     // Area Light: Full geometry term 필요
        //     // p̂ = Le × BRDF × (cosθ_i × cosθ_o') / d²
        //     //
        //     // cosθ_i = dot(wi, surface.normal)  → brdfWeight에 포함
        //     // cosθ_o' = dot(-wi, light.normal)  → 별도 계산 필요

        //     float distSq = sample.distance * sample.distance;

        //     // Light 표면에서의 cosine (light가 shading point를 바라보는 각도)
        //     float cosAtLight = max(0.f, dot(-sample.dir, sample.normalW));

        //     // Geometry term: cosθ_o' / d²
        //     float geometryTerm = cosAtLight / max(distSq, 1e-6f);

        //     result = sample.Li * bsdfVal * geometryTerm;
        //     break;
        // }

        // //────────────────────────────────────────────────────────────
        // // Environment Map
        // //────────────────────────────────────────────────────────────
        // case (uint)LightType::EnvMap:
        // {
        //     // EnvMap: 무한히 먼 광원, 거리 감쇠 없음
        //     // p̂ = Le(ω_i) × BRDF × cosθ_i
        //     // (cosθ_i는 brdfWeight에 포함)

        //     result = sample.Li * bsdfVal;
        //     break;
        // }

        // default:
        //     return 0.f;
        // }

        return luminance(result);
    }
};
