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

import ReservoirGI;

struct PackedSurfaceData
{
    uint3 posW;         ///< asuint(world position)
    uint depthW;        ///< linear depth in view space (negative sentinel for invalid)

    uint packedNormal;  ///< encodeNormal2x16
    uint packedWeights; ///< packUnorm4x8(float4(diffuse, specular, roughness, diffuseProb))
    uint2 _pad0 = 0;
};

// 44B / 32B
struct SurfaceData
{
    float3 posW;
    float depthW;
    float3 normal;
    float diffuseWeight;
    float specularWeight;
    float roughness;
    float diffuseProb;

    float3 viewDir; // Not stored in PackedSurfaceData

    __init()
    {
        depthW = -FLT_MAX;
    }

    static SurfaceData create( const ShadingData sd, const IMaterialInstance mi,
        const float3 cameraPos, const float3 cameraW,
        const HitInfo hit, const float3 viewDir)
    {
        SurfaceData s = {};
        BSDFProperties p = mi.getProperties(sd);
        s.posW = sd.posW;
        s.normal = sd.getOrientedFaceNormal();
        s.diffuseWeight = luminance(p.diffuseReflectionAlbedo);
        s.specularWeight = luminance(p.specularReflectionAlbedo);
        s.roughness = p.roughness;
        s.diffuseProb = s.diffuseWeight / max(s.diffuseWeight + s.specularWeight, 1e-4);
        s.depthW = dot(sd.posW - cameraPos, cameraW);
        s.viewDir = viewDir;

        return s;
    }

    static SurfaceData unpackSurfaceData(const PackedSurfaceData p)
    {
        SurfaceData s = {};
        s.posW = asfloat(p.posW);
        s.depthW = asfloat(p.depthW);
        s.normal = decodeNormal2x16(p.packedNormal);
        float4 w = unpackUnorm4x8(p.packedWeights);
        s.diffuseWeight = w.x;
        s.specularWeight = w.y;
        s.roughness = w.z;
        s.diffuseProb = w.w;

        return s;
    }

    PackedSurfaceData pack()
    {
        PackedSurfaceData p = {};
        p.posW = asuint(posW);
        p.depthW = asuint(depthW);
        p.packedNormal = encodeNormal2x16(normal);
        p.packedWeights = packUnorm4x8(float4(diffuseWeight, specularWeight, roughness, diffuseProb));

        return p;
    }

    bool isValid()
    {
        return depthW != -FLT_MAX;
    }

    /** Evaluates the visibility of a light sample ray from the surface point.
        Performs ray tracing to check if the light is occluded.
        \return 1.0 if visible (no occlusion), 0.0 if occluded.
    */
    bool evalVisibility(const Ray visibilityRay)
    {
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
    float evalTargetPDF(inout SampleGI sample, const float3 wo)
    {
        float3 result = float3(0.f);
        float3 sampleToLight = normalize(sample.samplePoint - posW);
        float f = evalBRDF(sampleToLight, wo);

        // Check that the sample direction is in the upper hemisphere
        float cosTheta = saturate(dot(sampleToLight, this.normal));
        if (cosTheta <= 0.f) return 0.f;

        // #1.
        result = f * sample.Le;
        // #2.
        // result = sample.Le;

        return luminance(result);
    }
};

static bool AreMaterialSimilar(const SurfaceData a, const SurfaceData b)
{
    // Invalid surface는 재사용 금지
    if (!a.isValid() || !b.isValid())
    {
        return false;
    }

    // Tunable thresholds
    const float kRoughnessThreshold = 0.12f;
    const float kDiffuseThreshold = 0.12f;
    const float kSpecularThreshold = 0.12f;
    const float kLobeMixThreshold = 0.08f;

    if (abs(a.roughness - b.roughness) > kRoughnessThreshold)
    {
        return false;
    }

    if (abs(a.diffuseWeight - b.diffuseWeight) > kDiffuseThreshold)
    {
        return false;
    }

    if (abs(a.specularWeight - b.specularWeight) > kSpecularThreshold)
    {
        return false;
    }

    if (abs(a.diffuseProb - b.diffuseProb) > kLobeMixThreshold)
    {
        return false;
    }

    return true;
}
