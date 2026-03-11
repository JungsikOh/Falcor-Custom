#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

import Rendering.Materials.Microfacet;
import Rendering.Materials.IsotropicGGX;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;

import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;

import StaticParams;
import ReservoirGI;
import LoadShadingData;
import PathTracer;

struct ShadingColorPass
{
    PathTracerParams params; ///< Runtime parameters.

    Texture2D<PackedHitInfo> gVBuffer;
    Texture2D<float4> directLighting;
    StructuredBuffer<PackedReservoirGI> gReservoirs;
    RWTexture2D<float4> gOutputColor;

    static const bool kUseOnlyIndirect = USE_ONLY_INDIRECT;
    static const bool kUseOnlyDirect = USE_ONLY_DIRECT;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        const uint bufferIndex = (pixel.y * params.frameDim.x) + pixel.x;

        const Ray primaryCameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        const HitInfo hit = unpackHitInfo(gVBuffer[pixel]);

        ShadingData sd = loadShadingData(hit, primaryCameraRay.origin, primaryCameraRay.dir);
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);

        // Create material instance and query its properties.
        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(hit, true /* primary hit */);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        ReservoirGI reservoir = ReservoirGI::unpackReservoirGI(gReservoirs[bufferIndex]);
        SampleGI sample = reservoir.sample;

        float3 rayOrigin = sd.computeRayOrigin();
        float3 dirToLight = normalize(sample.samplePoint - rayOrigin);
        float distToLight = distance(rayOrigin, sample.samplePoint);
        float3 Li = sample.Le * reservoir.W;

        bool valid = reservoir.M != 0 && reservoir.W > 0.f && sample.valid != 0;

        // if (valid)
        // {
        //     // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
        //     SceneRayQuery<1> rayQuery;
        //     const Ray ray = Ray(rayOrigin, dirToLight, 0.0f, distToLight);
        //     if (!rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff))
        //     {
        //         valid = false;
        //     }
        // }

        float3 indirectColor = float3(0.f);

        if (valid)
        {
            float3 bsdfVal = mi.eval(sd, dirToLight, sg);
            indirectColor = bsdfVal * Li;
        }

        float3 directColor = directLighting[pixel].xyz;

        float3 outputColor;
        if (kUseOnlyIndirect)
            outputColor = indirectColor;
        else if (kUseOnlyDirect)
            outputColor = directColor;
        else
            outputColor = directColor + indirectColor;

        gOutputColor[pixel] = float4(outputColor, 1.f);
    }
}

cbuffer CB
{
    ShadingColorPass gShadingColorPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gShadingColorPass.execute(dispatchThreadId.xy);
}
