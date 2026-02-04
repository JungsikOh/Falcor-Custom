#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;

import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;

import StaticParams;
import Reservoir;
import LoadShadingData;
import LightSample;

struct DirectLightingPass
{
    uint2 gFrameDim;
    uint gFrameCount;

    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    Texture2D<PackedHitInfo> gVBuffer;

    StructuredBuffer<Reservoir> gReservoirs;

    RWTexture2D<float4> gOutputColor;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        const uint bufferIndex = (pixel.y * gFrameDim.x) + pixel.x;

        const Ray primaryCameraRay = gScene.camera.computeRayPinhole(pixel, gFrameDim);
        const float3 viewVec = -normalize(primaryCameraRay.dir);
        const HitInfo hit = unpackHitInfo(gVBuffer[pixel]);

        ShadingData sd = loadShadingData(hit, primaryCameraRay.origin, primaryCameraRay.dir);

        // Create material instance and query its properties.
        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(hit, true /* primary hit */);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameCount);

        LightSamplingConfig lightConfig = LightSamplingConfig::create(
            envMapSampler,   // Only accessed if kUseEnvLight == true
            emissiveSampler, // Only accessed if kUseEmissiveLights == true
            kUseEnvLight     // Use ReSTIR
        );

        Reservoir reservoir = gReservoirs[bufferIndex];

        LightSample ls = {};
        bool valid = reconstructLightSample(sd.posW, sd.faceN, reservoir.lightType, reservoir.lightIndex, reservoir.lightUV, lightConfig, ls);

        Ray visibilityRay = ls.getVisibilityRay();
        // Perform inline ray query for visibility
        SceneRayQuery<1> rayQuery;
        bool isVisible= rayQuery.traceVisibilityRay(visibilityRay, RAY_FLAG_NONE, 0xff);

        float3 bsdfVal = mi.eval(sd, ls.dir, sg);

        float3 directLightingColor = bsdfVal * reservoir.Li * reservoir.W * float(isVisible);

        if (hit.isValid())
        {
            gOutputColor[pixel] = float4(directLightingColor, 1.f);
        }
        else
        {
            float3 color = float3(0.f);
            if (kUseEnvLight)
            {
                color = gScene.envMap.eval(-viewVec);
            }

            gOutputColor[pixel] = float4(color, 1.f);
        }
    }
}

cbuffer CB
{
    DirectLightingPass gDirectLightingPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gDirectLightingPass.execute(dispatchThreadId.xy);
}
