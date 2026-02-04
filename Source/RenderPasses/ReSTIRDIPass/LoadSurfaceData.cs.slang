#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

__exported import Scene.Scene;
__exported import Scene.Shading;
__exported import Scene.Material.ShadingUtils;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;

import Scene.SceneTypes;
import Scene.ShadingData;
import Rendering.Materials.IBSDF;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;

import LoadShadingData;
import PathVertex;
import LightSample;
import SurfaceData;
import Reservoir;

import StaticParams;

struct LoadSurfaceDataPass
{
    uint2 gFrameDim;
    uint gFrameCount;

    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    Texture2D<PackedHitInfo> gVBuffer;
    Texture2D<float2> gMotionVectors;

    RWStructuredBuffer<PackedSurfaceData> gSurfaceData;
    RWStructuredBuffer<Reservoir> gReservoirs;
    RWTexture2D<float4> gOutputColor;

    void writeInvalid(const uint bufferIndex, const uint2 pixel)
    {
        gSurfaceData[bufferIndex].depthW = -FLT_MAX;
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;
        gOutputColor[pixel] = float4(float3(0.f), 1.f);
        const uint bufferIndex = (pixel.y * gFrameDim.x) + pixel.x;

        const Ray primaryCameraRay = gScene.camera.computeRayPinhole(pixel, gFrameDim);
        const float3 viewVec = -normalize(primaryCameraRay.dir);
        const HitInfo hit = unpackHitInfo(gVBuffer[pixel]);
        bool isValid = hit.isValid();

        if (!isValid)
        {
            gOutputColor[pixel] = float4(float3(0.f), 1.f);
            writeInvalid(bufferIndex, pixel);
            return;
        }

        /////////////////////////////////////
        /// Load shading data
        /////////////////////////////////////

        ShadingData sd = loadShadingData(hit, primaryCameraRay.origin, primaryCameraRay.dir);

        // Create material instance and query its properties.
        let lod = ExplicitLodTextureSampler(0.f);
        let hints = getMaterialInstanceHints(hit, true /* primary hit */);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        uint lobeTypes = mi.getLobeTypes(sd);
        bool validSurface = (lobeTypes & (uint)LobeType::NonDeltaReflection) != 0;

        if (!validSurface)
        {
            gOutputColor[pixel] = float4(float3(0.f), 1.f);
            writeInvalid(bufferIndex, pixel);
            return;
        }

        // Valid surface - create and pack surface data
        SurfaceData surface = SurfaceData::create(sd, mi, gScene.camera.getPosition(), hit, gMotionVectors[pixel]);

        /////////////////////////////////////
        /// Generate reservoir
        /////////////////////////////////////

        Reservoir reservoir = {};

        if (surface.isValid())
        {
            TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameCount);

            LightSamplingConfig lightConfig = LightSamplingConfig::create(
                envMapSampler,   // Only accessed if kUseEnvLight == true
                emissiveSampler, // Only accessed if kUseEmissiveLights == true
                kUseEnvLight     // Use ReSTIR
            );

            PathVertex vertex = PathVertex(sd.posW, sd.faceN, sd.frontFacing);
            uint lightCount = gScene.getLightCount();

            LightSample selectedSample = {};
            bool sampleUpperHemisphere = (lobeTypes & (uint)LobeType::NonDeltaReflection) != 0;
            bool sampleLowerHemisphere = (lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0;

            for (uint i = 0; i < kCandidateCount; i++)
            {
                LightSample ls = {};
                bool valid = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, sg, lightConfig, ls);

                if (!valid)
                {
                    reservoir.M += 1;
                    continue;
                }
                else
                {
                    float targetPdf = surface.evalTargetPDF(ls, viewVec);

                    bool wasSelected = reservoir.update(ls, targetPdf, ls.pdf, sampleNext1D(sg));
                    if (wasSelected) selectedSample = ls;
                }
            }
            bool isVisible = surface.evalVisibility(selectedSample);

            if (!isVisible)
            {
                // Occluded - invalidate the reservoir
                reservoir.W = 0.f;
            }
            else
            {
                reservoir.computeW();
            }
        }

        gReservoirs[bufferIndex] = reservoir;
        gSurfaceData[bufferIndex] = surface.pack();
    }
};

cbuffer CB
{
    LoadSurfaceDataPass gLoadSurfaceDataPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gLoadSurfaceDataPass.execute(dispatchThreadId.xy);
}
