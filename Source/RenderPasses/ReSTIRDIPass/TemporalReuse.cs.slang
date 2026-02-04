#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Utils.Color.ColorHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;

import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;

import StaticParams;
import SurfaceData;
import LightSample;
import Reservoir;

struct TemporalReusePass
{
    uint2 gFrameDim;
    uint gFrameCount;

    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    RWStructuredBuffer<Reservoir> gReservoirs;

    StructuredBuffer<PackedSurfaceData> gSurfaceData;
    StructuredBuffer<Reservoir> gPrevReservoirs;
    StructuredBuffer<PackedSurfaceData> gPrevSurfaceData;

    /// Configuration constants (can be made into shader parameters)
    static const float kNormalThreshold = 0.9f; // cos(~25°) - reject if normals differ too much
    static const float kDepthThreshold = 0.05f;  // 5% relative depth difference allowed

    /// Basic neighbor validation using normal and depth
    /// \param[in] centerNormal Normal at current pixel (normalized)
    /// \param[in] neighborNormal Normal at neighbor pixel (normalized)
    /// \param[in] centerDepth Depth at current pixel
    /// \param[in] neighborDepth Depth at neighbor pixel
    /// \return true if neighbor is valid for reuse, false otherwise
    bool isValidNeighbor(
    float3 centerNormal,
    float3 neighborNormal,
    float centerDepth,
    float neighborDepth
)
    {
        // Invalid depth check
        if (centerDepth <= 0.f || neighborDepth <= 0.f)
            return false;

        // Normal similarity test
        // dot(n1, n2) = cos(angle between normals)
        // threshold 0.9 ≈ 25° difference allowed
        float normalSimilarity = dot(centerNormal, neighborNormal);
        if (normalSimilarity < kNormalThreshold)
            return false;

        // Depth similarity test (relative difference)
        // Allows for some tolerance based on depth magnitude
        float depthRatio = centerDepth / neighborDepth;
        if (abs(depthRatio - 1.f) > kDepthThreshold)
            return false;

        return true;
    }

    /// Extended neighbor validation with configurable thresholds
    bool isValidNeighbor(
        float3 centerNormal,
        float3 neighborNormal,
        float centerDepth,
        float neighborDepth,
        float normalThreshold,
        float depthThreshold
    )
    {
        if (centerDepth <= 0.f || neighborDepth <= 0.f)
            return false;

        float normalSimilarity = dot(centerNormal, neighborNormal);
        if (normalSimilarity < normalThreshold)
            return false;

        float depthRatio = centerDepth / neighborDepth;
        if (abs(depthRatio - 1.f) > depthThreshold)
            return false;

        return true;
    }

    /// Full neighbor validation using SurfaceData structures
    bool isValidNeighbor(SurfaceData center, SurfaceData neighbor)
    {
        // Check validity first
        if (!center.isValid() || !neighbor.isValid())
            return false;

        return isValidNeighbor(
        center.normal,
        neighbor.normal,
        center.depthW,
        neighbor.depthW
        );
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        const uint bufferIndex = (pixel.y * gFrameDim.x) + pixel.x;

        /////////////////////////////////////
        /// Temporal Reuse
        /////////////////////////////////////

        const Ray primaryCameraRay = gScene.camera.computeRayPinhole(pixel, gFrameDim);
        const float3 viewVec = -normalize(primaryCameraRay.dir);

        SurfaceData currSurfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[bufferIndex]);
        Reservoir currReservoir = gReservoirs[bufferIndex];

        if (!currSurfaceData.isValid()) return;

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameCount);

        uint historyLimit = 20 * currReservoir.M;

        // Reproject previous reservoir
        float2 motionVec = currSurfaceData.motionVec;
        int2 prevPixel = int2(float2(pixel) + motionVec * float2(gFrameDim) + sampleNext2D(sg));

        if (any(prevPixel >= gFrameDim) || any(prevPixel < 0)) return;

        const uint prevBufferIndex = (prevPixel.y * gFrameDim.x) + prevPixel.x;

        SurfaceData prevSurfaceData = SurfaceData::unpackSurfaceData(gPrevSurfaceData[prevBufferIndex]);
        Reservoir prevReservoir = gPrevReservoirs[prevBufferIndex];

        prevReservoir.M = min(historyLimit, prevReservoir.M);

        // Check if the reprojected surface data matches the current surface data
        // bool isValid = (prevSurfaceData.getPrimitiveID() == currSurfaceData.getPrimitiveID()) &&
        //                (prevSurfaceData.getMaterialID() == currSurfaceData.getMaterialID());

        bool isValid = isValidNeighbor(currSurfaceData, prevSurfaceData);

        LightSamplingConfig lightConfig = LightSamplingConfig::create(
            envMapSampler,   // Only accessed if kUseEnvLight == true
            emissiveSampler, // Only accessed if kUseEmissiveLights == true
            kUseEnvLight     // Use ReSTIR
        );

        Reservoir outputReservoir = {};

        if (!isValid) return;

        if (kBisedMode == 0)
        {
            // Biased mode
            // Update reservoir with previous reservoir
            LightSample currLightSample = {};
            bool validCurrLightSample = reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, currReservoir.lightType, currReservoir.lightIndex, currReservoir.lightUV, lightConfig, currLightSample );
            float currTargetPdf = currReservoir.W > 0.f ? currSurfaceData.evalTargetPDF(currLightSample, viewVec) : 0.f;
            outputReservoir.update(currReservoir, currLightSample, currTargetPdf, sampleNext1D(sg));

            LightSample prevLightSample = {};
            bool validPrevLightSample = reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, prevReservoir.lightType, prevReservoir.lightIndex, prevReservoir.lightUV, lightConfig, prevLightSample);
            float prevTargetPdf = currSurfaceData.evalTargetPDF(prevLightSample, viewVec);
            outputReservoir.update(prevReservoir, prevLightSample, prevTargetPdf, sampleNext1D(sg));

            if (outputReservoir.targetPdf > 0.f) outputReservoir.computeW();
        }
        else if (kBisedMode == 1)
        {
            // Unbiased Naive mode
            uint Z = 0u;
            LightSample selectedLightSample = {};
            bool selected = false;

            // Update reservoir with previous reservoir
            LightSample currLightSample = {};
            bool validCurrLightSample = reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, currReservoir.lightType, currReservoir.lightIndex, currReservoir.lightUV, lightConfig, currLightSample );
            float currTargetPdf = currReservoir.W > 0.f ? currSurfaceData.evalTargetPDF(currLightSample, viewVec) : 0.f;
            selected = outputReservoir.update(currReservoir, currLightSample, currTargetPdf, sampleNext1D(sg));
            if (selected) selectedLightSample = currLightSample;

            selected = false;
            LightSample prevLightSample = {};
            bool validPrevLightSample = reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, prevReservoir.lightType, prevReservoir.lightIndex, prevReservoir.lightUV, lightConfig, prevLightSample );
            float prevTargetPdf = currSurfaceData.evalTargetPDF(prevLightSample, viewVec);
            selected = outputReservoir.update(prevReservoir, prevLightSample, prevTargetPdf, sampleNext1D(sg));
            if (selected) selectedLightSample = prevLightSample;

            LightSample temp;

            reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, selectedLightSample.lightType, selectedLightSample.lightIndex, selectedLightSample.uv, lightConfig, temp );
            if (currSurfaceData.evalTargetPDF(temp, viewVec) > 0.f) Z += currReservoir.M;

            reconstructLightSample( prevSurfaceData.posW, prevSurfaceData.normal, selectedLightSample.lightType, selectedLightSample.lightIndex, selectedLightSample.uv, lightConfig, temp );
            if (prevSurfaceData.evalTargetPDF(temp, viewVec) > 0.f) Z += prevReservoir.M;

            float m = Z > 0 ? 1.f / float(Z) : 0.f;
            if (outputReservoir.targetPdf > 0.f && m > 0 && outputReservoir.weightSum > 0.f) outputReservoir.W = (1.f / outputReservoir.targetPdf) * (outputReservoir.weightSum * m);
        }
        else if (kBisedMode == 2)
        {
            // Unbiased MIS mode
            LightSample selectedLightSample = {};
            bool selected = false;
            bool isNeighborSelected = false;

            // Update reservoir with previous reservoir
            LightSample currLightSample = {};
            bool validCurrLightSample = reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, currReservoir.lightType, currReservoir.lightIndex, currReservoir.lightUV, lightConfig, currLightSample );
            float currTargetPdf = currReservoir.W > 0.f ? currSurfaceData.evalTargetPDF(currLightSample, viewVec) : 0.f;
            selected = outputReservoir.update(currReservoir, currLightSample, currTargetPdf, sampleNext1D(sg));
            if (selected) selectedLightSample = currLightSample;

            selected = false;
            LightSample prevLightSample = {};
            bool validPrevLightSample = reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, prevReservoir.lightType, prevReservoir.lightIndex, prevReservoir.lightUV, lightConfig, prevLightSample );
            float prevTargetPdf = currSurfaceData.evalTargetPDF(prevLightSample, viewVec);
            selected = outputReservoir.update(prevReservoir, prevLightSample, prevTargetPdf, sampleNext1D(sg));
            if (selected)
            {
                selectedLightSample = prevLightSample;
                isNeighborSelected = true;
            }

            float pSum = 0.f;

            LightSample temp;
            reconstructLightSample( currSurfaceData.posW, currSurfaceData.normal, selectedLightSample.lightType, selectedLightSample.lightIndex, selectedLightSample.uv, lightConfig, temp );
            float currPdf = currSurfaceData.evalTargetPDF(temp, viewVec);
            pSum += currPdf * currReservoir.M;

            reconstructLightSample( prevSurfaceData.posW, prevSurfaceData.normal, selectedLightSample.lightType, selectedLightSample.lightIndex, selectedLightSample.uv, lightConfig, temp );
            float prevPdf = prevSurfaceData.evalTargetPDF(temp, viewVec);
            pSum += prevPdf * prevReservoir.M;

            float m = pSum > 0.f ? (isNeighborSelected ? prevPdf : currPdf) / pSum : 0.f;
            if (outputReservoir.targetPdf > 0.f && m > 0 && outputReservoir.weightSum > 0.f) outputReservoir.W = (1.f / outputReservoir.targetPdf) * (outputReservoir.weightSum * m);
        }

        gReservoirs[bufferIndex] = outputReservoir;
    }
};

cbuffer CB
{
    TemporalReusePass gTemporalReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gTemporalReusePass.execute(dispatchThreadId.xy);
}
