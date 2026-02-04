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

struct SpatialReusePass
{
    uint2 gFrameDim;
    uint gFrameCount;

    // Samplers
    EnvMapSampler envMapSampler;          ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    StructuredBuffer<Reservoir> gReservoirs;
    StructuredBuffer<PackedSurfaceData> gSurfaceData;

    RWStructuredBuffer<Reservoir> gOutReservoirs;

    /// Neighbor validation for spatial reuse
    /// Checks geometric similarity between current pixel and neighbor pixel
    /// to avoid reusing samples across discontinuities (edges, depth gaps, etc.)

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

    // Hammersley sequence for low-discrepancy sampling
    float radicalInverse_VdC(uint bits)
    {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return float(bits) * 2.3283064365386963e-10; // / 0x100000000
    }

    float2 hammersley(uint i, uint N)
    {
    return float2(float(i) / float(N), radicalInverse_VdC(i));
    }

    // Convert uniform sample to disk sample (for circular neighborhood)
    float2 sampleDisk(float2 u)
    {
        float r = sqrt(u.x);
        float theta = 2.0 * M_PI * u.y;
        return float2(r * cos(theta), r * sin(theta));
    }

    // Get neighbor pixel using low-discrepancy sequence
    int2 getNeighborPixelLowDiscrepancy(
    uint2 centerPixel,
    uint sampleIndex,
    uint totalSamples,
    float radius,
    uint frameCount)
    {
        // 1. Hammersley 시퀀스로 균일 샘플 생성
        float2 u = hammersley(sampleIndex, totalSamples);

        // 2. 먼저 disk에 매핑 (균일 분포 유지)
        float2 diskSample = sampleDisk(u);
        // diskSample은 이제 [-1, 1]² 범위의 단위 디스크 내 점

        // 3. 프레임별 회전 적용 (디스크 좌표를 회전)
        float rotation = frac(float(frameCount) * 0.618033988749895);
        float angle = rotation * 2.0 * M_PI;

        float2 rotated;
        rotated.x = diskSample.x * cos(angle) - diskSample.y * sin(angle);
        rotated.y = diskSample.x * sin(angle) + diskSample.y * cos(angle);

        // 4. 반지름 스케일 적용
        int2 offset = int2(round(rotated * radius));

        return int2(centerPixel) + offset;
    }

    // ATrousSampling.slang

    // À-Trous sampling pattern offsets (3x3 kernel)
    static const int2 kATrousOffsets[9] = {
    int2(-1, -1), int2(0, -1), int2(1, -1),
    int2(-1, 0), int2(0, 0), int2(1, 0),
    int2(-1, 1), int2(0, 1), int2(1, 1)
    };

    // 5-tap cross pattern (more efficient)
    static const int2 kATrousOffsets5[5] = {
    int2(0, 0),
    int2(-1, 0), int2(1, 0),
    int2(0, -1), int2(0, 1)
    };

    // Get neighbor using À-Trous hierarchical sampling
    // iteration: 0, 1, 2, ... (doubles the step size each iteration)
    int2 getNeighborPixelATrous(
    int2 centerPixel,
    uint sampleIndex,
    uint iteration)
    {
        // Step size doubles each iteration: 1, 2, 4, 8, ...
        int stepSize = 1 << iteration;

        int2 offset = kATrousOffsets5[sampleIndex] * stepSize;

        return centerPixel + offset;
    }

    // Combined À-Trous with randomization (as in SVGF)
    int2 getNeighborPixelATrousRandomized<S : ISampleGenerator>(
    int2 centerPixel,
    uint sampleIndex,
    uint iteration,
    inout S sg)
    {
        int stepSize = 1 << iteration;

        // Add small random jitter to reduce aliasing
        float2 jitter = float2(sampleNext1D(sg) - 0.5, sampleNext1D(sg) - 0.5);

        int2 baseOffset = kATrousOffsets5[sampleIndex] * stepSize;
        int2 jitteredOffset = baseOffset + int2(round(jitter * float(stepSize) * 0.5));

        return centerPixel + jitteredOffset;
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= gFrameDim)) return;

        const uint bufferIndex = (pixel.y * gFrameDim.x) + pixel.x;

        /////////////////////
        /// Spatial Reuse ///
        /////////////////////

        const Ray primaryCameraRay = gScene.camera.computeRayPinhole(pixel, gFrameDim);
        const float3 viewVec = -normalize(primaryCameraRay.dir);

        Reservoir outputReservoir = {};
        Reservoir currentReservoir = gReservoirs[bufferIndex];
        SurfaceData surfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[bufferIndex]);

        if (!surfaceData.isValid())
        {
            gOutReservoirs[bufferIndex] = currentReservoir; // 기존 값 유지
            return;
        }

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameCount);
        LightSamplingConfig lightConfig = LightSamplingConfig::create(
            envMapSampler,   // Only accessed if kUseEnvLight == true
            emissiveSampler, // Only accessed if kUseEmissiveLights == true
            kUseEnvLight     // Use ReSTIR
        );

        LightSample ls = {};

        bool validLight = reconstructLightSample(
            surfaceData.posW, surfaceData.normal,
            currentReservoir.lightType, currentReservoir.lightIndex, currentReservoir.lightUV, lightConfig, ls);

        float currentTargetPdf = currentReservoir.W > 0.f ? surfaceData.evalTargetPDF(ls, viewVec) : 0.f;

        outputReservoir.update(currentReservoir, ls, currentTargetPdf, sampleNext1D(sg));

        const uint pixelRadius = 15; // Spatial reuse radius in pixels
        if (kBisedMode == 0) // BIASED
        {
            for (uint i = 0; i < kSpatialReuseNeighbors; i++)
            {
                int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialReuseNeighbors, pixelRadius, gFrameCount);

                if (any(neighborPixel < 0) || any(neighborPixel >= gFrameDim)) continue;

                uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * gFrameDim.x);

                Reservoir neighborReservoir = gReservoirs[neighborIdx];

                if (neighborReservoir.M == 0u) continue;

                SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[neighborIdx]);
                if (!neighborSurfaceData.isValid()) continue;
                if (!isValidNeighbor(surfaceData, neighborSurfaceData)) continue;

                // 문제1. 지금 lightSample.Li 항에는 이미 Le * GeomeryTerm * cosTheta 가 곱해져 있음
                // 그리고 이 G * cos Theta는 이웃픽셀의 표면을 기준으로 Lighting 계산을 한 값임
                // 따라서, 이웃 픽셀의 조명 기여도를 현재 픽셀로 직접 재사용하는 것은 부정확함
                // 이웃 픽셀의 조명이 무엇인지만을 파악하여 다시 Li를 재계산해야 함. -> 해결

                // 재계산 로직
                LightSample tmp = {};

                bool validLight2 = reconstructLightSample(
                    surfaceData.posW, surfaceData.normal,
                    neighborReservoir.lightType, neighborReservoir.lightIndex, neighborReservoir.lightUV, lightConfig, tmp);

                float neighborTargetPdf = surfaceData.evalTargetPDF(tmp, viewVec);

                outputReservoir.update(neighborReservoir, tmp, neighborTargetPdf, sampleNext1D(sg));
            }

            if (outputReservoir.targetPdf > 0.f) outputReservoir.computeW();
        }
        else if (kBisedMode == 1) // UNBIASED NAIVE
        {
            uint Z = 0u;
            LightSample selectedSample = ls;

            for (uint i = 0; i < kSpatialReuseNeighbors; i++)
            {
                int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialReuseNeighbors, pixelRadius, gFrameCount);

                if (any(neighborPixel < 0) || any(neighborPixel >= gFrameDim)) continue;

                uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * gFrameDim.x);

                Reservoir neighborReservoir = gReservoirs[neighborIdx];

                if (neighborReservoir.M == 0u) continue;

                SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[neighborIdx]);
                if (!neighborSurfaceData.isValid()) continue;
                if (!isValidNeighbor(surfaceData, neighborSurfaceData)) continue;


                // 재계산 로직
                LightSample tmp = {};

                bool validLight2 = reconstructLightSample(
                    surfaceData.posW, surfaceData.normal,
                    neighborReservoir.lightType, neighborReservoir.lightIndex, neighborReservoir.lightUV, lightConfig, tmp );

                float neighborTargetPdf = surfaceData.evalTargetPDF(tmp, viewVec);

                bool valid = outputReservoir.update(neighborReservoir, tmp, neighborTargetPdf, sampleNext1D(sg));
                if(valid) selectedSample = tmp;
            }

            if (outputReservoir.targetPdf > 0.f)
            {
                Z += currentReservoir.M;
            }

            for (uint i = 0; i < kSpatialReuseNeighbors; i++)
            {
                int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialReuseNeighbors, pixelRadius, gFrameCount);

                if (any(neighborPixel < 0) || any(neighborPixel >= gFrameDim)) continue;

                uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * gFrameDim.x);

                Reservoir neighborReservoir = gReservoirs[neighborIdx];

                if (neighborReservoir.M == 0u) continue;

                SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[neighborIdx]);
                if (!neighborSurfaceData.isValid()) continue;
                if (!isValidNeighbor(surfaceData, neighborSurfaceData)) continue;

                LightSample tmp;

                bool validLight = reconstructLightSample(
                    neighborSurfaceData.posW, neighborSurfaceData.normal,
                    selectedSample.lightType, selectedSample.lightIndex, selectedSample.uv, lightConfig, tmp);

                float neighborTargetPdf = neighborSurfaceData.evalTargetPDF(tmp, viewVec);
                if (neighborTargetPdf > 0.f)
                {
                    Z += neighborReservoir.M;
                }
            }
            float m = Z > 0 ? 1.f / float(Z) : 0.f;
            if (outputReservoir.targetPdf > 0.f && m > 0 && outputReservoir.weightSum > 0.f) outputReservoir.W = (1.f / outputReservoir.targetPdf) * (outputReservoir.weightSum * m);
        }
        else if (kBisedMode == 2)
        {
            // UNBIASED MIS
            LightSample selectedSample;
            int selectedIdx = -1;
            float pSum = 0.f;
            float pStar = 0.f;

            if (outputReservoir.targetPdf > 0.f)
            {
                selectedSample = ls;
            }

            for (uint i = 0; i < kSpatialReuseNeighbors; i++)
            {
                int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialReuseNeighbors, pixelRadius, gFrameCount);

                if (any(neighborPixel < 0) || any(neighborPixel >= gFrameDim)) continue;

                uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * gFrameDim.x);

                Reservoir neighborReservoir = gReservoirs[neighborIdx];

                if (neighborReservoir.M == 0u) continue;

                SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[neighborIdx]);
                if (!neighborSurfaceData.isValid()) continue;
                if (!isValidNeighbor(surfaceData, neighborSurfaceData)) continue;

                // 재계산 로직
                LightSample tmp = {};

                bool validLight2 = reconstructLightSample(
                    surfaceData.posW, surfaceData.normal,
                    neighborReservoir.lightType, neighborReservoir.lightIndex, neighborReservoir.lightUV, lightConfig, tmp );

                float neighborTargetPdf = surfaceData.evalTargetPDF(tmp, viewVec);

                bool valid = outputReservoir.update(neighborReservoir, tmp, neighborTargetPdf, sampleNext1D(sg));
                if (valid)
                {
                    selectedSample = tmp;
                    selectedIdx = int(i);
                }
            }

            pStar = outputReservoir.targetPdf;
            pSum += outputReservoir.targetPdf * float(currentReservoir.M);

            for (uint i = 0; i < kSpatialReuseNeighbors; i++)
            {
                int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialReuseNeighbors, pixelRadius, gFrameCount);

                if (any(neighborPixel < 0) || any(neighborPixel >= gFrameDim)) continue;

                uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * gFrameDim.x);

                Reservoir neighborReservoir = gReservoirs[neighborIdx];

                if (neighborReservoir.M == 0u) continue;

                SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(gSurfaceData[neighborIdx]);
                if (!neighborSurfaceData.isValid()) continue;
                if (!isValidNeighbor(surfaceData, neighborSurfaceData)) continue;

                LightSample tmp;

                bool validLight = reconstructLightSample(
                    neighborSurfaceData.posW, neighborSurfaceData.normal,
                    selectedSample.lightType, selectedSample.lightIndex, selectedSample.uv, lightConfig, tmp );

                float neighborTargetPdf = neighborSurfaceData.evalTargetPDF(tmp, viewVec);
                pSum += neighborTargetPdf * neighborReservoir.M;
                if (selectedIdx == int(i))
                {
                    pStar = neighborTargetPdf;
                }
            }

            float m = pSum > 0.f ? pStar / pSum : 0.f;
            if (outputReservoir.targetPdf > 0.f && m > 0 && outputReservoir.weightSum > 0.f) outputReservoir.W = (1.f / outputReservoir.targetPdf) * (outputReservoir.weightSum * m);
        }
        gOutReservoirs[bufferIndex] = outputReservoir;
    }
};

cbuffer CB
{
    SpatialReusePass gSpatialReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gSpatialReusePass.execute(dispatchThreadId.xy);
}
