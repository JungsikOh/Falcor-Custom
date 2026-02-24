#include "Utils/Math/MathConstants.slangh"
#include "Scene/SceneDefines.slangh"

__exported import Scene.Shading;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;

import Scene.RaytracingInline;
import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Rendering.Utils.PixelStats;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import LoadShadingData;
import NRDHelpers;
import PathTracer;
import PathState;
import ColorType;
import Params;

import SurfaceData;
import ReservoirGI;

struct SpatialReusePass
{
    // Resources (using bindShaderData func)
    PathTracerParams params; ///< Runtime parameters.

    RWTexture2D<float4> outputColor; ///< Output color buffer if kSamplesPerPixel == 1.

    StructuredBuffer<PackedSurfaceData> surfaceData; ///< Input surface data buffer.
    StructuredBuffer<PackedReservoirGI> reservoirs;   ///< Input previous frame reservoirs buffer.
    RWStructuredBuffer<PackedReservoirGI> outputReservoirs; ///< Input/output reservoirs buffer.

    /// Neighbor validation for spatial reuse
    /// Checks geometric similarity between current pixel and neighbor pixel
    /// to avoid reusing samples across discontinuities (edges, depth gaps, etc.)

    /// Configuration constants (can be made into shader parameters)
    static const float kNormalThreshold = 0.6f; // cos(~53°) - reject if normals differ too much
    static const float kDepthThreshold = 0.10f; // 10% relative depth difference allowed

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
        float neighborDepth)
    {
        return isValidNeighbor(centerNormal, neighborNormal, centerDepth, neighborDepth, kNormalThreshold, kDepthThreshold);
    }

    /// Extended neighbor validation with configurable thresholds
    bool isValidNeighbor(
        float3 centerNormal,
        float3 neighborNormal,
        float centerDepth,
        float neighborDepth,
        float normalThreshold,
        float depthThreshold)
    {
        // Invalid depth check
        if (centerDepth <= 0.f || neighborDepth <= 0.f)
            return false;

        // Normal similarity test: dot(n1, n2) = cos(angle between normals)
        if (dot(centerNormal, neighborNormal) < normalThreshold)
            return false;

        // Depth similarity test (relative difference)
        if (abs(centerDepth / neighborDepth - 1.f) > depthThreshold)
            return false;

        return true;
    }

    /// Full neighbor validation using SurfaceData structures
    bool isValidNeighbor(SurfaceData center, SurfaceData neighbor)
    {
        if (!center.isValid() || !neighbor.isValid())
            return false;

        return isValidNeighbor(center.normal, neighbor.normal, center.depthW, neighbor.depthW);
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

    bool evalSampleVisibility(float3 pos, float3 samplePoint)
    {
        float3 posToSample = samplePoint - pos;
        float distance = length(posToSample);
        float3 dir = posToSample / distance;

        const Ray ray = Ray(pos, dir, 0.f, distance);
        SceneRayQuery<1> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        const uint bufferIndex = (pixel.y * params.frameDim.x) + pixel.x;

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        float3 viewVec = -normalize(cameraRay.dir);

        ReservoirGI currReservoir = ReservoirGI::unpackReservoirGI(reservoirs[bufferIndex]);
        SurfaceData currSurface = SurfaceData::unpackSurfaceData(surfaceData[bufferIndex]);

        if (!currSurface.isValid()) return;

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);
        float3 rayOrigin = computeRayOrigin(currSurface.posW, currSurface.normal);

        // Spatial reuse implementation goes here.
        const uint spatialMaximumM = 30;    // Maximum M is a maximum M of temporal reservoir.
        float kSearchRadius = params.frameDim.x * 0.1f;
        const float kRadiusShrinkRatio = 0.5f; // Shrink by 50% each iteration
        const float kMinSearchRadius = 3.0f;   // Minimum radius in pixels
        const uint kMinTemporalM = 0;          // Minimum M in temporal reservoir to consider neighbor
        const uint kSpatialSamples = currReservoir.M < (spatialMaximumM / 2) ? 9 : 3;

        ReservoirGI outputReservoir = {};
        float currTargetPDF = currSurface.evalTargetPDF(currReservoir.sample, viewVec);
        outputReservoir.merge(currReservoir, currTargetPDF, /* random = */ 0.5, /* isCurrent = */ true);

#if USE_SPATIAL_UNBIASED
        uint validNeighborFlags = 0;
        float radiusCache[9]; // Max 9 samples
#endif
        for (uint i = 0; i < kSpatialSamples; ++i)
        {
            int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialSamples, kSearchRadius, params.frameCount);

            if (any(neighborPixel >= params.frameDim) || any(neighborPixel < 0))
            {
                kSearchRadius = max(kSearchRadius * kRadiusShrinkRatio, kMinSearchRadius);
                continue;
            }

            uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * params.frameDim.x);
            ReservoirGI neighborReservoir = ReservoirGI::unpackReservoirGI(reservoirs[neighborIdx]);
            if (neighborReservoir.M <= kMinTemporalM)
            {
                kSearchRadius = max(kSearchRadius * kRadiusShrinkRatio, kMinSearchRadius);
                continue;
            }

            SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(surfaceData[neighborIdx]);
            if (!neighborSurfaceData.isValid() || !isValidNeighbor(currSurface, neighborSurfaceData) || !AreMaterialSimilar(currSurface, neighborSurfaceData))
            {
                kSearchRadius = max(kSearchRadius * kRadiusShrinkRatio, kMinSearchRadius);
                continue;
            }

            float neighborReservoirTargetPDF = currSurface.evalTargetPDF(neighborReservoir.sample, viewVec);

            // Jacobian correction:
            // q1 = neighbor surface point, q2 = sample point, r1 = current surface point
            // We need to re-weight the sample from q1's solid angle to r1's solid angle
            float3 q2 = neighborReservoir.sample.samplePoint;
            float3 q1 = neighborReservoir.sample.surfacePoint; // neighbor surface
            float3 r1 = currSurface.posW;                      // current surface

            float3 q1q2 = q2 - q1; // q1 → sample point
            float3 r1q2 = q2 - r1; // r1 → sample point (FIXED: was using currReservoir.sample.surfacePoint)

            float q1q2LenSq = dot(q1q2, q1q2);
            float r1q2LenSq = dot(r1q2, r1q2);

            // Backface check: sample must be visible from current surface
            if (dot(currSurface.normal, normalize(r1q2)) <= 0.f)
            {
                neighborReservoirTargetPDF = 0.f;
            }

            if (neighborReservoirTargetPDF > 0.f)
            {
                float3 q1q2Dir = normalize(q1q2);
                float3 r1q2Dir = normalize(r1q2);

                float cosPhiA = dot(neighborReservoir.sample.sampleNormal, -r1q2Dir); // cos at sample, dir to r1
                float cosPhiB = dot(neighborReservoir.sample.sampleNormal, -q1q2Dir); // cos at sample, dir to q1

                // Jacobian = (|r1q2|² / |q1q2|²) * (cosPhiB / cosPhiA)
                // But standard ReSTIR GI Jacobian is: J(q1→r1) = (cosPhiA / cosPhiB) * (q1q2LenSq / r1q2LenSq)
                float jacobian = (cosPhiB <= 0.f || r1q2LenSq <= 0.f) ? 0.f :
                    clamp((q1q2LenSq * cosPhiA) / (r1q2LenSq * cosPhiB), 0.f, 1e20f);
                neighborReservoirTargetPDF *= jacobian;
            }

            bool isVisible = evalSampleVisibility(rayOrigin, neighborReservoir.sample.samplePoint);
            if (!isVisible) neighborReservoirTargetPDF = 0.f;

            outputReservoir.merge(neighborReservoir, neighborReservoirTargetPDF, sampleNext1D(sg));

#if USE_SPATIAL_UNBIASED
            radiusCache[i] = kSearchRadius;
            validNeighborFlags |= (1u << i);
#endif
            kSearchRadius = max(kSearchRadius * kRadiusShrinkRatio, kMinSearchRadius);
        }

#if USE_SPATIAL_UNBIASED
        float Z = 0.f;
        bool isCurrVisible = evalSampleVisibility(rayOrigin, outputReservoir.sample.samplePoint);
        if (isCurrVisible) Z += currReservoir.M;

        for (uint i = 0; i < kSpatialSamples; ++i)
        {
            if ((validNeighborFlags & (1u << i)) == 0u) continue;
            int2 neighborPixel = getNeighborPixelLowDiscrepancy(pixel, i, kSpatialSamples, radiusCache[i], params.frameCount);

            uint neighborIdx = uint(neighborPixel.x + neighborPixel.y * params.frameDim.x);
            ReservoirGI neighborReservoir = ReservoirGI::unpackReservoirGI(reservoirs[neighborIdx]);
            SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(surfaceData[neighborIdx]);

            rayOrigin = computeRayOrigin(neighborSurfaceData.posW, neighborSurfaceData.normal);
            bool visible = evalSampleVisibility(rayOrigin, outputReservoir.sample.samplePoint);

            if (visible)
            {
                Z += neighborReservoir.M;
            }
        }
        float m = Z > 0.f ? 1.f / Z : 0.f;
        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum * m) / outputReservoir.W : 0.f;
#else
        float pNew = currSurface.evalTargetPDF(outputReservoir.sample, viewVec);
        print("outputReservoir.pNew(jacobian): ", outputReservoir.W);
        print("outputReservoir.pNew: ", pNew);
        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / outputReservoir.W : 0.f;
#endif
        print("outputReservoir.weightSum: ", outputReservoir.weightSum);
        print("outputReservoir.W: ", outputReservoir.W);
        print("outputReservoir.M: ", outputReservoir.M);

        outputReservoir.W = clamp(outputReservoir.W, 0.f, 1e20f);
        outputReservoir.age++;
        outputReservoir.M = min(outputReservoir.M, 300);

        outputReservoirs[bufferIndex] = outputReservoir.pack();
    }
}

cbuffer CB
{
    SpatialReusePass gSpatialReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    logSetPixel(dispatchThreadId.xy);
    gSpatialReusePass.execute(dispatchThreadId.xy);
}
