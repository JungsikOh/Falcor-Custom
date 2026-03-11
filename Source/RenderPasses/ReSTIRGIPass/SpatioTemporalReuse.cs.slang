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

/// Combined spatio-temporal reuse pass following RTXDI GI reference.
/// Single unified loop: initial sample + temporal + spatial neighbors merge into one reservoir.
/// Single normalization at the end (biased 1/M or unbiased MIS).
/// selectedTargetPdf tracked WITHOUT Jacobian for correct normalization.
struct GIResamplingPass
{
    // ---- Resources (bound from C++ via bindShaderData) ----
    PathTracerParams params;

    Texture2D<float2> motionVec;

    StructuredBuffer<PackedSurfaceData> surfaceData;        ///< Current frame surface data.
    StructuredBuffer<PackedSurfaceData> prevSurfaceData;    ///< Previous frame surface data.

    StructuredBuffer<PackedReservoirGI> inputReservoirs;    ///< Current frame initial reservoirs (from trace pass).
    StructuredBuffer<PackedReservoirGI> prevReservoirs;     ///< Previous frame fully resampled reservoirs.
    RWStructuredBuffer<PackedReservoirGI> outputReservoirs; ///< Output reservoirs.

    // ---- Configuration constants ----
    static const float kNormalThreshold = 0.9f;     ///< cos(~25 deg) for neighbor rejection.
    static const float kDepthThreshold = 0.1f;     ///< 10% relative depth diff allowed.
    static const float kJacobianClamp = 10.f;       ///< Max jacobian to avoid fireflies.
    static const float kLargeFloat = 50.f;

    static const uint kTemporalMaxM = 30;
    static const uint kSpatialMaxM = 500;

    // ---- Neighbor validation ----
    bool isValidNeighbor(SurfaceData center, SurfaceData neighbor)
    {
        if (!center.isValid() || !neighbor.isValid())
            return false;

        if (center.depthW <= 0.f || neighbor.depthW <= 0.f)
            return false;

        if (dot(center.normal, neighbor.normal) < kNormalThreshold)
            return false;

        if (abs(center.depthW / neighbor.depthW - 1.f) > kDepthThreshold)
            return false;

        return true;
    }

    // ---- Visibility ray ----
    bool evalSegmentVisibility(float3 pos, float3 endPoint)
    {
        float3 offset = endPoint - pos;
        float tMax = length(offset);
        if (tMax < 1e-6f) return true;
        float3 dir = offset / tMax;

        const Ray ray = Ray(pos, dir, 0.001f, 0.999f * tMax);
        SceneRayQuery<1> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        const uint bufferIndex = pixel.y * params.frameDim.x + pixel.x;

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        float3 viewVec = -normalize(cameraRay.dir);

        // Load current-frame surface.
        SurfaceData currSurface = SurfaceData::unpackSurfaceData(surfaceData[bufferIndex]);
        if (!currSurface.isValid()) return;

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);
        float3 rayOrigin = computeRayOrigin(currSurface.posW, currSurface.normal);

        // ================================================================
        // Phase 1: Temporal Reuse
        // ================================================================

        ReservoirGI temporalReservoir = {};
        float temporalSelectedTargetPdf = 0.f;

        ReservoirGI initialReservoir = ReservoirGI::unpackReservoirGI(inputReservoirs[bufferIndex]);
        float currTargetPdf = currSurface.evalTargetPDF(initialReservoir.sample, viewVec);
        bool updated = temporalReservoir.update(initialReservoir.sample, currTargetPdf, /* random = */ 0.5);
        if (updated) temporalSelectedTargetPdf = currTargetPdf;

        float2 mv = motionVec[pixel];
        float2 prevPixelF = float2(pixel) + mv * float2(params.frameDim) + (sampleNext1D(sg) * 0.5f);
        int2 prevPixel = int2(prevPixelF);
        bool founndPrevReservoir = false;

        const uint prevIdx = uint(prevPixel.y) * params.frameDim.x + uint(prevPixel.x);

#if USE_TEMPORAL_REUSE
        if (params.frameCount > 0
            && all(prevPixel >= 0)
            && all(prevPixel < int2(params.frameDim)))
        {
            SurfaceData prevSurface = SurfaceData::unpackSurfaceData(prevSurfaceData[prevIdx]);

            if (prevSurface.isValid()
                && isValidNeighbor(currSurface, prevSurface)
                && AreMaterialSimilar(currSurface, prevSurface))
            {
                ReservoirGI prevReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[prevIdx]);

                if (prevReservoir.sample.valid != 0
                    && prevReservoir.M > 0
                    && prevReservoir.age <= kMaxReservoirAge)
                {
                    founndPrevReservoir = true;
                    prevReservoir.M = min(prevReservoir.M, kTemporalMaxM); // Temporal M clamp.
                    ++prevReservoir.age;                        // One frame older.

                    float prevTargetPdf = currSurface.evalTargetPDF(prevReservoir.sample, viewVec);
                    float jacobian = 1.0f;

                    // --- Jacobian for temporal (currently disabled) ---
                    // float3 q2 = prevReservoir.sample.samplePoint;
                    // float3 q1 = prevSurface.posW;
                    // float3 r1 = currSurface.posW;
                    // float3 q1q2 = q2 - q1;
                    // float3 r1q2 = q2 - r1;
                    // float q1q2LenSq = dot(q1q2, q1q2);
                    // float r1q2LenSq = dot(r1q2, r1q2);
                    // if (dot(currSurface.normal, normalize(r1q2)) <= 0.f)
                    //     prevTargetPdf = 0.f;
                    // if (prevTargetPdf > 0.f)
                    // {
                    //     float3 q1q2Dir = normalize(q1q2);
                    //     float3 r1q2Dir = normalize(r1q2);
                    //     float cosPhiA = dot(prevReservoir.sample.sampleNormal, -r1q2Dir);
                    //     float cosPhiB = dot(prevReservoir.sample.sampleNormal, -q1q2Dir);
                    //     jacobian = (cosPhiB <= 0.f || r1q2LenSq <= 0.f) ? 0.f :
                    //         clamp((q1q2LenSq * cosPhiA) / (r1q2LenSq * cosPhiB), 0.f, kJacobianClamp);
                    // }
                    // if (jacobian <= 0.f) prevTargetPdf = 0.f;

                    updated = temporalReservoir.merge(prevReservoir, prevTargetPdf * jacobian, sampleNext1D(sg));
                    if (updated) temporalSelectedTargetPdf = prevTargetPdf;
                }
            }
        }
#endif

        temporalReservoir.W = (temporalSelectedTargetPdf > 0.f && temporalReservoir.M > 0) ?
            temporalReservoir.weightSum / (float(temporalReservoir.M) * temporalSelectedTargetPdf) : 0.f;
        temporalReservoir.W = clamp(temporalReservoir.W, 0.f, kLargeFloat);

        // ================================================================
        // Phase 2: Spatial Reuse
        // ================================================================

        ReservoirGI outputReservoir = {};
        float selectedTargetPdf = 0.f;

        float tempTargetPdf = currSurface.evalTargetPDF(temporalReservoir.sample, viewVec);
        updated = outputReservoir.merge(temporalReservoir, tempTargetPdf, /* random = */ 0.5);
        if (updated) selectedTargetPdf = tempTargetPdf;

#if USE_SPATIAL_REUSE
        ReservoirGI spatialReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[prevIdx]);

         // Spatial reuse parameters
        float searchRadius = params.frameDim.x * 0.1f;
        const float kRadiusShrinkRatio = 0.5f;
        const float kMinSearchRadius = 3.0f;
        const uint kSpatialSamples = spatialReservoir.M < (kSpatialMaxM / 2) ? 9 : 3;

        // for unbiased
        uint validNeighborIndices[9];
        uint validNeighborFlags = 0;

        for (uint i = 0; i < kSpatialSamples; i++)
        {
            float2 randOffset = sampleNext2D(sg) * 2.f - 1.f; // [-1, 1]
            float2 neighborPixelF = float2(pixel) + randOffset * searchRadius;
            int2 neighborPixel = int2(clamp(neighborPixelF, float2(0), float2(params.frameDim - 1)));

            const uint neighborIdx = uint(neighborPixel.y) * params.frameDim.x + uint(neighborPixel.x);

            SurfaceData neighborSurface = SurfaceData::unpackSurfaceData(prevSurfaceData[neighborIdx]);
            if (!neighborSurface.isValid()
                || !isValidNeighbor(currSurface, neighborSurface)
                || !AreMaterialSimilar(currSurface, neighborSurface))
            {
                searchRadius = max(searchRadius * kRadiusShrinkRatio, kMinSearchRadius);
                continue;
            }

            ReservoirGI neighborReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[neighborIdx]);
            if (neighborReservoir.M <= 0 || neighborReservoir.sample.valid == 0)
            {
                searchRadius = max(searchRadius * kRadiusShrinkRatio, kMinSearchRadius);
                continue;
            }

            float neighborTargetPdf = currSurface.evalTargetPDF(neighborReservoir.sample, viewVec);
            float jacobian = 1.f;

            // --- Jacobian for spatial (currently disabled) ---
            float3 q2 = neighborReservoir.sample.samplePoint;
            float3 q1 = neighborSurface.posW;
            float3 r1 = currSurface.posW;
            float3 q1q2 = q2 - q1;
            float3 r1q2 = q2 - r1;
            float q1q2LenSq = dot(q1q2, q1q2);
            float r1q2LenSq = dot(r1q2, r1q2);
            if (q1q2LenSq < 1e-8f || r1q2LenSq < 1e-8f) neighborTargetPdf = 0.f;
            if (dot(currSurface.normal, normalize(r1q2)) <= 0.f) neighborTargetPdf = 0.f;
            if (neighborTargetPdf > 0.f)
            {
                float3 q1q2Dir = normalize(q1q2);
                float3 r1q2Dir = normalize(r1q2);
                float cosPhiA = dot(neighborReservoir.sample.sampleNormal, -r1q2Dir);
                float cosPhiB = dot(neighborReservoir.sample.sampleNormal, -q1q2Dir);
                jacobian = (cosPhiB <= 0.f || r1q2LenSq <= 0.f) ? 0.f :
                    clamp((q1q2LenSq * cosPhiA) / (r1q2LenSq * cosPhiB), 0.f, kJacobianClamp);
            }

            bool isVisible = evalSegmentVisibility(rayOrigin, neighborReservoir.sample.samplePoint);
            if (!isVisible) neighborTargetPdf = 0.f;

            updated = outputReservoir.merge(neighborReservoir, neighborTargetPdf * jacobian, sampleNext1D(sg));
            if (updated) selectedTargetPdf = neighborTargetPdf;

            validNeighborIndices[i] = neighborIdx;
            validNeighborFlags |= (1u << i);
        }
#endif

#if USE_SPATIAL_UNBIASED
        uint Z = temporalReservoir.M;

        for (uint i = 0; i < kSpatialSamples; ++i)
        {
            if ((validNeighborFlags & (1u << i)) == 0u) continue;

            uint neighborIdx = validNeighborIndices[i];

            ReservoirGI neighborReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[neighborIdx]);
            SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(prevSurfaceData[neighborIdx]);

            rayOrigin = computeRayOrigin(neighborSurfaceData.posW, neighborSurfaceData.normal);
            bool visible = evalSegmentVisibility(rayOrigin, outputReservoir.sample.samplePoint);

            if (visible)
            {
                Z += neighborReservoir.M;
            }
        }
        float m = Z > 0.f ? float(Z) : 0.f;
        outputReservoir.W = selectedTargetPdf > 0.f ?
            outputReservoir.weightSum / (m * selectedTargetPdf) : 0.f;
#elif USE_PAIRWISE_MIS
        float3 directionVec = outputReservoir.sample.samplePoint - currSurface.posW;
        float chosenWeight = saturate(dot(currSurface.normal, normalize(directionVec))) * luminance(outputReservoir.sample.Le);
        float totalWeight = temporalReservoir.M * chosenWeight;

        for (uint i = 0; i < kSpatialSamples; ++i)
        {
            if ((validNeighborFlags & (1u << i)) == 0u) continue;

            uint neighborIdx = validNeighborIndices[i];

            ReservoirGI neighborReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[neighborIdx]);
            SurfaceData neighborSurfaceData = SurfaceData::unpackSurfaceData(prevSurfaceData[neighborIdx]);

            directionVec = outputReservoir.sample.samplePoint - neighborSurfaceData.posW;
            rayOrigin = computeRayOrigin(neighborSurfaceData.posW, neighborSurfaceData.normal);
            bool visible = evalSegmentVisibility(rayOrigin, outputReservoir.sample.samplePoint);

            if (visible)
            {
                float misWeight = saturate(dot(neighborSurfaceData.normal, normalize(directionVec))) * luminance(outputReservoir.sample.Le);
                totalWeight += neighborReservoir.M * misWeight;

                float tempPdf = currSurface.evalTargetPDF(neighborReservoir.sample, viewVec);
                if (selectedTargetPdf == tempPdf)
                {
                    chosenWeight = misWeight;
                }
            }
        }
        float m = totalWeight > 0.f ? chosenWeight / totalWeight : 0.f;
        outputReservoir.W = selectedTargetPdf > 0.f ?
            (outputReservoir.weightSum * m) / selectedTargetPdf : 0.f;
#else
        outputReservoir.W = selectedTargetPdf > 0.f && outputReservoir.M > 0 ?
            outputReservoir.weightSum / (float(outputReservoir.M) * selectedTargetPdf) : 0.f;
#endif
        outputReservoir.W = clamp(outputReservoir.W, 0.f, kLargeFloat);
        outputReservoir.M = min(outputReservoir.M, kSpatialMaxM);

        // Record current surface
        outputReservoir.sample.surfacePoint = currSurface.posW;
        outputReservoir.sample.surfaceNormal = currSurface.normal;

        outputReservoirs[bufferIndex] = outputReservoir.pack();
    }
}

cbuffer CB
{
    GIResamplingPass gGIResamplingPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    logSetPixel(dispatchThreadId.xy);
    gGIResamplingPass.execute(dispatchThreadId.xy);
}
