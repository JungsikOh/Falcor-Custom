import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
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

struct TemporalReusePass
{
    // Resources (using bindShaderData func)
    PathTracerParams params; ///< Runtime parameters.

    RWTexture2D<float4> outputColor; ///< Output color buffer if kSamplesPerPixel == 1.

    Texture2D<float2> motionVec;

    StructuredBuffer<PackedSurfaceData> surfaceData; ///< Input surface data buffer.
    RWStructuredBuffer<PackedReservoirGI> outputReservoirs; ///< Input reservoirs buffer.
    StructuredBuffer<PackedReservoirGI> prevReservoirs;   ///< Input previous frame reservoirs buffer.
    StructuredBuffer<PackedSurfaceData> prevSurfaceData; ///< Input previous frame surface data buffer.

    /// Configuration constants (can be made into shader parameters)
    static const float kNormalThreshold = 0.6f; // cos(~53°) - reject if normals differ too much
    static const float kDepthThreshold = 0.1f; // 10% relative depth difference allowed

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

    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        const uint bufferIndex = (pixel.y * params.frameDim.x) + pixel.x;

        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        float3 viewVec = -normalize(cameraRay.dir);

        ReservoirGI currReservoir = ReservoirGI::unpackReservoirGI(outputReservoirs[bufferIndex]);
        SurfaceData currSurface = SurfaceData::unpackSurfaceData(surfaceData[bufferIndex]);
        bool foundTemporalReservoir = true;

        if (!currSurface.isValid()) return;

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);

        float2 motionVectors = motionVec[pixel];
        int2 prevPixel = int2(float2(pixel) + motionVectors * float2(params.frameDim) + sampleNext2D(sg));

        if (any(prevPixel >= params.frameDim) || any(prevPixel < 0))
        {
            foundTemporalReservoir = false;
        }

        const uint prevBufferIndex = (prevPixel.y * params.frameDim.x) + prevPixel.x;

        SurfaceData prevSurface = SurfaceData::unpackSurfaceData(prevSurfaceData[prevBufferIndex]);
        if (!prevSurface.isValid() || !isValidNeighbor(currSurface, prevSurface) || !AreMaterialSimilar(currSurface, prevSurface))
        {
            foundTemporalReservoir = false;
        }

        ReservoirGI prevReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[prevBufferIndex]);

        prevReservoir.M = min(prevReservoir.M, 30);
        prevReservoir.age++; // Make the sample older

        if (prevReservoir.age > kMaxReservoirAge)
        {
            foundTemporalReservoir = false;
        }

        ReservoirGI outputReservoir = {};

        float currTargetPDF = currSurface.evalTargetPDF(currReservoir.sample, viewVec);
        outputReservoir.merge(currReservoir, currTargetPDF, /* random = */ 0.5, /* isCurrent = */ true);

        if (foundTemporalReservoir)
        {
            float prevTargetPDF = currSurface.evalTargetPDF(prevReservoir.sample, viewVec);

            float3 q2 = prevReservoir.sample.samplePoint;
            float3 q1 = prevReservoir.sample.surfacePoint; // neighbor surface
            float3 r1 = currSurface.posW;                      // current surface

            float3 q1q2 = q2 - q1; // q1 → sample point
            float3 r1q2 = q2 - r1; // r1 → sample point (FIXED: was using currReservoir.sample.surfacePoint)

            float q1q2LenSq = dot(q1q2, q1q2);
            float r1q2LenSq = dot(r1q2, r1q2);

            // Backface check: sample must be visible from current surface
            if (dot(currSurface.normal, normalize(r1q2)) <= 0.f)
            {
                prevTargetPDF = 0.f;
            }

            if (prevTargetPDF > 0.f)
            {
                float3 q1q2Dir = normalize(q1q2);
                float3 r1q2Dir = normalize(r1q2);

                float cosPhiA = -dot(prevReservoir.sample.sampleNormal, r1q2Dir); // cos at sample, dir to r1
                float cosPhiB = -dot(prevReservoir.sample.sampleNormal, q1q2Dir); // cos at sample, dir to q1

                // Jacobian = (|r1q2|² / |q1q2|²) * (cosPhiB / cosPhiA)
                // But standard ReSTIR GI Jacobian is: J(q1→r1) = (cosPhiA / cosPhiB) * (q1q2LenSq / r1q2LenSq)
                float jacobian = (cosPhiB <= 0.f || r1q2LenSq <= 0.f) ? 0.f :
                    clamp((q1q2LenSq * cosPhiA) / (r1q2LenSq * cosPhiB), 0.f, 1e10f);
                prevTargetPDF *= jacobian;
            }

            outputReservoir.merge(prevReservoir, prevTargetPDF, sampleNext1D(sg));
        }

        print("currentReservoir.targetPDF: ", currTargetPDF);
        print("currentReservoir.sourcePDF: ", currReservoir.sample.sourcePdf);
        print("currentReservoir.weightSum: ", currReservoir.weightSum);
        print("currentReservoir.M: ", currReservoir.M);

        print("temporalOutputReservoir.weightSum: ", outputReservoir.weightSum);
        print("temporalOutputReservoir.W: ", outputReservoir.W);
        print("temporalOutputReservoir.M: ", outputReservoir.M);
        print("temporalOutputReservoir.age: ", outputReservoir.age);

        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / outputReservoir.W : 0.f;
        outputReservoir.W = clamp(outputReservoir.W, 0.f, 1e20f);

        outputReservoirs[bufferIndex] = outputReservoir.pack();
    }
}

cbuffer CB
{
    TemporalReusePass gTemporalReusePass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    printSetPixel(dispatchThreadId.xy);
    logSetPixel(dispatchThreadId.xy);
    gTemporalReusePass.execute(dispatchThreadId.xy);
}
