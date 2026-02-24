import Utils.Attributes;
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
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

struct GIResamplingPass
{
    // Resources (using bindShaderData func)
    PathTracerParams params; ///< Runtime parameters.

    RWTexture2D<float4> outputColor; ///< Output color buffer if kSamplesPerPixel == 1.

    Texture2D<float2> motionVec;

    RWStructuredBuffer<PackedReservoirGI> initialSamples;

    // Ping-pong buffers storing gather point samples.
    RWStructuredBuffer<PackedReservoirGI> reservoirs; ///< Input reservoirs buffer.
    StructuredBuffer<PackedReservoirGI> prevReservoirs;   ///< Input previous frame reservoirs buffer.

    StructuredBuffer<PackedSurfaceData> surfaceData;     ///< Input surface data buffer.
    StructuredBuffer<PackedSurfaceData> prevSurfaceData; ///< Input previous frame surface data buffer.

    /// Configuration constants (can be made into shader parameters)
    static const float kNormalThreshold = 0.9f; // cos(~25°) - reject if normals differ too much
    static const float kDepthThreshold = 0.05f; // 5% relative depth difference allowed

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
        if (any(pixel >= params.frameDim)) return;

        const uint pixelIndex = (pixel.y * params.frameDim.x) + pixel.x;

        ReservoirGI initialSample = ReservoirGI::unpackReservoirGI(initialSamples[pixelIndex]);
        SurfaceData currSurface = SurfaceData::unpackSurfaceData(surfaceData[pixelIndex]);

        if (!currSurface.isValid()) return;

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, params.frameCount);

        float2 motionVectors = motionVec[pixel];
        int2 prevPixel = int2(float2(pixel) + motionVectors * float2(params.frameDim) + sampleNext2D(sg));

        bool isPrevValid = currSurface.isValid() && (any(prevPixel < params.frameDim) && all(prevPixel >= 0));

        const uint prevPixelIndex = (prevPixel.y * params.frameDim.x) + prevPixel.x;

        ReservoirGI temporalReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[prevPixelIndex]);
        if(length(temporalReservoir.sample.surfacePoint - currSurface.posW) > 1.f) isPrevValid = false;

        temporalReservoir.M = clamp(temporalReservoir.M, 0, 30);
        if(!isPrevValid) temporalReservoir.M = 0;

        ReservoirGI outputReservoir = {};

        float pNew = currSurface.evalTargetPDF(initialSample.sample, currSurface.viewDir);
        temporalReservoir.merge(initialSample, pNew, sampleNext1D(sg));

        pNew = currSurface.evalTargetPDF(temporalReservoir.sample, currSurface.viewDir);
        temporalReservoir.W = pNew > 0.f ? (temporalReservoir.weightSum / temporalReservoir.M) / pNew : 0.f;

        print("temporalReservoir.weightSum", temporalReservoir.weightSum);
        print("pNew", pNew);

        temporalReservoir.M = clamp(temporalReservoir.M, 0, 30);




        SurfaceData prevSurface = SurfaceData::unpackSurfaceData(prevSurfaceData[prevPixelIndex]);
        if (!prevSurface.isValid() || !isValidNeighbor(currSurface, prevSurface))
        {
            return;
        }

        ReservoirGI prevReservoir = ReservoirGI::unpackReservoirGI(prevReservoirs[prevBufferIndex]);
        //ReservoirGI prevReservoir = prevReservoirs[prevBufferIndex];
        prevReservoir.M = min(prevReservoir.M, 30);
        float currTargetPDF = 0.f;
        float prevTargetPDF = 0.f;

        currTargetPDF = currReservoir.W > 0.f ? currSurface.evalTargetPDF(currReservoir.sample, currSurface.viewDir) : 0.f;
        prevTargetPDF = currSurface.evalTargetPDF(prevReservoir.sample, currSurface.viewDir);

        //outputReservoir.merge(currReservoir, currTargetPDF, sampleNext1D(sg));
        outputReservoir.merge(prevReservoir, prevTargetPDF, sampleNext1D(sg));

        outputReservoir.W = outputReservoir.W > 0.f ? (outputReservoir.weightSum / outputReservoir.M) / outputReservoir.W : 0.f;


        // outputReservoir.M = min(outputReservoir.M, 30);
        outputReservoirs[bufferIndex] = outputReservoir.pack();

        outputColor[pixel] = float4(outputReservoir.sample.Le * outputReservoir.W, 1.0f);
    }
}

cbuffer CB
{
    GIResamplingPass gGIResamplingPass;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gGIResamplingPass.execute(dispatchThreadId.xy);
}
