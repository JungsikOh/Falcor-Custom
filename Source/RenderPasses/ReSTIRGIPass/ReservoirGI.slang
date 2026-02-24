#include "Utils/Math/MathConstants.slangh"
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Materials.IBSDF;
import Utils.Math.PackedFormats;

struct PackedSampleGI
{
    uint4 surfaceGeometry;
    uint4 sampleGeometry;

    uint Le;
    uint sourcePdf;
    uint valid;
    uint _pad0 = 0;
};

// 68B / 36B
struct SampleGI
{
    float3 surfacePoint = float3(0.f);
    float3 surfaceNormal = float3(0.f);
    float3 samplePoint = float3(0.f);
    float3 sampleNormal = float3(0.f);
    float3 Le = float3(0.f);
    float sourcePdf = 0.f;
    uint valid = 0;

    static SampleGI unpackSampleGI(const PackedSampleGI packed)
    {
        SampleGI sample;

        sample.surfacePoint = asfloat(packed.surfaceGeometry.xyz);
        sample.surfaceNormal = decodeNormal2x16(packed.surfaceGeometry.w);
        sample.samplePoint = asfloat(packed.sampleGeometry.xyz);
        sample.sampleNormal = decodeNormal2x16(packed.sampleGeometry.w);
        sample.Le = decodeLogLuvHDR(packed.Le);
        sample.sourcePdf = asfloat(packed.sourcePdf);
        sample.valid = packed.valid;

        return sample;
    }

    PackedSampleGI pack()
    {
        PackedSampleGI packedSample = {};

        packedSample.surfaceGeometry.xyz = asuint(this.surfacePoint);
        packedSample.surfaceGeometry.w = encodeNormal2x16(this.surfaceNormal);
        packedSample.sampleGeometry.xyz = asuint(this.samplePoint);
        packedSample.sampleGeometry.w = encodeNormal2x16(this.sampleNormal);
        packedSample.Le = encodeLogLuvHDR(this.Le);
        packedSample.sourcePdf = asuint(this.sourcePdf);
        packedSample.valid = this.valid;

        return packedSample;
    }
};

struct PackedReservoirGI
{
    PackedSampleGI sample;

    uint weightSum;
    uint W;
    uint age;
    uint M;
};

struct ReservoirGI
{
    SampleGI sample;
    float weightSum;
    float W;
    uint age;
    uint M;

    __init()
    {
        sample = {};
        weightSum = 0.f;
        W = 0.f;
        age = 0;
        M = 0;
    }

    [mutating]
    bool update(const SampleGI lightSample, const float targetPDF, const float rnd)
    {
        float weight = targetPDF / lightSample.sourcePdf;
        weightSum += weight;
        M += 1;

        if (rnd * weightSum < weight)
        {
            sample = lightSample;
            W = targetPDF;
            return true;
        }
        return false;
    }

    [mutating]
    bool merge(const ReservoirGI other, const float targetPdf, const float rnd, bool isCurrent = false)
    {
        float weight = targetPdf * other.W * other.M; // W = 1/M * 1/q * sum of (q/p)
        weightSum += weight;
        M += other.M;

        if (rnd * weightSum < weight)
        {
            sample.Le = other.sample.Le;
            sample.samplePoint = other.sample.samplePoint;
            sample.sampleNormal = other.sample.sampleNormal;
            W = targetPdf;
            age = other.age;

            // When isCurrent, copy full surface geometry from the current reservoir.
            if (isCurrent)
            {
                sample.surfacePoint = other.sample.surfacePoint;
                sample.surfaceNormal = other.sample.surfaceNormal;
                sample.sourcePdf = other.sample.sourcePdf;
                sample.valid = other.sample.valid;
            }

            sample.valid = sample.valid || other.sample.valid;
            return true;
        }
        return false;
    }

    static ReservoirGI unpackReservoirGI(const PackedReservoirGI packed)
    {
        ReservoirGI reservoir;
        reservoir.sample = SampleGI::unpackSampleGI(packed.sample);
        reservoir.weightSum = asfloat(packed.weightSum);
        reservoir.W = asfloat(packed.W);
        reservoir.age = packed.age;
        reservoir.M = packed.M;

        // Guard against degenerate weights from NaN/Inf.
        if (isinf(reservoir.W) || isnan(reservoir.W))
        {
            reservoir.W = 0.f;
            reservoir.M = 0;
        }

        return reservoir;
    }

    PackedReservoirGI pack()
    {
        PackedReservoirGI p = {};
        p.sample = sample.pack();
        p.weightSum = asuint(weightSum);
        p.W = asuint(W);
        p.age = age;
        p.M = M;
        return p;
    }
};
