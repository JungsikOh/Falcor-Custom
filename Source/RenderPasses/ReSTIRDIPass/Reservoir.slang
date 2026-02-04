#include "Utils/Math/MathConstants.slangh"

import Rendering.Materials.IBSDF;
import Utils.Sampling.TinyUniformSampleGenerator;
import Scene.RaytracingInline;
import Utils.Color.ColorHelpers;
import Rendering.Materials.Microfacet;
import Rendering.Materials.Fresnel;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Geometry.GeometryHelpers;

import LightSample;
import StaticParams;

// /// Packed Reservoir structure (52 bytes → 40 bytes 최적화 가능)
// struct PackedReservoir
// {
//     // Compressed values (HDR colors, normalized directions)
//     uint packedLi;          // R11G11B10 (4 bytes)
//     uint packedDir;         // Octahedral 2x16 snorm (4 bytes)
//     uint packedBsdf;        // R11G11B10 (4 bytes)
//     uint packedLightNormal; // Octahedral 2x16 snorm (4 bytes)

//     /// 16

//     // Full precision values (정밀도 필요 - 압축 안 함)
//     float targetPdf; // (4 bytes) - RIS 가중치 계산에 필수
//     float weightSum; // (4 bytes) - 누적 가중치
//     float W;         // (4 bytes) - 최종 가중치
//     float3 lightPosW; // (12 bytes) - 광원 위치 (광원 샘플링에 필요)
//     float3 lightOrigin;  // (12 bytes) - 광원 위치 (geometry term 계산)

//     /// 36

//     // Integer values
//     uint M;         // (4 bytes) - 샘플 카운트
//     uint lightType; // (4 bytes) - Light type enum (for G term)

//     /// 8

//     // Total: 60 bytes
// };

struct Reservoir
{
    // for LightSample
    float3 Li;          // Incident radiance from the light sample
    float3 lightDir;      // Normalized direction from shading point to light sample
    uint lightType;
    uint lightIndex;
    float2 lightUV;

    float targetPdf;
    float weightSum;
    float W;
    uint M;

    __init()
    {
        Li = float3(0.f);
        lightDir = float3(0.f);

        lightType = 0;
        lightIndex = 0;
        lightUV = float2(0.f);

        targetPdf = 0.f;
        weightSum = 0.f;
        W = 0.f;
        M = 0;
    }

    // Update reservoir with a new sample using weighted reservoir sampling (WRS)
    [mutating] bool update(LightSample ls, float recomputedPdf, float sourcePdf, float random)
    {
        // Compute sample weight: w_i = p-hat(x_i) / p(x_i)
        float sampleWeight = recomputedPdf / sourcePdf;

        // Accumulate weight sum
        this.weightSum += sampleWeight;
        this.M += 1;

        // Weighted reservoir sampling
        if (random < (sampleWeight / this.weightSum))
        {
            this.Li = ls.Li;
            this.lightDir = ls.dir;
            this.lightType = ls.lightType;
            this.lightIndex = ls.lightIndex;
            this.lightUV = ls.uv;

            this.targetPdf = recomputedPdf;
            return true;
        }
        return false;
    }

    [mutating] bool update(Reservoir other, LightSample ls, float recomputedPdf, float random)
    {
        // Compute sample weight: w_i = p-hat(x_i) / p(x_i), p(x_i) = weightSum / M
        float sampleWeight = recomputedPdf * other.W * float(other.M);

        // Accumulate weight sum
        this.weightSum += sampleWeight;
        this.M += other.M;

        // Weighted reservoir sampling
        if (random < (sampleWeight / this.weightSum))
        {
            this.Li = ls.Li;
            this.lightDir = ls.dir;
            this.lightType = other.lightType;
            this.lightIndex = other.lightIndex;
            this.lightUV = other.lightUV;

            this.targetPdf = recomputedPdf;
            return true;
        }
        return false;
    }

    /// Finalize reservoir: compute W = (1/M) * (p-hat(y) / p-avg(y))
    [mutating] void computeW()
    {
        if (M > 0 && weightSum > 0.f)
        {
            float weight = weightSum / float(M);      // weight = 1/M * {g/p}
            W = (weight * (1.f / targetPdf));         // W = 1/g * Weight -> f * W = f/g * Weight
        }
        else
        {
            W = 0.f;
        }
    }

    bool isValid() { return M > 0 && W > 0.f; }



    // /// Pack Reservoir to PackedReservoir
    // PackedReservoir pack()
    // {
    //     PackedReservoir p;

    //     // HDR colors → R11G11B10 (손실 압축, but sufficient for radiance)
    //     p.packedLi = packR11G11B10(Li);
    //     p.packedBsdf = packR11G11B10(bsdfVal);

    //     // Normalized directions → Octahedral encoding (high quality)
    //     p.packedDir = encodeNormal2x16(SurfaceToLightDir);
    //     p.packedLightNormal = encodeNormal2x16(lightNormalW);

    //     // 정밀도 필요한 값들은 그대로 유지
    //     p.targetPdf = targetPdf;
    //     p.weightSum = weightSum;
    //     p.W = W;
    //     p.lightPosW = lightPosW;
    //     p.lightOrigin = lightOrigin;

    //     // Integer values
    //     p.M = M;
    //     p.lightType = lightType;

    //     return p;
    // }

    // // Unpack PackedReservoir to Reservoir
    // static Reservoir unpackReservoir(PackedReservoir p)
    // {
    //     Reservoir r;

    //     // R11G11B10 → HDR colors
    //     r.Li = unpackR11G11B10(p.packedLi);
    //     r.bsdfVal = unpackR11G11B10(p.packedBsdf);

    //     // Octahedral → Normalized directions
    //     r.SurfaceToLightDir = decodeNormal2x16(p.packedDir);
    //     r.lightNormalW = decodeNormal2x16(p.packedLightNormal);

    //     // 정밀도 필요한 값들
    //     r.targetPdf = p.targetPdf;
    //     r.weightSum = p.weightSum;
    //     r.W = p.W;
    //     r.lightPosW = p.lightPosW;
    //     r.lightOrigin = p.lightOrigin;

    //     // Integer values
    //     r.M = p.M;
    //     r.lightType = p.lightType;

    //     return r;
    // }
};
