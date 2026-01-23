import Params;
import Utils.Math.PackedFormats;
__exported import PathState;
__exported import Params;

/** Describes a light sample.
 */
struct LightSample
{
    float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float pdf;      ///< Pdf with respect to solid angle at the shading point.
    float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;     ///< Ray direction for visibility evaluation (normalized).
    uint lightType; ///< Light type this sample comes from (LightType casted to uint).

    Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
};

struct Reservoir
{
    float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float pdf;      ///< Pdf with respect to solid angle at the shading point.
    float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;     ///< Ray direction for visibility evaluation (normalized).
    uint lightType; ///< Light type this sample comes from (LightType casted to uint).

    float targetPdf;
    float totalWeight;
    uint M;


    __init()
    {
        this.Li = float3(0.f);
        this.pdf = 0.f;
        this.origin = float3(0.f);
        this.distance = 0.f;
        this.dir = float3(0.f);
        this.lightType = 0;
        this.targetPdf = 0.f;
        this.totalWeight = 0.f;
        this.M = 0;
    }

    [mutating] void update(const LightSample sample, const float targetPdf, const float sourcePdf, inout SampleGenerator sg)
    {
        float weight = targetPdf / sourcePdf;
        this.totalWeight += weight;

        this.M += 1;

        // Reservoir sampling.
        if (sampleNext1D(sg) < (weight / this.totalWeight))
        {

            this.Li = sample.Li;
            this.pdf = sample.pdf;
            this.origin = sample.origin;
            this.distance = sample.distance;
            this.dir = sample.dir;
            this.lightType = sample.lightType;

            this.targetPdf = targetPdf;
        }
    }

    float3 computeW()
    {
        return this.Li * (this.totalWeight / (this.M / this.targetPdf));
    }
};
