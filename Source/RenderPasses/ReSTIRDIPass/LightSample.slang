#include "Utils/Math/MathConstants.slangh"

__exported import Scene.Scene;
__exported import Scene.Shading;
__exported import Scene.Material.ShadingUtils;
import Scene.Lights.LightData;
import Scene.RaytracingInline;
import Scene.SceneTypes;
import Scene.ShadingData;

import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import Rendering.Materials.IBSDF;

__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;

import LoadShadingData;
import PathVertex;
import StaticParams;

/** Light sample for ReSTIR direct illumination.

    This structure represents a sampled point on a light source and contains
    all information needed for shading and resampling operations.
*/
struct LightSample
{
    float3 Li;          ///< Incident radiance at the shading point (unshadowed). Already divided by PDF if normalized.
    float pdf;          ///< Probability density with respect to solid angle at the shading point.
    float3 normalW;     ///< Light surface normal (world space).
    float3 posW;        ///< Light surface position (world space).
    float3 origin;      ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance;     ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;         ///< Ray direction for visibility evaluation (normalized).

    uint lightType;     ///< Type of light source (LightType enum casted to uint).
    uint lightIndex;   ///< Index of the light source (for emissive and analytic lights).
    float2 uv;          ///< Barycentric coordinates or encoded direction (for reconstruction).

    // Total : 72bytes

    /** Initialize an invalid light sample.
    */
    __init()
    {
        Li = float3(0.f);
        pdf = 0.f;
        normalW = float3(0.f);
        posW = float3(0.f);
        origin = float3(0.f);
        distance = 0.f;
        dir = float3(0.f);
        lightType = uint(-1);
        lightIndex = uint(-1);
        uv = float2(0.f);
    }

    /** Check if the light sample is valid.
        \return True if sample has positive PDF and non-zero radiance.
    */
    bool isValid()
    {
        return any(Li > 0.f);
    }

    /** Construct a visibility ray for shadow testing.
        \return Ray from shading point to light sample position.
    */
    Ray getVisibilityRay()
    {
        return Ray(origin, dir, 0.f, distance);
    }

    /** Get the unshadowed contribution (Li / pdf).
        This is useful for RIS target function evaluation.
        \return Incident radiance divided by sampling PDF.
    */
    float3 getUnshadowedContribution()
    {
        return pdf > 0.f ? Li : float3(0.f);
    }

    /** Check if this is an environment map sample.
    */
    bool isEnvMapSample()
    {
        return lightType == (uint)LightType::EnvMap;
    }

    /** Check if this is an emissive geometry sample.
    */
    bool isEmissiveSample()
    {
        return lightType == (uint)LightType::Emissive;
    }

    /** Check if this is an analytic light sample.
    */
    bool isAnalyticSample()
    {
        return lightType == (uint)LightType::Analytic;
    }
};

/** Types of samplable lights (must match PathTracer definition).
*/
enum class LightType
{
    EnvMap,
    Emissive,
    Analytic
};

/** Helper functions for light sampling (to be used with LightSample).
*/

/** Generates a light sample on the environment map.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEnvMapSample<S : ISampleGenerator>(const PathVertex vertex, inout S sg, out LightSample ls, EnvMapSampler envMapSampler)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    if (!kUseEnvLight) return false;

    // Sample environment map.
    EnvMapSample lightSample;
    if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

    // Setup returned sample.
    ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    ls.pdf = lightSample.pdf;
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    ls.distance = kRayTMax;
    ls.dir = lightSample.dir;
    ls.uv = ndir_to_oct_equal_area_unorm(ls.dir); // oct_to_ndir_equal_area_unorm

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample<S : ISampleGenerator>(const PathVertex vertex, const bool upperHemisphere, inout S sg, out LightSample ls, EmissiveLightSampler emissiveSampler)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights) return false;

    TriangleLightSample tls;
    if (!emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    ls.normalW = tls.normalW;
    ls.posW = tls.posW;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);
    ls.uv = tls.uv;
    ls.lightIndex = tls.triangleIndex;

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the analytic lights.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateAnalyticLightSample<S : ISampleGenerator>(const PathVertex vertex, inout S sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    uint lightCount = gScene.getLightCount();
    if (!kUseAnalyticLights || lightCount == 0) return false;

    // Sample analytic light source selected uniformly from the light list.
    // TODO: Sample based on estimated contributions as pdf.
    uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

    // Sample local light source.
    AnalyticLightSample lightSample;
    if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

    // Setup returned sample.
    ls.pdf = lightSample.pdf / lightCount;
    ls.Li = lightSample.Li * lightCount;
    ls.normalW = lightSample.normalW;
    ls.posW = lightSample.posW;
    // Offset shading position to avoid self-intersection.
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    // Analytic lights do not currently have a ge ometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;
    ls.lightIndex = lightIndex;

    return any(ls.Li > 0.f);
}

/** Return the probabilities for selecting different light types.
    \param[out] p Probabilities.
*/
void getLightTypeSelectionProbabilities(out float p[3], bool useRestir = false)
{
    // Set relative probabilities of the different sampling techniques.
    // If environment map is available, guarantee 25% selection probability.
    if (useRestir)
    {
        p[0] = 0.25f; // 25% for environment map

        // Remaining 75% split between emissive and analytic lights
        float remainingProb = 0.75f;
        float emissiveWeight = kUseEmissiveLights ? 1.f : 0.f;
        float analyticWeight = kUseAnalyticLights ? 1.f : 0.f;
        float totalWeight = emissiveWeight + analyticWeight;

        if (totalWeight > 0.f)
        {
            p[1] = (emissiveWeight / totalWeight) * remainingProb;
            p[2] = (analyticWeight / totalWeight) * remainingProb;
        }
        else
        {
            // Only environment map available - use full probability
            p[0] = 1.f;
            p[1] = 0.f;
            p[2] = 0.f;
        }
    }
    else
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }
}

float getEnvMapSelectionProbability(bool useRestir = false) { float p[3]; getLightTypeSelectionProbabilities(p, useRestir); return p[0]; }
float getEmissiveSelectionProbability(bool useRestir = false) { float p[3]; getLightTypeSelectionProbabilities(p, useRestir); return p[1]; }
float getAnalyicSelectionProbability(bool useRestir = false) { float p[3]; getLightTypeSelectionProbabilities(p, useRestir); return p[2]; }

/** Select a light type for sampling.
    \param[out] lightType Selected light type.
    \param[out] pdf Probability for selected type.
    \param[in,out] sg Sample generator.
    \return Return true if selection is valid.
*/
bool selectLightType<S : ISampleGenerator>(out uint lightType, out float pdf, inout S sg, bool useRestir = false)
{
    float p[3];
    getLightTypeSelectionProbabilities(p, useRestir);

    float u = sampleNext1D(sg);

    [unroll]
    for (lightType = 0; lightType < 3; ++lightType)
    {
        if (u < p[lightType])
        {
            pdf = p[lightType];
            return true;
        }
        u -= p[lightType];
    }

    lightType = {};
    pdf = {};

    return false;
}

/** Configuration for light sampling.
    Encapsulates all sampler state in a single object following Falcor's context pattern.
*/
struct LightSamplingConfig
{
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;
    bool useRestir; ///< Use ReSTIR-specific probabilities (25% env map)

    /** Create configuration for standard path tracing.
    \param[in] envMap Environment map sampler (can be null if kUseEnvLight == false).
    \param[in] emissive Emissive light sampler (can be null if kUseEmissiveLights == false).
*/
    static LightSamplingConfig create(EnvMapSampler envMap, EmissiveLightSampler emissive, bool useRestir = false)
    {
        LightSamplingConfig config;
        config.envMapSampler = envMap;
        config.emissiveSampler = emissive;
        config.useRestir = useRestir;
        return config;
    }
};

/** Generates a light sample using sampling configuration.
    Single entry point - no overload explosion.

    \param[in] vertex Path vertex.
    \param[in] sampleUpperHemisphere Sample upper hemisphere.
    \param[in] sampleLowerHemisphere Sample lower hemisphere.
    \param[in,out] sg Sample generator.
    \param[in] config Light sampling configuration.
    \param[out] ls Resulting light sample.
    \return True if sample is valid.
*/
bool generateLightSample<S : ISampleGenerator>(
    const PathVertex vertex,
    const bool sampleUpperHemisphere,
    const bool sampleLowerHemisphere,
    inout S sg,
    const LightSamplingConfig config,
    out LightSample ls )
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg, config.useRestir)) return false;

    bool valid = false;
    if (kUseEnvLight && lightType == (uint)LightType::EnvMap)
        valid = generateEnvMapSample(vertex, sg, ls, config.envMapSampler);
    if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
    {
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls, config.emissiveSampler);
    }
    if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
    {
        valid = generateAnalyticLightSample(vertex, sg, ls);
    }
    if (!valid) return false;

    float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
        return false;

    ls.lightType = lightType;
    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;

    return true;
}

//-----------------------------------------------------------------------------
// Light Sample Reconstruction Functions
// These functions reconstruct LightSample from stored lightIndex and UV
// for ReSTIR spatial/temporal reuse.
//-----------------------------------------------------------------------------

/** Reconstructs an environment map light sample from stored direction.
    Environment map samples are position-independent, only direction matters.
    Supports two modes:
    1. Direct direction (lightDir != 0): Use the provided world-space direction
    2. UV decode (lightDir == 0): Decode direction from octahedral-encoded UV

    \param[in] shadingPosW Shading point in world space.
    \param[in] faceNormalW Face normal at shading point (for ray origin offset).
    \param[in] uv Stored UV coordinates that encode the sampled direction (octahedral mapping).
    \param[in] lightDir Cached light direction (world space). If zero, decode from UV.
    \param[in] envMapSampler Environment map sampler.
    \param[out] ls Reconstructed light sample.
    \return True if reconstruction succeeded, false otherwise.
*/
bool reconstructEnvMapSample(
    const float3 shadingPosW,
    const float3 faceNormalW,
    const float2 uv,
    const float3 lightDir,
    EnvMapSampler envMapSampler,
    out LightSample ls)
{
    ls = {};

    if (!kUseEnvLight) return false;

    // Determine the light direction
    float3 worldDir;
    // if (any(lightDir != 0.f))
    // {
    //     // Use cached direction directly
    //     worldDir = normalize(lightDir);
    // }
    // else
    // Decode direction from UV using octahedral mapping
    float3 localDir = oct_to_ndir_equal_area_unorm(uv);
    worldDir = gScene.envMap.toWorld(localDir);


    // Evaluate environment map at this direction
    float3 Le = envMapSampler.eval(worldDir);
    float pdf = envMapSampler.evalPdf(worldDir);

    if (pdf <= 0.f || all(Le <= 0.f)) return false;

    // Setup light sample
    ls.Li = Le / pdf;
    ls.pdf = pdf;
    ls.dir = worldDir;
    ls.origin = computeRayOrigin(shadingPosW, faceNormalW);
    ls.distance = kRayTMax;
    ls.normalW = float3(0.f);
    ls.posW = float3(0.f);
    ls.lightType = (uint)LightType::EnvMap;
    ls.lightIndex = 0; // EnvMap doesn't use lightIndex
    ls.uv = uv; // Store encoded direction

    return true;
}

/** Reconstructs an emissive triangle light sample from stored triangle index and barycentric UV.
    The light contribution is re-evaluated from the new shading position.

    \param[in] shadingPosW Shading point in world space.
    \param[in] faceNormalW Face normal at shading point (for ray origin offset).
    \param[in] triangleIndex Index of the emissive triangle in the light collection.
    \param[in] uv Barycentric coordinates (2D) on the triangle surface.
    \param[out] ls Reconstructed light sample.
    \return True if reconstruction succeeded, false otherwise.
*/
bool reconstructEmissiveSample(
    const float3 shadingPosW,
    const float3 faceNormalW,
    const uint triangleIndex,
    const float2 uv,
    out LightSample ls)
{
    ls = {};

    if (!kUseEmissiveLights) return false;

    // // Validate triangle index
    // uint triangleCount = gScene.lightCollection.getActiveTriangleCount();
    // if (triangleIndex >= triangleCount) return false;

    // Use the helper function from EmissiveLightSamplerHelpers
    TriangleLightSample tls;
    if (!sampleTriangle(shadingPosW, triangleIndex, uv, tls)) return false;

    // Setup light sample
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    ls.normalW = tls.normalW;
    ls.posW = tls.posW;

    // Offset shading and light position to avoid self-intersection
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = computeRayOrigin(shadingPosW, faceNormalW);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);

    // Guard against zero distance (shading point on light surface)
    if (ls.distance < FLT_MIN) return false;

    ls.dir = toLight / ls.distance;
    ls.lightType = (uint)LightType::Emissive;
    ls.lightIndex = triangleIndex;
    ls.uv = uv;

    return any(ls.Li > 0.f);
}

/** Reconstructs an analytic light sample from stored light index.
    Analytic lights (point, spot, directional) are fully defined by their index.
    UV parameter is used for area lights (rect, sphere, disc).

    Note on delta lights (point, directional):
    - These have pdf=0 (Dirac delta distribution)
    - For ReSTIR, we treat them as pdf=1.0 since they are deterministic
    - Li already contains the full contribution (no division needed)

    \param[in] shadingPosW Shading point in world space.
    \param[in] faceNormalW Face normal at shading point (for ray origin offset).
    \param[in] lightIndex Index of the analytic light in the scene.
    \param[in] uv Random UV used for area light sampling (ignored for point/directional).
    \param[out] ls Reconstructed light sample.
    \return True if reconstruction succeeded, false otherwise.
*/
bool reconstructAnalyticSample(
    const float3 shadingPosW,
    const float3 faceNormalW,
    const uint lightIndex,
    const float2 uv,
    out LightSample ls)
{
    ls = {};

    uint lightCount = gScene.getLightCount();
    if (!kUseAnalyticLights || lightCount == 0 || lightIndex >= lightCount) return false;

    LightData light = gScene.getLight(lightIndex);
    AnalyticLightSample als;

    // For delta lights (point, directional), we can use evalLightApproximate
    // For area lights (rect, sphere, disc), we need to use the UV to reconstruct the sample point
    bool valid = false;
    bool isDeltaLight = false;

    switch (light.type)
    {
    case LightType::Point:
    case LightType::Directional:
        // Delta lights - UV is not needed, pdf is Dirac delta (set to 0 by Falcor)
        valid = evalLightApproximate(shadingPosW, light, als);
        isDeltaLight = true;
        break;
    case LightType::Rect:
        valid = sampleRectAreaLight(shadingPosW, light, uv, als);
        break;
    case LightType::Sphere:
        valid = sampleSphereAreaLight(shadingPosW, light, uv, als);
        break;
    case LightType::Disc:
        valid = sampleDiscAreaLight(shadingPosW, light, uv, als);
        break;
    case LightType::Distant:
        return sampleDistantLight(shadingPosW, light, uv, als);

    default:
        return false;
    }

    if (!valid) return false;

    // Setup light sample
    // Note: For ReSTIR, we don't divide by lightCount here since we're reconstructing a specific light
    ls.normalW = als.normalW;
    ls.posW = als.posW;
    ls.origin = computeRayOrigin(shadingPosW, faceNormalW);
    ls.distance = als.distance;
    ls.dir = als.dir;
    ls.lightType = (uint)LightType::Analytic;
    ls.lightIndex = lightIndex;
    ls.uv = uv;

    // Handle delta lights specially:
    // - Falcor sets pdf=0 for Dirac delta distributions
    // - For ReSTIR, we use pdf=1.0 since the sample is deterministic
    // - Li already contains the correct contribution (intensity / distance^2)
    if (isDeltaLight)
    {
        ls.Li = als.Li * gScene.getLightCount();  // Already correct, no pdf division needed
        ls.pdf = 1.0f;   // Treat as deterministic for ReSTIR weighting
    }
    else
    {
        // Area lights have proper pdf
        if (als.pdf <= 0.f) return false;
        ls.Li = als.Li * gScene.getLightCount(); // Scale by total light count for multiple importance sampling
        ls.pdf = als.pdf / gScene.getLightCount();
    }

    return any(ls.Li > 0.f);
}

/** Unified light sample reconstruction function.
    Dispatches to the appropriate reconstruction function based on light type.

    \param[in] shadingPosW Shading point in world space.
    \param[in] faceNormalW Face normal at shading point.
    \param[in] lightType Type of light (LightType enum).
    \param[in] lightIndex Light index (triangle index for emissive, light index for analytic).
    \param[in] uv UV coordinates (barycentric for emissive, direction encoding for envmap, sample UV for area lights).
    \param[in] config Light sampling configuration containing samplers.
    \param[out] ls Reconstructed light sample.
    \param[in] lightDir Optional cached light direction for EnvMap (if zero, decode from UV).
    \return True if reconstruction succeeded, false otherwise.
*/
bool reconstructLightSample(
    const float3 shadingPosW,
    const float3 faceNormalW,
    const uint lightType,
    const uint lightIndex,
    const float2 uv,
    const LightSamplingConfig config,
    out LightSample ls,
    const float3 lightDir = float3(0.f))
{
    ls = {};

    float selectionPdf = 1.0f;
    bool valid = false;

    switch (lightType)
    {
    case (uint)LightType::EnvMap:
        valid = reconstructEnvMapSample(shadingPosW, faceNormalW, uv, lightDir, config.envMapSampler, ls);
        selectionPdf = getEnvMapSelectionProbability(config.useRestir);
        break;

    case (uint)LightType::Emissive:
        valid = reconstructEmissiveSample(shadingPosW, faceNormalW, lightIndex, uv, ls);
        selectionPdf = getEmissiveSelectionProbability(config.useRestir);
        break;

    case (uint)LightType::Analytic:
        valid = reconstructAnalyticSample(shadingPosW, faceNormalW, lightIndex, uv, ls);
        selectionPdf = getAnalyicSelectionProbability(config.useRestir);
        break;
    }

    ls.Li /= selectionPdf;
    ls.pdf *= selectionPdf;

    return valid;
}

/** Overload for Reservoir-based reconstruction.
    Convenience function that extracts light info from a Reservoir structure.

    \param[in] shadingPosW Shading point in world space.
    \param[in] faceNormalW Face normal at shading point.
    \param[in] lightType Type of light from reservoir.
    \param[in] lightIndex Light index from reservoir.
    \param[in] uv UV coordinates from reservoir.
    \param[in] lightDir Cached light direction from reservoir.
    \param[in] config Light sampling configuration.
    \param[out] ls Reconstructed light sample.
    \return True if reconstruction succeeded, false otherwise.
*/
bool reconstructLightSampleFromReservoir(
    const float3 shadingPosW,
    const float3 faceNormalW,
    const uint lightType,
    const uint lightIndex,
    const float2 uv,
    const float3 lightDir,
    const LightSamplingConfig config,
    out LightSample ls)
{
    return reconstructLightSample(shadingPosW, faceNormalW, lightType, lightIndex, uv, config, ls, lightDir);
}
