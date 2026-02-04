/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ReSTIR.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"


extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRDIPass>();
}

namespace
{
const std::string kLoadSurfaceDataPassFile = "RenderPasses/ReSTIRDIPass/LoadSurfaceData.cs.slang";
const std::string kSpatialReuseFile = "RenderPasses/ReSTIRDIPass/SpatialReuse.cs.slang";
const std::string kTemporalReuseFile = "RenderPasses/ReSTIRDIPass/TemporalReuse.cs.slang";
const std::string kComputeDirectLightingFile = "RenderPasses/ReSTIRDIPass/DirectLighting.cs.slang";
const std::string kReflectTypesFile = "RenderPasses/ReSTIRDIPass/ReflectTypes.cs.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

// Render Pass inputs and outputs.
// Inputs from VBufferRT
const std::string kInputVBuffer = "vbuffer";
const std::string kInputMotionVectors = "mvec";
const std::string kInputViewDir = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    {kInputVBuffer,         "gVBuffer",     "Visibility buffer in packed format" },
    {kInputMotionVectors,   "gMotionVectors", "Motion vector buffer (float format)", true /* optional */},
    {kInputViewDir,         "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const std::string kOutputColor = "color";
const std::string kDebug = "debugTest3";
const ChannelList kOutputChannels = {
    // clang-format off
    { kOutputColor,          "", "Output color (sum of direct and indirect)", true, ResourceFormat::RGBA32Float },
    { kDebug,                "",       "Debug output", true, ResourceFormat::RGBA32Float },
    // clang-format on
};

const Gui::DropdownList kRestirModeList = {
    {0, "No Resampling"},
    {1, "Spatial Resampling"},
    {2, "Temporal Resampling"},
    {3, "Spatiotemporal Resampling"},
};

const Gui::DropdownList kBiasedModeList = {
    {0, "Biased"},
    {1, "Unbiased Naive"},
    {2, "Unbiased MIS"},
};

const char kMaxBounces[] = "maxBounces";

const char kCandidateCount[] = "candidateCount";
const char kSelectRestirMode[] = "restirMode";
const char kSpatialReuseIteration[] = "spatialReuseIteration";
const char kSpatialReuseNeighbors[] = "spatialReuseNeighbors";
const char kBiasedMode[] = "biasedMode";
const char kUseImportanceSampling[] = "useImportanceSampling";

} // namespace

ReSTIRDIPass::ReSTIRDIPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("PathTracer requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("PathTracer requires Raytracing Tier 1.1 support.");

    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void ReSTIRDIPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kCandidateCount)
            mStaticParams.candidateCount = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else if (key == kSelectRestirMode)
            mStaticParams.restirMode = value;
        else if (key == kBiasedMode)
            mStaticParams.biasedMode = value;
        else if (key == kSpatialReuseIteration)
            mStaticParams.spatialReuseIteration = value;
        else if (key == kSpatialReuseNeighbors)
            mStaticParams.spatialReuseNeighbors = value;
        else
            logWarning("Unknown property '{}' in MinimalPathTracer properties.", key);
    }
}

void ReSTIRDIPass::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = frameDim;

    mFrameDim = frameDim;
    if (any(mFrameDim != prevFrameDim))
    {
        mVarsChanged = true;
    }
}

Properties ReSTIRDIPass::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kCandidateCount] = mStaticParams.candidateCount;
    props[kSpatialReuseIteration] = mStaticParams.spatialReuseIteration;
    props[kSpatialReuseNeighbors] = mStaticParams.spatialReuseNeighbors;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    props[kBiasedMode] = (uint32_t)mStaticParams.biasedMode;
    return props;
}

RenderPassReflection ReSTIRDIPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void ReSTIRDIPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData))
        return;

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    loadSurfaceDataPass(pRenderContext, renderData);

    switch (mStaticParams.restirMode)
    {
    case 0: // No resampling
        break;
    case 1: // Spatial resampling
        spatialReusePass(pRenderContext, renderData);
        break;
    case 2: // Temporal resampling
        temporalReusePass(pRenderContext, renderData);
        break;
    case 3: // Spatiotemporal resampling
        temporalReusePass(pRenderContext, renderData);
        spatialReusePass(pRenderContext, renderData);
        break;
    default:
        break;
    }

    computeDirectLightingPass(pRenderContext, renderData);

    endFrame(pRenderContext, renderData);
}

void ReSTIRDIPass::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.var("Candidate Count", mStaticParams.candidateCount, 1u, 64u);
    widget.tooltip("Select the Candidate Count for ReSTIR", true);

    dirty |= widget.dropdown("ReSTIR Mode", kRestirModeList, (uint32_t&)mStaticParams.restirMode);
    dirty |= widget.dropdown("Biased Mode", kBiasedModeList, (uint32_t&)mStaticParams.biasedMode);


    dirty |= widget.var("Spatial Reuse Iteration", mStaticParams.spatialReuseIteration, 1u, 16u);
    widget.tooltip("Number of spatial reuse iterations", true);

    dirty |= widget.var("Spatial Reuse Neighbors", mStaticParams.spatialReuseNeighbors, 1u, 64u);
    widget.tooltip("Number of spatial neighbors to consider for spatial reuse", true);


    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mRecompile = true;
        mOptionsChanged = true;
    }
}

void ReSTIRDIPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    if (mpTracePass)
    {
        mpTracePass->pProgram = nullptr;
        mpTracePass->pBindingTable = nullptr;
        mpTracePass->pVars = nullptr;
    }

    mFrameCount = 0;
    mFrameDim = {0, 0};

    resetPrograms();
    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ReSTIRDIPass: This render pass does not support custom primitives.");
        }
        mRecompile = true;
    }
}

void ReSTIRDIPass::resetPrograms()
{
    if (mpTracePass)
    {
        mpTracePass->pProgram = nullptr;
        mpTracePass->pBindingTable = nullptr;
        mpTracePass->pVars = nullptr;
    }


    mpLoadSurfaceDataPass = nullptr;
    mpSpatialResamplingPass = nullptr;
    mpTemporalResamplingPass = nullptr;
    mpComputeDirectLightingPass = nullptr;
    mpReflectTypesPass = nullptr;
    mRecompile = true;
}

void ReSTIRDIPass::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false) return;

    // If we get here, a change that require recompilation of shader programs has occurred.
    // This may be due to change of scene defines, type conformances, shader modules, or other changes that require recompilation.
    // When type conformances and/or shader modules change, the programs need to be recreated. We assume programs have been reset upon such
    // changes. When only defines have changed, it is sufficient to update the existing programs and recreate the program vars.

    auto defines = mStaticParams.getDefines(*this);
    TypeConformanceList globalTypeConformances;
    mpScene->getTypeConformances(globalTypeConformances);


    // Create compute passes.
    ProgramDesc baseDesc;
    mpScene->getShaderModules(baseDesc.shaderModules);
    baseDesc.addTypeConformances(globalTypeConformances);

    if(!mpLoadSurfaceDataPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kLoadSurfaceDataPassFile).csEntry("main");
        mpLoadSurfaceDataPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if(!mpSpatialResamplingPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSpatialReuseFile).csEntry("main");
        mpSpatialResamplingPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if(!mpTemporalResamplingPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kTemporalReuseFile).csEntry("main");
        mpTemporalResamplingPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if(!mpComputeDirectLightingPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kComputeDirectLightingFile).csEntry("main");
        mpComputeDirectLightingPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    if (!mpReflectTypesPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypesPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    auto preparePass = [&](ref<ComputePass> pass)
    {
        // Note that we must use set instead of add defines to replace any stale state.
        pass->getProgram()->setDefines(defines);

        // Recreate program vars. This may trigger recompilation if needed.
        // Note that program versions are cached, so switching to a previously used specialization is faster.
        pass->setVars(nullptr);
    };
    preparePass(mpLoadSurfaceDataPass);
    preparePass(mpSpatialResamplingPass);
    preparePass(mpTemporalResamplingPass);
    preparePass(mpComputeDirectLightingPass);
    preparePass(mpReflectTypesPass);

    mVarsChanged = true;
    mRecompile = false;
}

void ReSTIRDIPass::resetLighting()
{
    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void ReSTIRDIPass::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for material changes and performs any necessary update.
    // For now all we need to do is to trigger a recompile so that the right defines get set.
    // In the future, we might want to do additional material-specific setup here.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::MaterialsChanged))
    {
        mRecompile = true;
    }
}

bool ReSTIRDIPass::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }
    if(is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }
    if(is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }
    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if(mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getILightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getILightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler =
                    std::make_unique<LightBVHSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext), mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            default:
                FALCOR_THROW("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->pProgram->addDefines(defines))
            mRecompile = true;
    }

    return lightingChanged;
}

void ReSTIRDIPass::prepareVars()
{
    FALCOR_ASSERT(mpScene);

    // Configure program.
    mpTracePass->pProgram->addDefines(mpSampleGenerator->getDefines());
    mpTracePass->pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mpTracePass->pVars = RtProgramVars::create(mpDevice, mpTracePass->pProgram, mpTracePass->pBindingTable);

    // Bind utility classes into shared data.
    auto var = mpTracePass->pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}

void ReSTIRDIPass::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint32_t screenPixelCount = mFrameDim.x * mFrameDim.y;

    auto var = mpReflectTypesPass->getRootVar();
    if (!mpSurfaceData || mpSurfaceData->getElementCount() < screenPixelCount)
    {
        mpSurfaceData = mpDevice->createStructuredBuffer(
            var["surfaceData"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mVarsChanged = true;
    }

    if (!mpReservoirs || mpReservoirs->getElementCount() < screenPixelCount)
    {
        mpReservoirs = mpDevice->createStructuredBuffer(
            var["reservoirs"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mVarsChanged = true;
    }

    if(!mpPrevSurfaceData || mpPrevSurfaceData->getElementCount() < screenPixelCount)
    {
        mpPrevSurfaceData = mpDevice->createStructuredBuffer(
            var["prevSurfaceData"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mVarsChanged = true;
    }
    if(!mpPrevReservoirs || mpPrevReservoirs->getElementCount() < screenPixelCount)
    {
        mpPrevReservoirs = mpDevice->createStructuredBuffer(
            var["prevReservoirs"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mVarsChanged = true;
    }
}

void ReSTIRDIPass::loadSurfaceDataPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "loadSurfaceDataPass");
    if (!mpLoadSurfaceDataPass)
    {
        logError("Failed to create LoadSurfaceDataPass");
        return;
    }

    auto var = mpLoadSurfaceDataPass->getRootVar()["CB"]["gLoadSurfaceDataPass"];
    mpLoadSurfaceDataPass->addDefine("CANDIDATE_COUNT", std::to_string(mStaticParams.candidateCount));

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    if (mVarsChanged) mpSampleGenerator->bindShaderData(var);
    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["envMapSampler"]);
    if (mpEmissiveSampler) mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);

    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);
    var["gMotionVectors"] = renderData.getTexture(kInputMotionVectors);

    var["gSurfaceData"] = mpSurfaceData;
    var["gReservoirs"] = mpReservoirs;

    var["gOutputColor"] = renderData.getTexture(kOutputColor);

    mpScene->bindShaderData(mpLoadSurfaceDataPass->getRootVar()["gScene"]);

    mpLoadSurfaceDataPass->execute(pRenderContext, {mFrameDim.x, mFrameDim.y, 1u});
}

void ReSTIRDIPass::spatialReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "spatialReusePass");
    if (!mpSpatialResamplingPass)
    {
        logError("Failed to create SpatialResamplingPass");
        return;
    }

    for (size_t i = 0; i < mStaticParams.spatialReuseIteration; i++)
    {
        auto var = mpSpatialResamplingPass->getRootVar()["CB"]["gSpatialReusePass"];

        var["gFrameDim"] = mFrameDim;
        var["gFrameCount"] = mFrameCount;

        std::swap(mpReservoirs, mpPrevReservoirs);

        if (mVarsChanged) mpSampleGenerator->bindShaderData(var);
        if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["envMapSampler"]);

        var["gSurfaceData"] = mpSurfaceData;
        var["gReservoirs"] = mpPrevReservoirs;
        var["gOutReservoirs"] = mpReservoirs;

        mpScene->bindShaderData(mpSpatialResamplingPass->getRootVar()["gScene"]);

        mpSpatialResamplingPass->execute(pRenderContext, {mFrameDim.x, mFrameDim.y, 1u});
    }
}

void ReSTIRDIPass::temporalReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "temporalReusePass");
    if (!mpTemporalResamplingPass)
    {
        logError("Failed to create TemporalResamplingPass");
        return;
    }

    auto var = mpTemporalResamplingPass->getRootVar()["CB"]["gTemporalReusePass"];

    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    if (mVarsChanged) mpSampleGenerator->bindShaderData(var);
    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["envMapSampler"]);

    var["gSurfaceData"] = mpSurfaceData;
    var["gPrevSurfaceData"] = mpPrevSurfaceData;

    var["gReservoirs"] = mpReservoirs;
    var["gPrevReservoirs"] = mpPrevReservoirs;

    mpScene->bindShaderData(mpTemporalResamplingPass->getRootVar()["gScene"]);

    mpTemporalResamplingPass->execute(pRenderContext, {mFrameDim.x, mFrameDim.y, 1u});
}

void ReSTIRDIPass::computeDirectLightingPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "computeDirectLightingPass");
    if (!mpComputeDirectLightingPass)
    {
        logError("Failed to create ComputeDirectLightingPass");
        return;
    }

    auto var = mpComputeDirectLightingPass->getRootVar()["CB"]["gDirectLightingPass"];
    var["gFrameDim"] = mFrameDim;
    var["gFrameCount"] = mFrameCount;

    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["envMapSampler"]);

    var["gVBuffer"] = renderData.getTexture(kInputVBuffer);

    var["gReservoirs"] = mpReservoirs;
    var["gOutputColor"] = renderData.getTexture(kOutputColor);

    mpScene->bindShaderData(mpComputeDirectLightingPass->getRootVar()["gScene"]);

    mpComputeDirectLightingPass->execute(pRenderContext, {mFrameDim.x, mFrameDim.y, 1u});
}

bool ReSTIRDIPass::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels)
    {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mFrameDim.x || pTexture->getHeight() != mFrameDim.y))
                resolutionMismatch = true;
        }
    };
    validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("PathTracer I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged)
                flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }
        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged)
            flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged)
            flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    return true;
}

void ReSTIRDIPass::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    std::swap(mpReservoirs, mpPrevReservoirs);
    std::swap(mpSurfaceData, mpPrevSurfaceData);

    // Copy pixel stats to outputs if available.

    mVarsChanged = false;
    mFrameCount++;
}

DefineList ReSTIRDIPass::StaticParams::getDefines(const ReSTIRDIPass& owner) const
{
    DefineList defines;

    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");

    defines.add("CANDIDATE_COUNT", std::to_string(candidateCount));
    defines.add("SPATIAL_REUSE_ITERATION", std::to_string(spatialReuseIteration));
    defines.add("SPATIAL_REUSE_NEIGHBORS", std::to_string(spatialReuseNeighbors));

    defines.add("BIASED_MODE", std::to_string(biasedMode));

    defines.add("USE_DEBUG_OUTPUT", useDebugOutput ? "1" : "0");

    defines.add("USE_MIS", owner.mStaticParams.useMIS ? "1" : "0");

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());

    // Scene-specific configuration.
    // Set defaults
    defines.add("USE_ENV_LIGHT", "0");
    defines.add("USE_ANALYTIC_LIGHTS", "0");
    defines.add("USE_EMISSIVE_LIGHTS", "0");

    const auto& scene = owner.mpScene;
    if(scene) {
    defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene->useEmissiveLights() ? "1" : "0");
    }

    return defines;
}
