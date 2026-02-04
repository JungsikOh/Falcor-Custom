/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Materials/TexLODTypes.slang"
#include "Rendering/Utils/PixelStats.h"

using namespace Falcor;

/**
 * Minimal path tracer.
 *
 * This pass implements a minimal brute-force path tracer. It does purposely
 * not use any importance sampling or other variance reduction techniques.
 * The output is unbiased/consistent ground truth images, against which other
 * renderers can be validated.
 *
 * Note that transmission and nested dielectrics are not yet supported.
 */
class ReSTIRDIPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRDIPass, "ReSTIRDIPass", "path tracer using ReSTIR DI");

    static ref<ReSTIRDIPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReSTIRDIPass>(pDevice, props); }

    ReSTIRDIPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    enum class Mode
    {
        NoResampling,
        SpatialResampling,
        TemporalResampling,
        SpatiotemporalResampling,
    };

    enum class BiasedMode
    {
        Biased,
        UnbiasedNaive,
        UnbiasedMIS
    };

private:
    struct TracePass
    {
        std::string name;
        std::string passDefine;
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        TracePass(
            ref<Device> pDevice,
            const std::string& name,
            const std::string& passDefine,
            const ref<Scene>& pScene,
            const DefineList& defines,
            const TypeConformanceList& globalTypeConformances
        );
        static std::unique_ptr<TracePass> create(
            ref<Device> pDevice,
            const std::string& name,
            const std::string& passDefine,
            const ref<IScene>& pScene,
            const DefineList& defines,
            const TypeConformanceList& globalTypeConformances
        )
        {
            if (auto scene = dynamic_ref_cast<Scene>(pScene))
                return std::make_unique<TracePass>(std::move(pDevice), name, passDefine, std::move(scene), defines, globalTypeConformances);
            return {};
        }

        void prepareProgram(ref<Device> pDevice, const DefineList& defines);
    };

    void parseProperties(const Properties& props);
    void setFrameDim(const uint2 frameDim);
    void resetPrograms();
    void updatePrograms();

    void resetLighting();
    void prepareMaterials(RenderContext* pRenderContext);
    bool prepareLighting(RenderContext* pRenderContext);

    void prepareVars();
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);

    // Render Passes
    void loadSurfaceDataPass(RenderContext* pRenderContext, const RenderData& renderData);
    void spatialReusePass(RenderContext* pRenderContext, const RenderData& renderData);
    void temporalReusePass(RenderContext* pRenderContext, const RenderData& renderData);
    void computeDirectLightingPass(RenderContext* pRenderContext, const RenderData& renderData);

    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);

    struct StaticParams
    {
        uint samplesPerPixel = 1;
        bool adjustShadingNormals = false;

        uint candidateCount = 16;       ///< Number of light candidates per pixel (M_initial)
        uint spatialReuseIteration = 3;///< Number of spatial reuse iterations
        uint spatialReuseNeighbors = 1;///< Number of spatial neighbors to consider (K)
        uint restirMode = 3;         ///< 0: no resampling, 1: spatial resampling, 2: temporal resampling, 3: spatiotemporal resampling
        uint biasedMode = (uint)BiasedMode::Biased;          ///< Use biased mode for spatiotemporal resampling

        bool useDebugOutput = true;
        bool useRussianRoulette = false;
        bool useMIS = true;
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Uniform;
        DefineList getDefines(const ReSTIRDIPass& owner) const;
    };

    // Configuration
    StaticParams mStaticParams;                                    ///< Static parameters.
    mutable LightBVHSampler::Options mLightBVHOptions; ///< Current options for the light BVH sampler.
    bool mEnabled = true;       ///< Switch to enable/disable the path tracer. When disabled the pass outputs are cleared.

    // Internal state
    ref<Scene> mpScene;                                      ///< The Current scene.
    ref<SampleGenerator> mpSampleGenerator;                  ///< GPU sample generator.
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;          ///< Environment map sampler or nullptr if not used.
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler; ///< Emissive light sampler or nullptr if not used.


    /// Max number of indirect bounces (0 = none).
    uint mMaxBounces = 3;
    /// Compute direct illumination (otherwise indirect only).
    bool mComputeDirect = true;
    /// Use importance sampling for materials.
    bool mUseImportanceSampling = true;

    // Runtime data

    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    uint2 mFrameDim = {0, 0};

    bool mOptionsChanged = false;   ///< Flag indicating whether the options have changed.
    bool mRecompile = false;        ///< Set to true when program specialization has changed.
    bool mVarsChanged = true;       ///< This is set to true whenever the program vars have changed and resources need to be rebound.

    ref<ComputePass> mpLoadSurfaceDataPass;          ///< Fullscreen compute pass loading surface data from VBuffer.
    ref<ComputePass> mpSpatialResamplingPass;        ///< Fullscreen compute pass performing spatial resampling.
    ref<ComputePass> mpTemporalResamplingPass;       ///< Fullscreen compute pass performing temporal resampling.
    ref<ComputePass> mpComputeDirectLightingPass;    ///< Fullscreen compute pass performing shading using the ReSTIR reservoirs.
    ref<ComputePass> mpReflectTypesPass;  ///< Helper for reflecting structured buffer types.

    std::unique_ptr<TracePass> mpTracePass; ///< Ray tracing program.

    ref<Buffer> mpDirectLightingSampleColor;                ///< Buffer for Direct Lighting sample color.
    ref<Buffer> mpReservoirs;                               ///< Buffer for ReSTIR reservoirs.
    ref<Buffer> mpSurfaceData;                              ///< Buffer for surface data(pos, normal, weight, ...) storage.

    ref<Buffer> mpPrevReservoirs;  ///< Buffer for Previous ReSTIR reservoirs.
    ref<Buffer> mpPrevSurfaceData; ///< Buffer for Previous surface data(pos, normal, weight, ...) storage.
};
