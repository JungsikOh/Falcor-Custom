add_plugin(ReSTIRGIPass)

target_sources(ReSTIRGIPass PRIVATE
    SurfaceData.slang
    ReservoirGI.slang
    SpatioTemporalReuse.cs.slang
    ShadingColor.cs.slang

    ColorType.slang
    GeneratePaths.cs.slang
    GuideData.slang
    LoadShadingData.slang
    NRDHelpers.slang
    Params.slang
    PathState.slang
    PathTracer.slang
    ReSTIRGIPass.cpp
    ReSTIRGIPass.h
    PathTracerNRD.slang
    ReflectTypes.cs.slang
    ResolvePass.cs.slang
    StaticParams.slang
    TracePass.rt.slang
)

target_copy_shaders(ReSTIRGIPass RenderPasses/ReSTIRGIPass)

target_source_group(ReSTIRGIPass "RenderPasses")
