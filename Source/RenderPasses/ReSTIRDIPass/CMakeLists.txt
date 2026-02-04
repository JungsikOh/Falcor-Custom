add_plugin(ReSTIRDIPass)

target_sources(ReSTIRDIPass PRIVATE
    StaticParams.slang
    Params.slang
    PathVertex.slang
    LightSample.slang
    Reservoir.slang
    SurfaceData.slang
    LoadShadingData.slang
    ReSTIR.cpp
    ReSTIR.h
    LoadSurfaceData.cs.slang
    SpatialReuse.cs.slang
    TemporalReuse.cs.slang
    DirectLighting.cs.slang
    ReflectTypes.cs.slang
)

target_copy_shaders(ReSTIRDIPass RenderPasses/ReSTIRDIPass)

target_source_group(ReSTIRDIPass "RenderPasses")
