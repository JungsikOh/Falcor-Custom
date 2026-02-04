/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
__exported import Rendering.Materials.TexLODTypes;

/** Translation of defines set by the host to compile-time constants used to
    configure the path tracer without overhead from dynamic control flow.
    This will eventually be replaced by specialization constants in Slang.
*/

#ifndef SAMPLES_PER_PIXEL
#error StaticParams.slang is missing defines
#endif

static const float kRayTMax = 1e30f;
static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;

// ReSTIR parameters
static const uint kCandidateCount = CANDIDATE_COUNT;
static const uint kBisedMode = BIASED_MODE; // 0: Biased, 1: Unbiased Naive, 2: Unbiased MIS
static const uint kSpatialReuseIteration = SPATIAL_REUSE_ITERATION;
static const uint kSpatialReuseNeighbors = SPATIAL_REUSE_NEIGHBORS;

static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;

static const bool kUseDebugOutput = USE_DEBUG_OUTPUT;
