#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"
#include "unpack_attributes.h"


layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
    PushConst params;
};

layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

layout (location = 0 ) out VS_OUT
{
    vec2 texCoord;
} vOut;

void main(void)
{
    vOut.texCoord = vTexCoordAndTang.xy;

    gl_Position   = Params.lightMatrix * params.model * vec4(vPosNorm.xyz, 1.0);
}
