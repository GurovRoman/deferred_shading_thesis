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

layout (location = 0) out VS_OUT
{
    vec3 sNorm;
    vec3 sTangent;
    vec2 texCoord;
} vOut;


void main(void)
{
    const vec3 wNorm = DecodeNormal(floatBitsToUint(vPosNorm.w));
    const vec3 wTang = DecodeNormal(floatBitsToUint(vTexCoordAndTang.z));

    mat4 modelView = Params.view * params.model;

    mat3 normalModelView = transpose(inverse(mat3(modelView)));

    vOut.sNorm    = normalize(mat3(normalModelView) * wNorm.xyz);
    vOut.sTangent = normalize(mat3(normalModelView) * wTang.xyz);
    vOut.texCoord = vTexCoordAndTang.xy;

    gl_Position   = Params.proj * modelView * vec4(vPosNorm.xyz, 1.0f);
}