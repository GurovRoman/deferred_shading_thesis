#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"


layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

layout(binding = 1, set = 0) uniform sampler2D inColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 out_fragColor;

void main()
{
    vec2 offset = 1 / vec2(Params.screenWidth, Params.screenHeight);

    vec4 color = vec4(0);
    for (uint i = 0; i < SSAA_RATIO; ++i)
        for (uint j = 0; j < SSAA_RATIO; ++j)
          color += textureLod(inColor, inUV + offset * vec2(i, j), 0);

    out_fragColor = color / (SSAA_RATIO * SSAA_RATIO);
}
