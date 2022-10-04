#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"

layout (location = 0 ) in VS_OUT
{
    vec3 sNorm;
    vec3 sTangent;
    vec2 texCoord;
} surf;

layout(push_constant) uniform params_t
{
    PushConst params;
};

layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

layout (location = 0) out vec4 outNormal;
layout (location = 1) out vec4 outTangent;
layout (location = 2) out vec4 outAlbedo;
layout (location = 3) out vec4 outMetRough;


layout (binding = 1, set = 0, std430) buffer Materials { MaterialData_pbrMR materials[]; };
layout (binding = 2, set = 0, std430) buffer MaterialsID { uint matID[]; };
layout (binding = 3, set = 0) uniform sampler2D all_textures[MAX_TEXTURES];
layout (binding = 4, set = 0) buffer MeshInfos { uvec2 o[]; } infos;


void main()
{
    const uint offset = infos.o[params.instanceID].x;
    const uint matIdx = matID[gl_PrimitiveID + offset / 3];
    vec4 base_color = materials[matIdx].baseColor;
    if(materials[matIdx].baseColorTexId >= 0) {
        const vec4 tex_val = texture(all_textures[materials[matIdx].baseColorTexId], surf.texCoord);
        base_color.xyz *= tex_val.xyz;
        base_color.a    = tex_val.a;
    }

    if (materials[matIdx].alphaMode == 1) {
        if (base_color.a < materials[matIdx].alphaCutoff) {
            discard;
        }
    }
    vec3 N = surf.sNorm;
    if(materials[matIdx].normalTexId >= 0) {
        vec3 T = surf.sTangent;
        vec3 B = normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);
        N = TBN * normalize(texture(all_textures[materials[matIdx].normalTexId], surf.texCoord).xyz * 2.0 - vec3(1.0));
    }

    float metallic  = materials[matIdx].metallic;
    float roughness = materials[matIdx].roughness;
    if (materials[matIdx].metallicRoughnessTexId >= 0) {
        metallic *= texture(all_textures[materials[matIdx].metallicRoughnessTexId], surf.texCoord).b;
        roughness *= texture(all_textures[materials[matIdx].metallicRoughnessTexId], surf.texCoord).g;
    }

    outNormal = vec4(surf.sNorm, 0.0);
    outTangent = vec4(surf.sTangent, 0.0);
    outAlbedo = vec4(base_color.xyz, 1.0);
    outMetRough = vec4(metallic, roughness, 0.f, 0.f);
}
