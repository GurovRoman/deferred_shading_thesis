#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"
#include "normal_packing.glsl"

layout (location = 0) in VS_OUT
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

layout (location = 0) out vec2 outNormal;
#ifndef UV_BUFFER
layout (location = 1) out vec4 outAlbedo;
layout (location = 2) out vec2 outMetRough;
#else
layout (location = 1) out vec2 outUV;
layout (location = 2) out uint outMaterialID;
#endif

layout (binding = 1, set = 0, std430) buffer Materials { MaterialData_pbrMR materials[]; };
layout (binding = 2, set = 0, std430) buffer MaterialsID { uint matID[]; };
layout (binding = 3, set = 0) uniform sampler2D all_textures[MAX_TEXTURES];
layout (binding = 4, set = 0) buffer MeshInfos { uvec2 o[]; } infos;

#include "texture_data.glsl"

void main()
{
    const uint offset = infos.o[params.instanceID].x;
    const uint matIdx = matID[gl_PrimitiveID + offset / 3];


    // Alpha testing has to be done here anyway
    MaterialData_pbrMR material = materials[matIdx];
    float alpha = texture(all_textures[material.baseColorTexId], surf.texCoord).a;

    if (material.alphaMode == 1) {
        if (alpha < material.alphaCutoff) {
            discard;
        }
    }

#ifndef UV_BUFFER
    TextureData data = sampleTextures(material, surf.texCoord);

    vec3 N = surf.sNorm;
    /*if(material.normalTexId >= 0) {
        vec3 T = surf.sTangent;
        vec3 B = normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);
        N = TBN * data.normal;
    }*/

    outNormal = encode_normal(N);
    outAlbedo = vec4(data.albedo.xyz, 1.0);
    outMetRough = vec2(data.metallic, data.roughness);
#else
    outNormal = encode_normal(surf.sNorm);
    outUV = mod(surf.texCoord, vec2(1.0));
    outMaterialID = matIdx;
#endif
}
