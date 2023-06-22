#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
//#extension GL_EXT_debug_printf : enable

#include "common.h"
#include "normal_packing.glsl"

layout (location = 0) in VS_OUT
{
    vec3 sNorm;
    vec2 texCoord;
    vec3 worldPos;
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
layout (location = 2) out vec3 outMetRoughAO;
layout (location = 3) out vec4 outEmissive;
#else
layout (location = 1) out vec2 outUV;
layout (location = 2) out uint outMaterialID;
layout (location = 3) out vec4 outGrad;
#ifdef DRIST
layout (location = 4) out vec4 outGradFull;
layout (location = 5) out vec3 outPosXGradFull;
layout (location = 6) out vec3 outPosYGradFull;
layout (location = 7) out vec3 outFullNormals;
#endif
#endif

layout (binding = 1, set = 0, std430) buffer Materials { MaterialData_pbrMR materials[]; };
layout (binding = 2, set = 0, std430) buffer MaterialsID { uint matID[]; };
layout (binding = 3, set = 0) uniform sampler2D all_textures[];
layout (binding = 4, set = 0) buffer MeshInfos { uvec2 o[]; } infos;

#ifndef UV_BUFFER
#include "texture_data.glsl"
#endif

vec3 getNormal(vec3 N, vec3 N_tex)
{
    // Perturb normal, see http://www.thetenthplanet.de/archives/1180
    /*vec3 q1 = dFdx(surf.worldPos);
    vec3 q2 = dFdy(surf.worldPos);
    vec2 st1 = dFdx(surf.texCoord);
    vec2 st2 = dFdy(surf.texCoord);

    N = normalize(N);
    vec3 T = normalize(q1 * st2.t - q2 * st1.t);
    vec3 B = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);*/

    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx(surf.worldPos);
    vec3 dp2 = dFdy(surf.worldPos);
    vec2 duv1 = dFdx(surf.texCoord);
    vec2 duv2 = dFdy(surf.texCoord);

    //N = -N;

    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    mat3 TBN = mat3( T * invmax, B * invmax, N );

    return normalize(TBN * N_tex);
}

void main()
{
    const uint meshOffset = infos.o[params.meshID].x;
    const uint matIdx = matID[meshOffset / 3 + gl_PrimitiveID];

    // Alpha testing has to be done here anyway
    MaterialData_pbrMR material = materials[matIdx];
    if (material.baseColorTexId >= 0) {
        float alpha = texture(all_textures[material.baseColorTexId], surf.texCoord).a;

        if (material.alphaMode == 1) {
            if (alpha < material.alphaCutoff) {
                discard;
            }
        }
    }

    vec3 N = normalize(surf.sNorm);

#ifndef UV_BUFFER
    TextureData data = sampleTextures(material, surf.texCoord, vec4(0));

    if(material.normalTexId >= 0) {
        N = getNormal(N, data.normal);
    }

    outNormal = encode_normal(N);
    outAlbedo = vec4(data.albedo.xyz, 1.0);
    outMetRoughAO = vec3(data.metallic, data.roughness, data.occlusion);
    float emissiveMultiplier = max(1, max(max(data.emissive.x, data.emissive.y), data.emissive.z));
    outEmissive = vec4(data.emissive / emissiveMultiplier, (emissiveMultiplier - 1) / 32);
#else
    if(material.normalTexId >= 0) {
        N = getNormal(N, normalize(texture(all_textures[material.normalTexId], surf.texCoord).xyz * 2.0 - 1.0));
    }
    outNormal = encode_normal(N);
    outUV = mod(surf.texCoord, vec2(1.0));
    outMaterialID = matIdx;

/*   if (material.baseColorTexId >= 0) {
        float max_lvl = textureQueryLevels(all_textures[material.baseColorTexId]) - 1;
        outTextureLod = textureQueryLod(all_textures[material.baseColorTexId], surf.texCoord).x / max_lvl;
    } else {
        outTextureLod = 0;
    }*/

    vec4 grad = vec4(dFdx(surf.texCoord), dFdy(surf.texCoord));
    outGrad = sqrt(abs(grad)) * sign(grad);
    #ifdef DRIST
    outGradFull = (grad);
    vec3 fuckme = (dFdy(surf.worldPos));
    //outGradFull = vec4(fuckme, 1.);
    //outGradFull = vec4(surf.worldPos, 1.);
    outPosXGradFull = dFdx(surf.worldPos);
    outPosYGradFull = dFdy(surf.worldPos);

    if(material.normalTexId >= 0) {
        N = getNormal(N, normalize(texture(all_textures[material.normalTexId], surf.texCoord).xyz * 2.0 - 1.0));
    }
    outFullNormals = N;
    #endif
#endif
}
