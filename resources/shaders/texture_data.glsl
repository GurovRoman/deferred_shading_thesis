#include "common.h"

struct TextureData
{
    vec4 albedo;
    vec3 normal;
    vec4 emissive;
    float occlusion;

    float metallic;
    float roughness;
};

vec4 sampleTexture(sampler2D tex, vec2 uv, float lod, vec4 dUV) {
#ifndef UV_BUFFER
    return texture(tex, uv);
#else
    lod = textureQueryLevels(tex) - lod - 1;
    return textureGrad(tex, uv, dUV.xy, dUV.zw);
#endif
}

// all_textures array cannot be passed as argument :(
TextureData sampleTextures(MaterialData_pbrMR material, vec2 uv) {
    TextureData data;

#ifndef UV_BUFFER
    float lod = 0;
    vec4 dUV = vec4(0);
#else
    float lod = subpassLoad(inTextureLod).r;
    if (material.baseColorTexId >= 0) {
        float max_lvl = textureQueryLevels(all_textures[material.baseColorTexId]) - 1;
        lod = (1 - lod) * max_lvl;
    }

    vec4 dUV = subpassLoad(inGrad);
    dUV = (dUV * dUV) * sign(dUV);
#endif

    data.albedo = material.baseColor;
    if(material.baseColorTexId >= 0) {
        const vec4 tex_val = sampleTexture(all_textures[material.baseColorTexId], uv, lod, dUV);
        data.albedo.xyz *= tex_val.xyz;
        data.albedo.a    = tex_val.a;
    }

    data.normal = vec3(0.);
    if(material.normalTexId >= 0) {
        data.normal = normalize(sampleTexture(all_textures[material.normalTexId], uv, lod, dUV).xyz * 2.0 - vec3(1.0));
    }

    data.metallic  = material.metallic;
    data.roughness = material.roughness;
    if (material.metallicRoughnessTexId >= 0) {
        data.metallic = sampleTexture(all_textures[material.metallicRoughnessTexId], uv, lod, dUV).b;
        data.roughness = sampleTexture(all_textures[material.metallicRoughnessTexId], uv, lod, dUV).g;
    }

    return data;
}