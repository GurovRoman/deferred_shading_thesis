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


// all_textures array cannot be passed as argument :(
TextureData sampleTextures(MaterialData_pbrMR material, vec2 uv) {
    TextureData data;

    data.albedo = material.baseColor;
    if(material.baseColorTexId >= 0) {
        const vec4 tex_val = texture(all_textures[material.baseColorTexId], uv);
        data.albedo.xyz *= tex_val.xyz;
        data.albedo.a    = tex_val.a;
    }

    data.normal = normalize(texture(all_textures[material.normalTexId], uv).xyz * 2.0 - vec3(1.0));

    data.metallic  = material.metallic;
    data.roughness = material.roughness;
    if (material.metallicRoughnessTexId >= 0) {
        data.metallic = texture(all_textures[material.metallicRoughnessTexId], uv).b;
        data.roughness = texture(all_textures[material.metallicRoughnessTexId], uv).g;
    }

    return data;
}