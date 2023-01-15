#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : enable

#include "common.h"
#include "normal_packing.glsl"


layout(location = 0) out vec4 out_fragColor;

layout(binding = 0) uniform AppData
{
    UniformParams Params;
};

layout (binding = 1) uniform samplerCube samplerIrradiance;
layout (binding = 2) uniform samplerCube prefilteredMap;
layout (binding = 3) uniform sampler2D samplerBRDFLUT;
#ifdef UV_BUFFER
layout (binding = 4, std430) buffer Materials { MaterialData_pbrMR materials[]; };
layout (binding = 5) uniform sampler2D all_textures[MAX_TEXTURES];
#endif

layout (input_attachment_index = 0, set = 1, binding = 0) uniform subpassInput inDepth;
layout (input_attachment_index = 1, set = 1, binding = 1) uniform subpassInput inNormal;
#ifndef UV_BUFFER
layout (input_attachment_index = 2, set = 1, binding = 2) uniform subpassInput inAlbedo;
layout (input_attachment_index = 3, set = 1, binding = 3) uniform subpassInput inMetRough;
#else
layout (input_attachment_index = 2, set = 1, binding = 2) uniform subpassInput inUV;
layout (input_attachment_index = 3, set = 1, binding = 3) uniform usubpassInput inMatID;
#endif

layout (binding = 0, set = 2) uniform sampler2DShadow shadowmapTex;

layout (location = 0) in vec2 outUV;

#ifdef UV_BUFFER
#include "texture_data.glsl"
#endif


const float M_PI = 3.141592653589793;

float sq(float x) { return x*x; }

// Converts a color from linear light gamma to sRGB gamma
vec3 fromLinear(vec3 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB, vec3(0.0031308));
    vec3 higher = vec3(1.055)*pow(linearRGB, vec3(1.0/2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);

    return mix(higher, lower, cutoff);
}

// Converts a color from sRGB gamma to linear light gamma
vec3 toLinear(vec3 sRGB)
{
    bvec3 cutoff = lessThan(sRGB, vec3(0.04045));
    vec3 higher = pow((sRGB + vec3(0.055))/vec3(1.055), vec3(2.4));
    vec3 lower = sRGB/vec3(12.92);

    return mix(higher, lower, cutoff);
}

vec3 tonemapLottes(vec3 x) {
    // Lottes 2016, "Advanced Techniques and Optimization of HDR Color Pipelines"
    const float a = 1.6;
    const float d = 0.977;
    const float hdrMax = 8.0;
    const float midIn = 0.18;
    const float midOut = 0.267;

    // Can be precomputed
    const float b =
    (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
    ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const float c =
    (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
    ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, vec3(a)) / (pow(x, vec3(a * d)) * b + c);
}

float calculateShadow(const vec3 lightSpacePos, const float bias) {
    vec3 pos_proj = lightSpacePos * vec3(0.5f, 0.5f, 1.f) + vec3(0.5f, 0.5f, -bias);

    float shadow_opacity = 0;

    vec2 texelSize = 1.0 / textureSize(shadowmapTex, 0);

    for (int x = -1; x <= 2; ++x) {
        for (int y = -1; y <= 2; ++y) {
            shadow_opacity += texture(shadowmapTex, pos_proj + vec3((vec2(x, y) - vec2(0.5)) * texelSize, 0.));
        }
    }
    shadow_opacity /= 16;

    /*vec2 offset = vec2(greaterThan(fract(gl_FragCoord.xy * 0.5), vec2(0.25)));
    // mod
    offset.y += offset.x;
    // y ^= x in floating point
    if (offset.y > 1.1)
        offset.y = 0;

    shadow_opacity =
        ( texture(shadowmapTex, vec3(pos_proj.xy + (offset + vec2(-1.5,  0.5)) * texelSize, pos_proj.z))
        + texture(shadowmapTex, vec3(pos_proj.xy + (offset + vec2( 0.5,  0.5)) * texelSize, pos_proj.z))
        + texture(shadowmapTex, vec3(pos_proj.xy + (offset + vec2(-1.5, -1.5)) * texelSize, pos_proj.z))
        + texture(shadowmapTex, vec3(pos_proj.xy + (offset + vec2( 0.5, -1.5)) * texelSize, pos_proj.z))) * 0.25;*/

    float pos_depth = pos_proj.z;
    return pos_depth >= 1. ? 0. : clamp(shadow_opacity, 0., 1.);
}

const float PI = 3.14159265359;

vec2 rotate(vec2 v, float a) {
    a *= PI / 180.f;
    float s = sin(a);
    float c = cos(a);
    mat2 m = mat2(c, -s, s, c);
    return m * v;
}

vec3 rotateXZ(vec3 v, float a) {
    vec3 res = v;
    res.xz = rotate(res.xz, a);
    return res;
}

vec3 toSky(vec3 v) {
    return rotateXZ(v, Params.envMapRotation);
}

vec3 fromSky(vec3 v) {
    return rotateXZ(v, -Params.envMapRotation);
}

struct PBRData {
    vec3 L;
    vec3 V;
    vec3 N;
    float dotNV;
    float dotNL;
    float dotLH;
    float dotNH;

    vec3 albedo;
    float metallic;
    float roughness;
};

// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
    return (alpha2)/(PI * denom*denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;
    float GL = dotNL / (dotNL * (1.0 - k) + k);
    float GV = dotNV / (dotNV * (1.0 - k) + k);
    return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, float metallic, vec3 albedo)
{
    vec3 F0 = mix(vec3(0.04), albedo, metallic);// * material.specular
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
    return F;
}

vec3 getIBLContribution(
    vec3 diffuseColor,
    vec3 specularColor,
    float NdotV,
    float roughness,
    vec3 N,
    vec3 reflection)
{
    float lod = (roughness * Params.prefilteredCubeMipLevels);
    // retrieve a scale and bias to F0. See [1], Figure 3
    vec3 brdf = (texture(samplerBRDFLUT, vec2(NdotV, 1.0 - roughness))).rgb;
    vec3 diffuseLight = texture(samplerIrradiance, toSky(N)).rgb;

    vec3 specularLight = textureLod(prefilteredMap, toSky(reflection), lod).rgb;

    vec3 diffuse = diffuseLight * diffuseColor;
    vec3 specular = specularLight * (specularColor * brdf.x + brdf.y);

    return diffuse + specular;
}

// Specular BRDF composition --------------------------------------------

vec3 BRDF(PBRData d, float shadowmap_visibility)
{
    vec3 f0 = vec3(0.04);

    vec3 diffuseColor = d.albedo * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - d.metallic;

    vec3 specularColor = mix(f0, d.albedo, d.metallic);


    // Light color fixed
    const vec3 lightColor = vec3(1.f);

    vec3 color = vec3(0.0);

    if (d.dotNL > 0.0)
    {
        float rroughness = max(0.05, d.roughness);
        // D = Normal distribution (Distribution of the microfacets)
        float D = D_GGX(d.dotNH, d.roughness);
        // G = Geometric shadowing term (Microfacets shadowing)
        float G = G_SchlicksmithGGX(d.dotNL, d.dotNV, rroughness);
        // F = Fresnel factor (Reflectance depending on angle of incidence)
        vec3 F = F_Schlick(d.dotNV, d.metallic, d.albedo);

        vec3 diffuseContrib = (1.0 - F) * diffuseColor;// / PI;
        vec3 spec = D * F * G / (4.0 * d.dotNL * d.dotNV);

        color += Params.lightIntensity * (diffuseContrib + spec) * d.dotNL * lightColor * shadowmap_visibility;
    }

    vec3 reflection = -normalize(reflect(d.V, d.N));
    reflection.y *= -1.0f;
    color += getIBLContribution(diffuseColor, specularColor, d.dotNV, d.roughness, d.N, reflection)
        * (1 - (1 - shadowmap_visibility) * (1 - Params.IBLShadowedRatio));

    return color;
}

void main()
{
    vec4 screenSpacePos = vec4(
    2.0 * gl_FragCoord.xy / vec2(Params.screenWidth, Params.screenHeight) - 1.0,
    subpassLoad(inDepth).r,
    1.0);

    /*if (screenSpacePos.z == 1.) {
        out_fragColor = vec4(0.192, 0.373, 0.91, 1.);
        return;
    }*/

    vec4 worldSpacePos = inverse(Params.proj * Params.view) * screenSpacePos;

    vec3 position = worldSpacePos.xyz / worldSpacePos.w;


    PBRData pbrData;

    mat4 mViewInv = inverse(Params.view);

    pbrData.V = normalize((mViewInv * vec4(0., 0., 0., 1.)).xyz - position);

    if (screenSpacePos.z == 1.) {
        out_fragColor = textureLod(prefilteredMap, toSky(pbrData.V * vec3(-1, 1, -1)), 0);
        if ((Params.debugFlags & 4) == 0)
            out_fragColor = vec4(tonemapLottes(out_fragColor.xyz * Params.exposure), 1.);
        return;
    }

#ifndef UV_BUFFER
    pbrData.N = decode_normal(subpassLoad(inNormal).xy);
    pbrData.albedo = subpassLoad(inAlbedo).rgb;
    pbrData.metallic = subpassLoad(inMetRough).r;
    pbrData.roughness = subpassLoad(inMetRough).g;
#else
    pbrData.N = decode_normal(subpassLoad(inNormal).xy);
    // Normal mapping is missing here

    vec2 uv = subpassLoad(inUV).rg;
    uint matId = subpassLoad(inMatID).x;
    MaterialData_pbrMR material = materials[matId];
    TextureData texData = sampleTextures(material, uv);

    pbrData.albedo = texData.albedo.rgb;
    pbrData.metallic = texData.metallic;
    pbrData.roughness = texData.roughness;
#endif

    if ((Params.debugFlags & 2) > 0) {
        pbrData.metallic = Params.debugMetallic;
        pbrData.roughness = Params.debugRoughness;
    }

    mat4 lightMat = Params.lightMatrix;

    // from lightspace to worldspace
    vec3 lightDir = -normalize(transpose(mat3(lightMat)) * vec3(0., 0., 1.));

    //if (gl_FragCoord.x == 0.5 && gl_FragCoord.y == 0.5)
    //    debugPrintfEXT("lightDir: %v3f\n", lightDir.xyz);


    vec3 lightSpacePos = (lightMat * vec4(position, 1.)).xyz;
    float shadowmap_visibility = calculateShadow(lightSpacePos, 0.001f);
    //shadowmap_visibility = texture(shadowmapTex, lightSpacePos * vec3(0.5f, 0.5f, 1.f) + vec3(0.5f, 0.5f, -0.010f));


    pbrData.N = normalize(transpose(mat3(Params.view)) * pbrData.N);
    pbrData.L = lightDir;

    // Precalculate vectors and dot products
    vec3 H = normalize (pbrData.V + pbrData.L);
    pbrData.dotNV = clamp(abs(dot(pbrData.N, pbrData.V)), 0.001, 1.0);
    pbrData.dotNL = clamp(dot(pbrData.N, pbrData.L), 0.0, 1.0);
    pbrData.dotLH = clamp(dot(pbrData.L, H), 0.0, 1.0);
    pbrData.dotNH = clamp(dot(pbrData.N, H), 0.0, 1.0);

    // Specular contribution
    vec3 color = BRDF(pbrData, shadowmap_visibility);


    if ((Params.debugFlags & 4) == 0)
        out_fragColor = vec4(tonemapLottes(color * Params.exposure), 1.);
    else
        out_fragColor = vec4(color, 1.);

    // display ldr overflow
    if ((Params.debugFlags & 8) > 0)
        if (any(greaterThan(out_fragColor, vec4(1.))))
            out_fragColor = vec4(0., 0., 1., 1.);
    else
        out_fragColor = clamp(out_fragColor, 0., 1.);

    if ((Params.debugFlags & 1) > 0) {
        out_fragColor = vec4(pbrData.N, 1.);
    }

    out_fragColor = vec4(fromLinear(out_fragColor.rgb), out_fragColor.a);
}
