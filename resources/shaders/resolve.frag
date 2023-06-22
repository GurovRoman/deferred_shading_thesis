#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
//#extension GL_EXT_debug_printf : enable

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
layout (binding = 5) uniform sampler2D all_textures[];
#endif

layout (set = 1, binding = 0) uniform sampler2D inDepth;
layout (set = 1, binding = 1) uniform sampler2D inNormal;
#ifndef UV_BUFFER
layout (set = 1, binding = 2) uniform sampler2D inAlbedo;
layout (set = 1, binding = 3) uniform sampler2D inMetRoughAO;
layout (set = 1, binding = 4) uniform sampler2D inEmissive;
#else
layout (set = 1, binding = 2) uniform sampler2D inUV;
layout (set = 1, binding = 3) uniform usampler2D inMatID;
layout (set = 1, binding = 4) uniform sampler2D inGrad;
#ifdef DRIST
layout (set = 1, binding = 5) uniform sampler2D inGradFull;
layout (set = 1, binding = 6) uniform sampler2D inPosXGradFull;
layout (set = 1, binding = 7) uniform sampler2D inPosYGradFull;
layout (set = 1, binding = 8) uniform sampler2D inFullNormals;
#endif
#endif

layout (binding = 0, set = 2) uniform sampler2DShadow shadowmapTex;

layout (location = 0) in vec2 outUV;

#ifdef UV_BUFFER
#include "texture_data.glsl"
#endif


const float M_PI = 3.141592653589793;

float sq(float x) { return x*x; }

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
    vec3 emissive;
    float metallic;
    float roughness;
    float occlusion;
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

    return color * d.occlusion + d.emissive;
}

void main()
{
    vec2 screenUV = gl_FragCoord.xy / vec2(Params.screenWidth, Params.screenHeight);

    vec4 screenSpacePos = vec4(
    2.0 * screenUV - 1.0,
    textureLod(inDepth, screenUV, 0).r,
    1.0);

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
    pbrData.N = decode_normal(texture(inNormal, screenUV).xy);
    pbrData.albedo = texture(inAlbedo, screenUV).rgb;
    vec3 texVal = texture(inMetRoughAO, screenUV).rgb;
    pbrData.metallic = texVal.r;
    pbrData.roughness = texVal.g;
    pbrData.occlusion = texVal.b;

    vec4 emissive = texture(inEmissive, screenUV);
    pbrData.emissive = emissive.rgb * (1 + emissive.a * 32);
#else
    pbrData.N = decode_normal(texture(inNormal, screenUV).xy);

    vec2 uv = textureLod(inUV, screenUV, 0).rg;
    uint matId = textureLod(inMatID, screenUV, 0).x;

    vec3 camspace_position = (Params.view * vec4(position, 1)).xyz;
    //GradData grad = reconstructGradients(camspace_position, uv, matId);

    vec4 fdUV = textureLod(inGrad, screenUV, 0);
    fdUV = (fdUV * fdUV) * sign(fdUV);

    MaterialData_pbrMR material = materials[matId];
    TextureData texData = sampleTextures(material, uv, fdUV);

    //pbrData.N = getNormalWithGrad(pbrData.N, texData.normal, grad);

    pbrData.albedo = texData.albedo.rgb;
    pbrData.metallic = texData.metallic;
    pbrData.roughness = texData.roughness;
    pbrData.occlusion = texData.occlusion;
    pbrData.emissive = texData.emissive;
#endif
    vec3 N_restored = pbrData.N;

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

    out_fragColor = clamp(out_fragColor, 0., 1.);

    if ((Params.debugFlags & 1) > 0) {
        out_fragColor = vec4(vec3(-N_restored.z), 1.);
    }

    if ((Params.debugFlags & 16) > 0) {
        out_fragColor = vec4(pbrData.albedo, 1.);
    }

#ifdef UV_BUFFER
    #ifdef DRIST
    if ((Params.debugFlags & 32) > 0) {
        vec3 NFull = textureLod(inFullNormals, screenUV, 0).xyz;

        out_fragColor = vec4(abs(N_restored - NFull) * 1, 1);
        //out_fragColor = vec4(fromLinear(out_fragColor.rgb), out_fragColor.a);
        return;
    }

    if ((Params.debugFlags & 128) > 0) {
        vec4 dUVFull = textureLod(inGradFull, screenUV, 0);
        vec4 dPFullX = textureLod(inPosXGradFull, screenUV, 0);
        vec4 dPFullY = textureLod(inPosYGradFull, screenUV, 0);
        vec4 dP = vec4(dFdy(camspace_position), 0);
        vec4 dUV = vec4(dFdx(uv), dFdy(uv));

        if ((Params.debugFlags & 64) > 0) {
            out_fragColor = vec4(abs(decode_normal(texture(inNormal, screenUV).xy)), 1);
            out_fragColor = vec4(vec3(abs(dFdx(matId)), abs(dFdy(matId)), 0), 1.);
            //out_fragColor = vec4(all(equal(dUV, vec4(0))));
        } else {
            out_fragColor = vec4(abs(grad.dUV - dUVFull) * 100);
            //out_fragColor = vec4(abs(vec4(camspace_position, 0) - dUVFull) * 100);
        }
        return;
    }
    #endif
#endif

    out_fragColor = vec4(fromLinear(out_fragColor.rgb), out_fragColor.a);
}
