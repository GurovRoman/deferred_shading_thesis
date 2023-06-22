#include "common.h"

struct TextureData
{
    vec4 albedo;
    vec3 normal;
    vec3 emissive;

    float metallic;
    float roughness;
    float occlusion;
};

#ifdef UV_BUFFER
#ifdef DRIST
struct GradData {
    vec4 dUV;
    vec3 dPdx;
    vec3 dPdy;
};

vec3 sampleViewPos(vec2 screenUvPos) {
    vec4 screenPos = vec4(
        2.0 * screenUvPos - 1.0,
        textureLod(inDepth, screenUvPos, 0).r,
        1.0);
    vec4 viewPos_ = inverse(Params.proj) * screenPos;
    return viewPos_.xyz / viewPos_.w;
}

GradData reconstructGradients(vec3 view_pos, vec2 uv, uint matId) {
    GradData data;
    vec2 screenDelta = 1. / vec2(Params.screenWidth, Params.screenHeight);
    vec2 screenUV = gl_FragCoord.xy * screenDelta;

    data.dUV = textureLod(inGradFull, screenUV, 0);
    //    data.dUV = (data.dUV * data.dUV) * sign(data.dUV);

    vec4 dUV = vec4(dFdxFine(uv), dFdyFine(uv));
    /*if (length(dUV - data.dUV) < 0.01) {
        data.dUV = dUV;
    }*/

    if ((Params.debugFlags & 8) > 0) {
        data.dPdx = dFdxFine(view_pos);
        data.dPdy = dFdyFine(view_pos);
        data.dUV = dUV;
        return data;
    }

    /*vec2 screenUVoffset = screenUV + screenDelta;
    vec3 position_ = sampleViewPos(screenUVoffset);
    data.dPdx = view_pos - position_;

    vec2 uv_ = textureLod(inUV, screenUVoffset, 0).rg;
    data.dUV.xy = uv - uv_;*/

    ivec2 neighborDir = -((ivec2(gl_FragCoord.xy) % 2) * 2 - 1);
    {
        vec2 screenUVoffset = screenUV - vec2(screenDelta.x, 0) * neighborDir.x;
        vec3 position_ = sampleViewPos(screenUVoffset);

        data.dPdx = (view_pos - position_) * neighborDir.x;

        vec2 uv_ = textureLod(inUV, screenUVoffset, 0).rg;
        data.dUV.xy = (uv - uv_) * neighborDir.x;
    }
    {
        vec2 screenUVoffset = screenUV - vec2(0, screenDelta.y) * neighborDir.y;
        vec3 position_ = sampleViewPos(screenUVoffset);

        data.dPdy = (view_pos - position_) * neighborDir.y;

        vec2 uv_ = textureLod(inUV, screenUVoffset, 0).rg;
        data.dUV.zw = (uv - uv_) * neighborDir.y;
    }
    /*if (dFdx(matId) < 0.9) {
        data.dPdx = dFdxFine(view_pos);
    } else {
        vec2 screenUVoffset = screenUV - dFdx(screenUV) * neighborDir.x;
        vec4 screenSpacePos = vec4(
        2.0 * screenUVoffset - 1.0,
        textureLod(inDepth, screenUVoffset, 0).r,
        1.0);
        vec4 camspacePos_ = inverse(Params.proj) * screenSpacePos;
        vec3 position_ = camspacePos_.xyz / camspacePos_.w;

        data.dPdx = (view_pos - position_) * neighborDir.x;
    }

    if (dFdy(matId) < 0.9) {
        data.dPdy = dFdyFine(view_pos);
    } else {
        vec2 screenUVoffset = screenUV - dFdy(screenUV) * neighborDir.y;
        vec4 screenSpacePos = vec4(
        2.0 * screenUVoffset - 1.0,
        textureLod(inDepth, screenUVoffset, 0).r,
        1.0);
        vec4 camspacePos_ = inverse(Params.proj) * screenSpacePos;
        vec3 position_ = camspacePos_.xyz / camspacePos_.w;

        data.dPdy = -(position_ - view_pos) * neighborDir.y;
    }*/
    /*if (dFdx(matId) < 0.9) {
        if (length(dUV.xy) < length(data.dUV.xy)) {
            data.dPdx = dFdx(view_pos);
            data.dUV.xy = dUV.xy;
        }
    }
    if (dFdy(matId) < 0.9) {
        if (length(dUV.zw) < length(data.dUV.zw)) {
            data.dPdy = dFdy(view_pos);
            data.dUV.zw = dUV.zw;
        }
    }*/


    //data.dPdx = textureLod(inPosXGradFull, screenUV, 0).xyz;
    //data.dPdy = textureLod(inPosYGradFull, screenUV, 0).xyz;

    vec4 dUVFull = textureLod(inGradFull, screenUV, 0);
    vec3 dPFullX = textureLod(inPosXGradFull, screenUV, 0).xyz;
    vec3 dPFullY = textureLod(inPosYGradFull, screenUV, 0).xyz;

    if (length(dUV.xy - dUVFull.xy) < length(data.dUV.xy - dUVFull.xy)) {
        data.dPdx = dFdx(view_pos);
        data.dUV.xy = dUV.xy;
    }

    if (length(dUV.zw - dUVFull.zw) < length(data.dUV.zw - dUVFull.zw)) {
        data.dPdy = dFdy(view_pos);
        data.dUV.zw = dUV.zw;
    }

    GradData gradsX[6];
    GradData gradsY[6];
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 2; ++j) {
            uint ix = i * 3 + j;

            vec2 screenUVoffset1 = screenUV - screenDelta + screenDelta * vec2(i, j);
            vec2 screenUVoffset2 = screenUVoffset1 + vec2(screenDelta.x, 0);
            vec3 position = sampleViewPos(screenUVoffset2);
            vec3 position_ = sampleViewPos(screenUVoffset1);

            gradsX[ix].dPdx = (view_pos - position_);

            vec2 uv = textureLod(inUV, screenUVoffset2, 0).rg;
            vec2 uv_ = textureLod(inUV, screenUVoffset1, 0).rg;
            gradsX[ix].dUV.xy = (uv - uv_);
        }
    }
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 2; ++j) {
            uint ix = i * 3 + j;

            vec2 screenUVoffset1 = screenUV - screenDelta + screenDelta * vec2(j, i);
            vec2 screenUVoffset2 = screenUVoffset1 + vec2(0, screenDelta.y);
            vec3 position = sampleViewPos(screenUVoffset2);
            vec3 position_ = sampleViewPos(screenUVoffset1);

            gradsY[ix].dPdx = (view_pos - position_);

            vec2 uv = textureLod(inUV, screenUVoffset2, 0).rg;
            vec2 uv_ = textureLod(inUV, screenUVoffset1, 0).rg;
            gradsY[ix].dUV.xy = (uv - uv_);
        }
    }


    for (int i = 0; i < 6; ++i) {
        if (length(gradsX[i].dUV.xy - dUVFull.xy) < length(data.dUV.xy - dUVFull.xy)) {
            data.dPdx = gradsX[i].dPdx;
            data.dUV.xy = gradsX[i].dUV.xy;
        }
    }

    for (int i = 0; i < 6; ++i) {
        if (length(gradsY[i].dUV.zw - dUVFull.zw) < length(data.dUV.zw - dUVFull.zw)) {
            data.dPdy = gradsY[i].dPdy;
            data.dUV.zw = gradsY[i].dUV.zw;
        }
    }

    return data;
}

vec3 getNormalWithGrad(vec3 N, vec3 N_tex, GradData grad)
{
    // Perturb normal, see http://www.thetenthplanet.de/archives/1180

    // get edge vectors of the pixel triangle
    vec3 dp1 = grad.dPdx;
    vec3 dp2 = grad.dPdy;
    vec2 duv1 = grad.dUV.xy;
    vec2 duv2 = grad.dUV.zw;

    N = normalize(N);
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
#endif
#endif

vec4 sampleTexture(sampler2D tex, vec2 uv, vec4 dUV) {
#ifndef UV_BUFFER
    return texture(tex, uv);
#else
    return textureGrad(tex, uv, dUV.xy, dUV.zw);
#endif
}

// all_textures array cannot be passed as argument :(
TextureData sampleTextures(MaterialData_pbrMR material, vec2 uv, vec4 dUV) {
    TextureData data;

    data.albedo = material.baseColor;
    if(material.baseColorTexId >= 0) {
        const vec4 tex_val = sampleTexture(all_textures[material.baseColorTexId], uv, dUV);
        data.albedo.xyz *= tex_val.xyz;
        data.albedo.a    = tex_val.a;
    }

    data.normal = vec3(0.);
    if(material.normalTexId >= 0) {
        data.normal = normalize(sampleTexture(all_textures[material.normalTexId], uv, dUV).xyz * 2.0 - 1.0);
    }

    data.metallic  = material.metallic;
    data.roughness = material.roughness;
    if (material.metallicRoughnessTexId >= 0) {
        vec3 texVal = sampleTexture(all_textures[material.metallicRoughnessTexId], uv, dUV).bgr;
        data.metallic = texVal.x;
        data.roughness = texVal.y;
        data.occlusion = texVal.z;
    }

    if (material.occlusionTexId >= 0) {
        if (material.occlusionTexId != material.metallicRoughnessTexId) {
            data.occlusion = sampleTexture(all_textures[material.occlusionTexId], uv, dUV).r;
        }
    } else {
        data.occlusion = 1;
    }

    data.emissive = material.emissionColor;
    if (material.emissionTexId >= 0) {
        data.emissive *= sampleTexture(all_textures[material.emissionTexId], uv, dUV).rgb;
    }

    return data;
}