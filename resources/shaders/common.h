#ifndef VK_GRAPHICS_BASIC_COMMON_H
#define VK_GRAPHICS_BASIC_COMMON_H

#ifdef __cplusplus
#include <LiteMath.h>
using LiteMath::uint;
using LiteMath::uint2;
using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::make_float2;
using LiteMath::make_float4;

typedef uint2        uvec2;
typedef float4       vec4;
typedef float3       vec3;
typedef float2       vec2;
typedef float4x4     mat4;
#define BOOL(x) bool x; char __pad__##x[3];
#define A alignas(16)
#else
#define BOOL(x) bool x;
#define A
#endif


#define MAX_TEXTURES 256

struct PushConst
{
  mat4 model;
  uint instanceID;
};

struct UniformParams
{
  mat4 proj;
  mat4 view;
  mat4 lightMatrix;
  vec3 baseColor;
  float lightIntensity;
  vec3 lightPosition;
  float exposure;
  float time;
  float screenWidth;
  float screenHeight;
  BOOL(animateLightColor)
  uint debugFlags;
  float debugMetallic;
  float debugRoughness;
  float prefilteredCubeMipLevels;
  float IBLShadowedRatio;
  float envMapRotation;
};

struct MaterialData_pbrMR
{
  vec4 baseColor;

  float metallic;
  float roughness;
  int baseColorTexId;
  int metallicRoughnessTexId;

  vec3 emissionColor;
  int emissionTexId;

  int normalTexId;
  int occlusionTexId;
  float alphaCutoff;
  int alphaMode;

};

#endif //VK_GRAPHICS_BASIC_COMMON_H
