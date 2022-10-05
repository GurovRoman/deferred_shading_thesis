#ifndef SIMPLE_RENDER_H
#define SIMPLE_RENDER_H

#define VK_NO_PROTOTYPES

#include "../../render/scene_mgr.h"
#include "../../render/render_common.h"
#include "../../render/render_gui.h"
#include "../../../resources/shaders/common.h"
#include <geom/vk_mesh.h>
#include <vk_descriptor_sets.h>
#include <vk_fbuf_attachment.h>
#include <vk_images.h>
#include <vk_swapchain.h>
#include <vk_quad.h>
#include <string>
#include <iostream>

struct GBufferLayer
{
  vk_utils::VulkanImageMem image {};
  VkSampler sampler {VK_NULL_HANDLE};
};

struct GBuffer
{
  std::vector<GBufferLayer> color_layers;
  GBufferLayer depth_stencil_layer;
  VkRenderPass renderpass {VK_NULL_HANDLE};
};

class SimpleRender : public IRender
{
public:
  const std::string GBUFFER_VERTEX_SHADER_PATH = "../resources/shaders/gbuffer.vert";
  const std::string GBUFFER_FRAGMENT_SHADER_PATH = "../resources/shaders/gbuffer.frag";

  const std::string SHADING_VERTEX_SHADER_PATH = "../resources/shaders/shading.vert";
  const std::string SHADING_FRAGMENT_SHADER_PATH = "../resources/shaders/shading.frag";

  const std::string SHADOWMAP_VERTEX_SHADER_PATH = "../resources/shaders/depth_only.vert";
  const std::string SHADOWMAP_FRAGMENT_SHADER_PATH = "../resources/shaders/depth_only.frag";

  SimpleRender(uint32_t a_width, uint32_t a_height);
  ~SimpleRender()  { Cleanup(); };

  inline uint32_t     GetWidth()      const override { return m_width; }
  inline uint32_t     GetHeight()     const override { return m_height; }
  inline VkInstance   GetVkInstance() const override { return m_instance; }
  void InitVulkan(const char** a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId) override;

  void InitPresentation(VkSurfaceKHR& a_surface, bool initGUI) override;

  void ProcessInput(const AppInput& input) override;
  void UpdateCamera(const Camera* cams, uint32_t a_camsCount) override;
  Camera GetCurrentCamera() override {return m_cam;}
  void UpdateView();

  void LoadScene(const char *path, bool transpose_inst_matrices) override;
  void DrawFrame(float a_time, DrawMode a_mode) override;

  void ClearPipeline(pipeline_data_t &pipeline);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // debugging utils
  //
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
    VkDebugReportFlagsEXT                       flags,
    VkDebugReportObjectTypeEXT                  objectType,
    uint64_t                                    object,
    size_t                                      location,
    int32_t                                     messageCode,
    const char* pLayerPrefix,
    const char* pMessage,
    void* pUserData)
  {
    (void)flags;
    (void)objectType;
    (void)object;
    (void)location;
    (void)messageCode;
    (void)pUserData;
    std::cout << pLayerPrefix << ": " << pMessage << std::endl;
    return VK_FALSE;
  }

  VkDebugReportCallbackEXT m_debugReportCallback = nullptr;
protected:

  VkInstance       m_instance       = VK_NULL_HANDLE;
  VkCommandPool    m_commandPool    = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice         m_device         = VK_NULL_HANDLE;
  VkQueue          m_graphicsQueue  = VK_NULL_HANDLE;
  VkQueue          m_transferQueue  = VK_NULL_HANDLE;

  vk_utils::QueueFID_T m_queueFamilyIDXs {UINT32_MAX, UINT32_MAX, UINT32_MAX};

  struct
  {
    uint32_t    currentFrame      = 0u;
    VkQueue     queue             = VK_NULL_HANDLE;
    VkSemaphore imageAvailable    = VK_NULL_HANDLE;
    VkSemaphore renderingFinished = VK_NULL_HANDLE;
  } m_presentationResources;

  std::vector<VkFence> m_frameFences;
  std::vector<VkCommandBuffer> m_cmdBuffersDrawMain;

  PushConst pushConst;

  UniformParams m_uniforms {};
  VkBuffer m_ubo = VK_NULL_HANDLE;
  VkDeviceMemory m_uboAlloc = VK_NULL_HANDLE;
  void* m_uboMappedMem = nullptr;

  pipeline_data_t m_gBufferPipeline {};
  pipeline_data_t m_shadingPipeline {};
  pipeline_data_t m_shadowmapPipeline {};

  VkDescriptorSet m_graphicsDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_graphicsDescriptorSetLayout = VK_NULL_HANDLE;

  VkDescriptorSet m_lightingDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_lightingDescriptorSetLayout = VK_NULL_HANDLE;

  VkDescriptorSet m_lightingFragmentDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_lightingFragmentDescriptorSetLayout = VK_NULL_HANDLE;

  VkDescriptorSet m_shadowMapDescriptorSet             = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_shadowMapDescriptorSetLayout = VK_NULL_HANDLE;

  std::shared_ptr<vk_utils::DescriptorMaker> m_pBindings = nullptr;
  vk_utils::DescriptorMaker& GetDescMaker();

  // *** presentation
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VulkanSwapChain m_swapchain;
  std::vector<VkFramebuffer> m_frameBuffers;
  // ***

  // *** GUI
  std::shared_ptr<IRenderGUI> m_pGUIRender;
  virtual void SetupGUIElements();
  void DrawFrameWithGUI();
  //

  Camera   m_cam;
  uint32_t m_width  = 1024u;
  uint32_t m_height = 1024u;
  uint32_t m_framesInFlight  = 2u;
  bool m_vsync = true;

  VkPhysicalDeviceFeatures m_enabledDeviceFeatures = {};
  std::vector<const char*> m_deviceExtensions      = {};
  std::vector<const char*> m_instanceExtensions    = {};

  bool m_enableValidation;
  std::vector<const char*> m_validationLayers;

  std::shared_ptr<SceneManager> m_pScnMgr;

  GBuffer m_gbuffer;

  GBufferLayer m_shadow_map;
  VkFramebuffer m_shadowMapFrameBuffer;
  VkRenderPass m_shadowMapRenderPass;

  LiteMath::uint2 m_shadowMapSize = {8192, 8192};

  bool m_shadowMapDebugQuadEnabled = false;
  std::unique_ptr<vk_utils::IQuad> m_shadowMapDebugQuad;
  VkDescriptorSet       m_shadowMapQuadDS;
  VkDescriptorSetLayout m_shadowMapQuadDSLayout = nullptr;

  vec3 m_light_direction = LiteMath::normalize({-0.8, -0.44, -0.42});
  float m_light_radius = 100;
  float m_light_length = 150;

  void CreateInstance();
  void CreateDevice(uint32_t a_deviceId);

  void BuildCommandBufferSimple(VkCommandBuffer cmdBuff, size_t frameBufferIndex);
  void AddCmdsShadowmapPass(VkCommandBuffer a_cmdBuff, size_t frameBufferIndex);

  void SetupGBufferPipeline();
  void SetupShadingPipeline();
  void SetupShadowmapPipeline();
  void CleanupPipelineAndSwapchain();
  void RecreateSwapChain();

  void CreateUniformBuffer();
  void UpdateUniformBuffer(float a_time);

  void Cleanup();

  void SetupDeviceFeatures();
  void SetupDeviceExtensions();
  void SetupValidationLayers();

  void ClearGBuffer();
  void CreateGBuffer();

  void ClearShadowmap();
  void CreateShadowmap();

  void generateBRDFLUT();
  void generateCubemaps();

  GBufferLayer loadEnvMap(const std::string& filename,
    VkFormat format,
    VkImageUsageFlags imageUsageFlags = VK_IMAGE_USAGE_SAMPLED_BIT,
    VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  GBufferLayer m_brdf_lut;
  GBufferLayer m_env_map;
  GBufferLayer m_irradiance_map;
  GBufferLayer m_prefiltered_map;
};


#endif //SIMPLE_RENDER_H
