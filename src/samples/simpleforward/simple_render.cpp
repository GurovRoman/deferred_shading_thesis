#include "utils/imGuIZMO.quat/imGuIZMOquat.h"

#include "simple_render.h"
#include "../../utils/input_definitions.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <chrono>

void SimpleRender::set_debug_name(uint64_t object, VkObjectType object_type, const char* name) {
  VkDebugUtilsObjectNameInfoEXT debugNameInfo {
    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
    .objectType = object_type,
    .objectHandle = object,
    .pObjectName = name,
  };
  vkSetDebugUtilsObjectNameEXT(m_device, &debugNameInfo);
}

SimpleRender::SimpleRender(uint32_t a_width, uint32_t a_height) : m_windowWidth(a_width), m_windowHeight(a_height)
{
#ifdef NDEBUG
  m_enableValidation = false;
#else
  m_enableValidation = true;
#endif
}

void SimpleRender::SetupDeviceFeatures()
{
  // m_enabledDeviceFeatures.fillModeNonSolid = VK_TRUE;
  m_enabledDeviceFeatures.geometryShader   = VK_TRUE;
  m_enabledDeviceFeatures.samplerAnisotropy = VK_TRUE;

  m_indexingDeviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
  m_indexingDeviceFeatures.descriptorBindingPartiallyBound = true;
  m_indexingDeviceFeatures.runtimeDescriptorArray = true;
}

void SimpleRender::SetupDeviceExtensions()
{
  m_deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  m_deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
}

void SimpleRender::SetupValidationLayers()
{
  m_validationLayers.push_back("VK_LAYER_KHRONOS_validation");
  m_validationLayers.push_back("VK_LAYER_LUNARG_monitor");
}

void SimpleRender::InitVulkan(const char** a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId)
{
  for(size_t i = 0; i < a_instanceExtensionsCount; ++i)
  {
    m_instanceExtensions.push_back(a_instanceExtensions[i]);
  }
  m_instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  SetupValidationLayers();
  VK_CHECK_RESULT(volkInitialize());
  CreateInstance();
  volkLoadInstance(m_instance);

  static_assert(VK_EXT_debug_report == 1);

  CreateDevice(a_deviceId);
  volkLoadDevice(m_device);

  m_commandPool = vk_utils::createCommandPool(m_device, m_queueFamilyIDXs.graphics,
                                              VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  m_cmdBuffersDrawMain.reserve(m_framesInFlight);
  m_cmdBuffersDrawMain = vk_utils::createCommandBuffers(m_device, m_commandPool, m_framesInFlight);

  m_frameFences.resize(m_framesInFlight);
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < m_framesInFlight; i++)
  {
    VK_CHECK_RESULT(vkCreateFence(m_device, &fenceInfo, nullptr, &m_frameFences[i]));
  }

  auto m_pCopyHelper = std::make_shared<vk_utils::PingPongCopyHelper>(m_physicalDevice, m_device, m_transferQueue,
    m_queueFamilyIDXs.transfer, 16 * 16 * 16 * 1024u);

  LoaderConfig conf = {};
  conf.load_geometry = true;
  conf.load_materials = MATERIAL_LOAD_MODE::MATERIALS_AND_TEXTURES;
  conf.instance_matrices_buffer = true;
  m_pScnMgr = std::make_shared<SceneManager>(m_device, m_physicalDevice,
    m_queueFamilyIDXs.graphics, m_pCopyHelper, conf);

  generateBRDFLUT();
  loadEnvMap("../resources/textures/Arches_E_PineTree_3k.hdr", VK_FORMAT_R16G16B16A16_SFLOAT);
  generateCubemaps();
}

void SimpleRender::InitPresentation(VkSurfaceKHR &a_surface, bool initGUI)
{
  m_surface = a_surface;

  m_presentationResources.queue = m_swapchain.CreateSwapChain(m_physicalDevice, m_device, m_surface,
    m_windowWidth, m_windowHeight, m_framesInFlight, m_vsync);
  m_presentationResources.currentFrame = 0;

  // CreateSwapChain could've changed the sizes
  m_width  = m_windowWidth * m_SSMultiplier;
  m_height = m_windowHeight * m_SSMultiplier;

  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VK_CHECK_RESULT(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_presentationResources.imageAvailable));
  VK_CHECK_RESULT(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_presentationResources.renderingFinished));

  CreateGBuffer(m_uv_buffer);
  CreateShadowmap();
  CreatePostFx();

  // create full screen quad for debug purposes
  //
  m_shadowMapDebugQuad = std::make_unique<vk_utils::QuadRenderer>(0,0, 512, 512);
  m_shadowMapDebugQuad->Create(m_device, "../resources/shaders/quad3_vert.vert.spv", "../resources/shaders/quad.frag.spv",
    vk_utils::RenderTargetInfo2D{ VkExtent2D{ m_width, m_height }, m_gbuffer.finalImage.image.format, // this is debug full screen quad
      VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }); // seems we need LOAD_OP_LOAD if we want to draw quad to part of screen

  m_shadowMapDebugQuad->SetRenderTarget(m_gbuffer.finalImage.image.view);

  m_pGUIRender = std::make_shared<ImGuiRender>(m_instance, m_device, m_physicalDevice, m_queueFamilyIDXs.graphics, m_graphicsQueue, m_swapchain);
}

void SimpleRender::CreateInstance()
{
  VkApplicationInfo appInfo = {};
  appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext              = nullptr;
  appInfo.pApplicationName   = "VkRender";
  appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.pEngineName        = "SimpleForward";
  appInfo.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
  appInfo.apiVersion         = VK_MAKE_VERSION(1, 2, 0);

  m_instance = vk_utils::createInstance(m_enableValidation, m_validationLayers, m_instanceExtensions, &appInfo);

  if (m_enableValidation)
    vk_utils::initDebugReportCallback(m_instance, &debugReportCallbackFn, &m_debugReportCallback);
}

void SimpleRender::CreateDevice(uint32_t a_deviceId)
{
  SetupDeviceExtensions();
  m_physicalDevice = vk_utils::findPhysicalDevice(m_instance, true, a_deviceId, m_deviceExtensions);

  SetupDeviceFeatures();
  m_device = vk_utils::createLogicalDevice(m_physicalDevice, m_validationLayers, m_deviceExtensions,
                                           m_enabledDeviceFeatures, m_queueFamilyIDXs,
                                           VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT, &m_indexingDeviceFeatures);

  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.graphics, 0, &m_graphicsQueue);
  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.transfer, 0, &m_transferQueue);
}

vk_utils::DescriptorMaker& SimpleRender::GetDescMaker()
{
  if(m_pBindings == nullptr)
  {
    std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 4},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_TEXTURES + 1}
    };
    m_pBindings = std::make_unique<vk_utils::DescriptorMaker>(m_device, dtypes, 9);
  }

  return *m_pBindings;
}

void SimpleRender::SetupGBufferPipeline(bool uv_buffer)
{
  auto& bindings = GetDescMaker();

  bindings.BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);
  bindings.BindBuffer(0, m_ubo, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  bindings.BindBuffer(1, m_pScnMgr->GetMaterialsBuffer());
  bindings.BindBuffer(2, m_pScnMgr->GetMaterialIDsBuffer());
  bindings.BindImageArray(3, m_pScnMgr->GetTextureViews(), m_pScnMgr->GetTextureSamplers());
  bindings.BindBuffer(4, m_pScnMgr->GetMeshInfoBuffer());
  bindings.BindBuffer(5, m_pScnMgr->GetInstanceMatBuffer());
  bindings.BindEnd(&m_graphicsDescriptorSet, &m_graphicsDescriptorSetLayout);

  auto make_deferred_pipeline = [this](const std::unordered_map<VkShaderStageFlagBits, std::string>& shader_paths)
  {
    vk_utils::GraphicsPipelineMaker maker;
    maker.LoadShaders(m_device, shader_paths);
    pipeline_data_t result;
    result.layout = maker.MakeLayout(m_device,
      {m_graphicsDescriptorSetLayout}, sizeof(pushConst));

    maker.SetDefaultState(m_width, m_height);

    std::vector<VkPipelineColorBlendAttachmentState> cba_state{ m_gbuffer.color_layers.size(),
      {
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      } };

    maker.colorBlending.attachmentCount = static_cast<uint32_t>(cba_state.size());
    maker.colorBlending.pAttachments = cba_state.data();

    result.pipeline = maker.MakePipeline(m_device, m_pScnMgr->GetPipelineVertexInputStateCreateInfo(),
      m_gbuffer.renderpass_gbuffer, {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});

    return result;
  };

  m_gBufferPipeline = make_deferred_pipeline(
    std::unordered_map<VkShaderStageFlagBits, std::string> {
      {VK_SHADER_STAGE_FRAGMENT_BIT, std::string{GBUFFER_FRAGMENT_SHADER_PATH} + (uv_buffer ? ".uvbuf" : "") + ".spv"},
      {VK_SHADER_STAGE_VERTEX_BIT, std::string{GBUFFER_VERTEX_SHADER_PATH} + ".spv"}
    });

  set_debug_name((uint64_t)m_gBufferPipeline.pipeline, VK_OBJECT_TYPE_PIPELINE, "gbuffer");
}

void SimpleRender::SetupShadingPipeline(bool uv_buffer)
{
  auto& bindings = GetDescMaker();

  bindings.BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);
  bindings.BindBuffer(0, m_ubo, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  bindings.BindImage(1, m_irradiance_map.image.view, m_irradiance_map.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  bindings.BindImage(2, m_prefiltered_map.image.view, m_prefiltered_map.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  bindings.BindImage(3, m_brdf_lut.image.view, m_brdf_lut.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  if (uv_buffer)
  {
    bindings.BindBuffer(4, m_pScnMgr->GetMaterialsBuffer());
    bindings.BindImageArray(5, m_pScnMgr->GetTextureViews(), m_pScnMgr->GetTextureSamplers());
  }
  bindings.BindEnd(&m_lightingDescriptorSet, &m_lightingDescriptorSetLayout);

  m_lightingFragmentDescriptorSet = !uv_buffer ? &m_lightingFragmentDescriptorSetReference.first : &m_lightingFragmentDescriptorSetUVBuffer.first;
  m_lightingFragmentDescriptorSetLayout = !uv_buffer ? &m_lightingFragmentDescriptorSetReference.second : &m_lightingFragmentDescriptorSetUVBuffer.second;

  // images may be recreated, so DescriptorMaker cache won't work
  if (*m_lightingFragmentDescriptorSet == VK_NULL_HANDLE)
  {
    bindings.BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings.BindImage(0, m_gbuffer.depth_stencil_layer.image.view,
      m_gbuffer.finalImage.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    for (size_t i = 0; i < m_gbuffer.color_layers.size(); ++i)
      bindings.BindImage(i + 1, m_gbuffer.color_layers[i].image.view,
        m_gbuffer.finalImage.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    bindings.BindEnd(m_lightingFragmentDescriptorSet, m_lightingFragmentDescriptorSetLayout);
  }
  else
  {
    std::vector image_infos {
      VkDescriptorImageInfo {
        .sampler = m_gbuffer.finalImage.sampler,
        .imageView = m_gbuffer.depth_stencil_layer.image.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
    };

    for (auto & color_layer : m_gbuffer.color_layers)
      image_infos.push_back({
        .sampler = m_gbuffer.finalImage.sampler,
        .imageView = color_layer.image.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      });

    std::vector<VkWriteDescriptorSet> writes(image_infos.size());

    for (std::size_t i = 0; i < image_infos.size(); ++i) {
      writes[i] = VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *m_lightingFragmentDescriptorSet,
        .dstBinding = static_cast<uint32_t>(i),
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = image_infos.data() + i
      };
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  vk_utils::GraphicsPipelineMaker maker;

  maker.LoadShaders(m_device, std::unordered_map<VkShaderStageFlagBits, std::string> {
                                {VK_SHADER_STAGE_FRAGMENT_BIT, std::string{RESOLVE_FRAGMENT_SHADER_PATH} + (uv_buffer ? ".uvbuf" : "") + ".spv"},
                                {VK_SHADER_STAGE_VERTEX_BIT, std::string{RESOLVE_VERTEX_SHADER_PATH} + ".spv"}
                              });

  m_shadingPipeline.layout = maker.MakeLayout(m_device,
    {m_lightingDescriptorSetLayout, *m_lightingFragmentDescriptorSetLayout, m_shadowMapQuadDSLayout}, sizeof(pushConst));

  maker.SetDefaultState(m_width, m_height);

  maker.rasterizer.cullMode = VK_CULL_MODE_NONE;
  maker.depthStencilTest.depthTestEnable = false;

  maker.colorBlendAttachments = {VkPipelineColorBlendAttachmentState {
    .blendEnable = false,
    .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
    .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
    .colorBlendOp = VK_BLEND_OP_ADD,
    .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
    .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
    .alphaBlendOp = VK_BLEND_OP_ADD,
    .colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
  }};

  VkPipelineVertexInputStateCreateInfo in_info {
    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    .vertexBindingDescriptionCount = 0,
    .pVertexBindingDescriptions = nullptr,
    .vertexAttributeDescriptionCount = 0,
    .pVertexAttributeDescriptions = nullptr,
  };

  m_shadingPipeline.pipeline = maker.MakePipeline(m_device, in_info,
    m_gbuffer.renderpass_resolve, {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}, vk_utils::IA_TList(), 0);

  set_debug_name((uint64_t)m_shadingPipeline.pipeline, VK_OBJECT_TYPE_PIPELINE, "resolve");
}

void SimpleRender::SetupShadowmapPipeline()
{
  auto& bindings = GetDescMaker();

  bindings.BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);
  bindings.BindBuffer(0, m_ubo, VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  bindings.BindBuffer(1, m_pScnMgr->GetInstanceMatBuffer());
  bindings.BindEnd(&m_shadowMapDescriptorSet, &m_shadowMapDescriptorSetLayout);

  // descriptors for shadowmap usage (images may be recreated, so DescriptorMaker cache won't work)
  if (m_shadowMapQuadDS == VK_NULL_HANDLE) {
    bindings.BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings.BindImage(0, m_shadow_map.image.view, m_shadow_map.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    bindings.BindEnd(&m_shadowMapQuadDS, &m_shadowMapQuadDSLayout);
  }
  else {
    VkDescriptorImageInfo image_info {
      .sampler = m_shadow_map.sampler,
      .imageView = m_shadow_map.image.view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet write {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = m_shadowMapQuadDS,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &image_info
    };

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
  }

  auto make_shadowmap_pipeline = [this](const std::unordered_map<VkShaderStageFlagBits, std::string>& shader_paths)
  {
    vk_utils::GraphicsPipelineMaker maker;
    maker.LoadShaders(m_device, shader_paths);
    pipeline_data_t result;
    result.layout = maker.MakeLayout(m_device,
      { m_shadowMapDescriptorSetLayout }, sizeof(pushConst));

    maker.SetDefaultState(m_shadowMapSize.x, m_shadowMapSize.y);

    result.pipeline = maker.MakePipeline(m_device, m_pScnMgr->GetPipelineVertexInputStateCreateInfo(),
      m_shadowMapRenderPass, {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});

    return result;
  };

  m_shadowmapPipeline = make_shadowmap_pipeline(
    std::unordered_map<VkShaderStageFlagBits, std::string> {
      {VK_SHADER_STAGE_VERTEX_BIT, std::string{SHADOWMAP_VERTEX_SHADER_PATH} + ".spv"},
      //{VK_SHADER_STAGE_FRAGMENT_BIT, std::string{SHADOWMAP_FRAGMENT_SHADER_PATH} + ".spv"}
    });
  set_debug_name((uint64_t)m_shadowmapPipeline.pipeline, VK_OBJECT_TYPE_PIPELINE, "shadowmap");
}

void SimpleRender::SetupPostfxPipeline()
{
  auto& bindings = GetDescMaker();

  // images may be recreated, so DescriptorMaker cache won't work
  if (m_postFxDescriptorSet == VK_NULL_HANDLE)
  {
    bindings.BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings.BindBuffer(0, m_ubo, nullptr, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    bindings.BindImage(1, m_gbuffer.finalImage.image.view,
      m_gbuffer.finalImage.sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    bindings.BindEnd(&m_postFxDescriptorSet, &m_postFxDescriptorSetLayout);
  }
  else
  {
    auto image_info =
      VkDescriptorImageInfo {
        .sampler = m_gbuffer.finalImage.sampler,
        .imageView = m_gbuffer.finalImage.image.view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

    auto write = VkWriteDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = m_postFxDescriptorSet,
      .dstBinding = 1,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &image_info
    };

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
  }

  vk_utils::GraphicsPipelineMaker maker;

  maker.LoadShaders(m_device, {
    { VK_SHADER_STAGE_VERTEX_BIT, std::string{ RESOLVE_VERTEX_SHADER_PATH } + ".spv" },
    { VK_SHADER_STAGE_FRAGMENT_BIT, std::string{ POSTFX_FRAGMENT_SHADER_PATH } + ".spv" },
  });

  m_postFxPipeline.layout = maker.MakeLayout(m_device,
    {m_postFxDescriptorSetLayout}, sizeof(pushConst));

  maker.SetDefaultState(m_windowWidth, m_windowHeight);

  m_postFxPipeline.pipeline = maker.MakePipeline(m_device,
    VkPipelineVertexInputStateCreateInfo {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    },
    m_postFxRenderPass, {}, vk_utils::IA_TList(), 0);
}

void SimpleRender::CreateUniformBuffer()
{
  VkMemoryRequirements memReq;
  m_ubo = vk_utils::createBuffer(m_device, sizeof(UniformParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &memReq);

  VkMemoryAllocateInfo allocateInfo = {};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.pNext = nullptr;
  allocateInfo.allocationSize = memReq.size;
  allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memReq.memoryTypeBits,
                                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                                          m_physicalDevice);
  VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_uboAlloc));

  VK_CHECK_RESULT(vkBindBufferMemory(m_device, m_ubo, m_uboAlloc, 0));

  vkMapMemory(m_device, m_uboAlloc, 0, sizeof(m_uniforms), 0, &m_uboMappedMem);

  m_uniforms.baseColor = LiteMath::float3(0.9f, 0.92f, 1.0f);
  m_uniforms.animateLightColor = false;
  m_uniforms.lightIntensity = 1.;
  m_uniforms.exposure = 1.;
  m_uniforms.IBLShadowedRatio = 1.;
  m_uniforms.envMapRotation = 0.;

  UpdateUniformBuffer(0.0f);
}

void SimpleRender::UpdateUniformBuffer(float a_time)
{
// most uniforms are updated in GUI -> SetupGUIElements()
  m_uniforms.time = a_time;
  vec3 lightDirection = LiteMath::rotate4x4Y(-m_uniforms.envMapRotation * DEG_TO_RAD) * m_light_direction;
  m_uniforms.lightMatrix = ortoMatrix(-m_light_radius, m_light_radius, -m_light_radius, m_light_radius, -m_light_length / 2, m_light_length / 2)
                           * LiteMath::lookAt({0, 0, 0},  lightDirection * 10.0f, {0, 1, 0});
  m_uniforms.screenWidth = m_width;
  m_uniforms.screenHeight = m_height;
  memcpy(m_uboMappedMem, &m_uniforms, sizeof(m_uniforms));
}

void SimpleRender::AddCmdsShadowmapPass(VkCommandBuffer a_cmdBuff, size_t frameBufferIndex) {
  vk_utils::setDefaultViewport(a_cmdBuff, static_cast<float>(m_shadowMapSize.x), static_cast<float>(m_shadowMapSize.y));
  vk_utils::setDefaultScissor(a_cmdBuff, m_shadowMapSize.x, m_shadowMapSize.y);

  VkRenderPassBeginInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = m_shadowMapRenderPass;
  renderPassInfo.framebuffer = m_shadowMapFrameBuffer;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = {m_shadowMapSize.x, m_shadowMapSize.y};

  VkClearValue clearDepth = {};
  clearDepth.depthStencil.depth   = 1.0f;
  clearDepth.depthStencil.stencil = 0;
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearDepth;

  vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
  {
    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowmapPipeline.pipeline);

    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowmapPipeline.layout, 0, 1, &m_shadowMapDescriptorSet, 0, VK_NULL_HANDLE);

    VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT);

    VkDeviceSize zero_offset = 0u;
    VkBuffer vertexBuf       = m_pScnMgr->GetVertexBuffer();
    VkBuffer indexBuf        = m_pScnMgr->GetIndexBuffer();

    vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
    vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

    for (uint32_t i = 0; i < m_pScnMgr->MeshesNum(); ++i)
    {
      pushConst.meshID = i;
      vkCmdPushConstants(a_cmdBuff, m_shadowmapPipeline.layout, stageFlags, 0, sizeof(pushConst), &pushConst);

      auto mesh_info = m_pScnMgr->GetMeshInfo(i);
      vkCmdDrawIndexed(a_cmdBuff, mesh_info.m_indNum, m_pScnMgr->GetInstanceCountForMesh(i),
        mesh_info.m_indexOffset, mesh_info.m_vertexOffset, m_pScnMgr->GetInstanceOffsetForMesh(i));
    }
  }
  vkCmdEndRenderPass(a_cmdBuff);
}

void SimpleRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, size_t frameBufferIndex)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  AddCmdsShadowmapPass(a_cmdBuff, frameBufferIndex);

  ///// draw final scene to screen
  {
    ///// Geometry pass
    vk_utils::setDefaultViewport(a_cmdBuff, static_cast<float>(m_width), static_cast<float>(m_height));
    vk_utils::setDefaultScissor(a_cmdBuff, m_width, m_height);

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_gbuffer.renderpass_gbuffer;
    renderPassInfo.framebuffer = m_gbufferPassFrameBuffer;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = { m_width, m_height };

    std::vector<VkClearValue> clearValues {m_gbuffer.color_layers.size() + 1};
    clearValues[0].depthStencil = {1.0f, 0};
    for (size_t i = 1; i < clearValues.size(); ++i)
      clearValues[i].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    renderPassInfo.clearValueCount = clearValues.size();
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferPipeline.pipeline);

      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gBufferPipeline.layout, 0, 1, &m_graphicsDescriptorSet, 0, VK_NULL_HANDLE);

      VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

      VkDeviceSize zero_offset = 0u;
      VkBuffer vertexBuf       = m_pScnMgr->GetVertexBuffer();
      VkBuffer indexBuf        = m_pScnMgr->GetIndexBuffer();

      vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
      vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

      for (uint32_t i = 0; i < m_pScnMgr->MeshesNum(); ++i)
      {
        pushConst.meshID = i;
        vkCmdPushConstants(a_cmdBuff, m_gBufferPipeline.layout, stageFlags, 0, sizeof(pushConst), &pushConst);

        auto mesh_info = m_pScnMgr->GetMeshInfo(i);
        vkCmdDrawIndexed(a_cmdBuff, mesh_info.m_indNum, m_pScnMgr->GetInstanceCountForMesh(i),
          mesh_info.m_indexOffset, mesh_info.m_vertexOffset, m_pScnMgr->GetInstanceOffsetForMesh(i));
      }
    }
    vkCmdEndRenderPass(a_cmdBuff);


    ///// Resolve pass
    renderPassInfo.renderPass = m_gbuffer.renderpass_resolve;
    renderPassInfo.framebuffer = m_resolvePassFrameBuffer;

    renderPassInfo.clearValueCount = 1;

    vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadingPipeline.pipeline);

      std::array dsets {m_lightingDescriptorSet, *m_lightingFragmentDescriptorSet, m_shadowMapQuadDS};
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadingPipeline.layout, 0,
        static_cast<uint32_t>(dsets.size()), dsets.data(), 0, VK_NULL_HANDLE);

      VkShaderStageFlags stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
      vkCmdPushConstants(a_cmdBuff, m_shadingPipeline.layout, stageFlags, 0,
        sizeof(pushConst), &pushConst);

      vkCmdDraw(a_cmdBuff, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(a_cmdBuff);


    ///// Debug quads
    if (m_shadowMapDebugQuadEnabled)
    {
      float scaleAndOffset[4] = {0.5f, 0.5f, -0.5f, +0.5f};
      m_shadowMapDebugQuad->DrawCmd(a_cmdBuff, m_shadowMapQuadDS, scaleAndOffset);
    }


    ///// Post-FX
    vk_utils::setDefaultViewport(a_cmdBuff, static_cast<float>(m_windowWidth), static_cast<float>(m_windowHeight));
    vk_utils::setDefaultScissor(a_cmdBuff, m_windowWidth, m_windowHeight);

    std::array postFxClearValues {
      VkClearValue {
        .color = {{0.0f, 0.0f, 0.0f, 1.0f}}
      },
    };
    VkRenderPassBeginInfo postFxInfo{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = m_postFxRenderPass,
      .framebuffer = m_frameBuffers[frameBufferIndex],
      .renderArea = {
         .offset = {0, 0},
         .extent = m_swapchain.GetExtent(),
      },
      .clearValueCount = static_cast<uint32_t>(postFxClearValues.size()),
      .pClearValues = postFxClearValues.data(),
    };

    vkCmdBeginRenderPass(a_cmdBuff, &postFxInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
      vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postFxPipeline.pipeline);
      vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postFxPipeline.layout,
        0, 1, &m_postFxDescriptorSet, 0, nullptr);

      vkCmdDraw(a_cmdBuff, 3, 1, 0, 0);
    }
    vkCmdEndRenderPass(a_cmdBuff);

  }

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}


void SimpleRender::CleanupPipelineAndSwapchain()
{
  if (!m_cmdBuffersDrawMain.empty())
  {
    vkFreeCommandBuffers(m_device, m_commandPool, static_cast<uint32_t>(m_cmdBuffersDrawMain.size()),
                         m_cmdBuffersDrawMain.data());
    m_cmdBuffersDrawMain.clear();
  }

  for (size_t i = 0; i < m_frameFences.size(); i++)
  {
    vkDestroyFence(m_device, m_frameFences[i], nullptr);
  }
  m_frameFences.clear();

  ClearGBuffer();
  ClearShadowmap();
  ClearPostFx();

  m_swapchain.Cleanup();
}

void SimpleRender::RecreateSwapChain()
{
  vkDeviceWaitIdle(m_device);

  ClearPipeline(m_gBufferPipeline);
  ClearPipeline(m_shadingPipeline);
  ClearPipeline(m_shadowmapPipeline);
  ClearPipeline(m_postFxPipeline);

  CleanupPipelineAndSwapchain();
  auto oldImagesNum = m_swapchain.GetImageCount();
  m_presentationResources.queue = m_swapchain.CreateSwapChain(m_physicalDevice, m_device, m_surface, m_windowWidth, m_windowHeight,
    oldImagesNum, m_vsync);

  // CreateSwapChain could've changed the sizes
  m_width  = m_windowWidth * m_SSMultiplier;
  m_height = m_windowHeight * m_SSMultiplier;

  CreateShadowmap();
  SetupShadowmapPipeline();
  CreateGBuffer(m_uv_buffer);
  SetupGBufferPipeline(m_uv_buffer);
  SetupShadingPipeline(m_uv_buffer);
  CreatePostFx();
  SetupPostfxPipeline();

  m_shadowMapDebugQuad->SetRenderTarget(m_gbuffer.finalImage.image.view);

  m_frameFences.resize(m_framesInFlight);
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < m_framesInFlight; i++)
  {
    VK_CHECK_RESULT(vkCreateFence(m_device, &fenceInfo, nullptr, &m_frameFences[i]));
  }

  m_cmdBuffersDrawMain = vk_utils::createCommandBuffers(m_device, m_commandPool, m_framesInFlight);

  m_pGUIRender->OnSwapchainChanged(m_swapchain);
}

void SimpleRender::Cleanup()
{
  vkDeviceWaitIdle(m_device);

  m_pGUIRender = nullptr;
  ImGui::DestroyContext();

  CleanupPipelineAndSwapchain();

  if(m_surface != VK_NULL_HANDLE)
  {
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    m_surface = VK_NULL_HANDLE;
  }

  m_shadowMapDebugQuad = nullptr;

  ClearPipeline(m_gBufferPipeline);
  ClearPipeline(m_shadingPipeline);
  ClearPipeline(m_shadowmapPipeline);
  ClearPipeline(m_postFxPipeline);

  // Cleanup BRDF LUT and prefiltered maps
  {
    auto clearLayer = [this](GBufferLayer& layer)
    {
      vk_utils::deleteImg(m_device, &layer.image);

      if (layer.sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, layer.sampler, VK_NULL_HANDLE);
        layer.sampler = VK_NULL_HANDLE;
      }
    };

    clearLayer(m_brdf_lut);
    clearLayer(m_prefiltered_map);
    clearLayer(m_irradiance_map);
    clearLayer(m_env_map);
  }

  if (m_presentationResources.imageAvailable != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(m_device, m_presentationResources.imageAvailable, nullptr);
    m_presentationResources.imageAvailable = VK_NULL_HANDLE;
  }
  if (m_presentationResources.renderingFinished != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(m_device, m_presentationResources.renderingFinished, nullptr);
    m_presentationResources.renderingFinished = VK_NULL_HANDLE;
  }

  if (m_commandPool != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    m_commandPool = VK_NULL_HANDLE;
  }

  if(m_ubo != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_ubo, nullptr);
    m_ubo = VK_NULL_HANDLE;
  }

  if(m_uboAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, m_uboAlloc, nullptr);
    m_uboAlloc = VK_NULL_HANDLE;
  }

  m_pBindings = nullptr;
  m_pScnMgr   = nullptr;

  if(m_device != VK_NULL_HANDLE)
  {
    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
  }

  if(m_debugReportCallback != VK_NULL_HANDLE)
  {
    vkDestroyDebugReportCallbackEXT(m_instance, m_debugReportCallback, nullptr);
    m_debugReportCallback = VK_NULL_HANDLE;
  }

  if(m_instance != VK_NULL_HANDLE)
  {
    vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
  }
}

void SimpleRender::ProcessInput(const AppInput &input)
{
  if(input.keyReleased[GLFW_KEY_1])
    m_cam_ix = 0;
  if(input.keyReleased[GLFW_KEY_2])
    m_cam_ix = 1;
  if(input.keyReleased[GLFW_KEY_3])
    m_cam_ix = 2;
  // add keyboard controls here
  // camera movement is processed separately
  if(input.keyReleased[GLFW_KEY_Q])
    m_shadowMapDebugQuadEnabled = !m_shadowMapDebugQuadEnabled;

  // recreate pipeline to reload shaders
  if(input.keyPressed[GLFW_KEY_B])
  {
#ifdef WIN32
    std::system("cd ../resources/shaders && python compile_simple_render_shaders.py");
#else
    std::system("cd ../resources/shaders && python3 compile_simple_render_shaders.py");
#endif

    /*SetupGBufferPipeline(m_uv_buffer);
    SetupShadingPipeline(m_uv_buffer);
    SetupShadowmapPipeline();*/
    RecreateSwapChain();
  }
}

void SimpleRender::UpdateCamera(const Camera* cams, uint32_t a_camsCount)
{
  assert(a_camsCount > 0);
  m_cam[m_cam_ix] = cams[0];
  UpdateView();
}

void SimpleRender::UpdateView()
{
  const auto& cam = m_cam[m_cam_ix];

  const float aspect   = float(m_width) / float(m_height);
  auto mProjFix        = OpenglToVulkanProjectionMatrixFix();
  auto mProj           = projectionMatrix(cam.fov, aspect, 0.1f, 1000.0f);
  auto mLookAt         = LiteMath::lookAt(cam.pos, cam.lookAt, cam.up);
  auto mWorldProj  = mProjFix * mProj;
  m_uniforms.proj = mWorldProj;
  m_uniforms.view = mLookAt;
}

void SimpleRender::LoadScene(const char* path, bool transpose_inst_matrices)
{
  m_pScnMgr->LoadScene(path);

  CreateUniformBuffer();
  SetupShadowmapPipeline();
  SetupGBufferPipeline(m_uv_buffer);
  SetupShadingPipeline(m_uv_buffer);
  SetupPostfxPipeline();

  for (size_t i = 0; i < 3; ++i)
  {
    auto loadedCam = m_pScnMgr->GetCamera(i);

    Camera cam;
    cam.fov      = loadedCam.fov;
    cam.pos      = float3(loadedCam.pos);
    cam.up       = float3(loadedCam.up);
    cam.lookAt   = float3(loadedCam.lookAt);
    cam.tdist    = loadedCam.farPlane;

    m_cam.push_back(std::move(cam));
  }

  UpdateView();
}

void SimpleRender::ClearPipeline(pipeline_data_t &pipeline)
{
  // if we are recreating pipeline (for example, to reload shaders)
  // we need to cleanup old pipeline
  if(pipeline.layout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, pipeline.layout, nullptr);
    pipeline.layout = VK_NULL_HANDLE;
  }
  if(pipeline.pipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
    pipeline.pipeline = VK_NULL_HANDLE;
  }
}

void SimpleRender::DrawFrame(float a_time, DrawMode a_mode)
{
  UpdateUniformBuffer(a_time);
  switch (a_mode)
  {
  case DrawMode::WITH_GUI:
    SetupGUIElements();
    DrawFrameWithGUI();
    break;
  case DrawMode::NO_GUI:
    // DrawFrameSimple();
    break;
  }
}


/////////////////////////////////

void SimpleRender::SetupGUIElements()
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  {
//    ImGui::ShowDemoWindow();
    ImGui::Begin("Simple render settings");

    // ImGui::ColorEdit3("Meshes base color", m_uniforms.baseColor.M, ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs);
    // ImGui::Checkbox("Animate light source color", &m_uniforms.animateLightColor);
    // ImGui::SliderFloat3("Light source position", m_uniforms.lightPos.M, -10.f, 10.f);

    ImGui::BeginGroup();
    static_assert(sizeof(vg::vec3) == sizeof(m_light_direction));
    vg::vec3 temp;
    std::memcpy(&temp, &m_light_direction, sizeof(m_light_direction));
    ImGui::gizmo3D("Sun Direction", temp);
    std::memcpy(&m_light_direction, &temp, sizeof(m_light_direction));

    {
      bool old = m_uv_buffer;
      ImGui::Checkbox("UV-buffer", &m_uv_buffer);
      if (old != m_uv_buffer)
        RecreateSwapChain();
      ImGui::EndGroup();
    }

    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.6f);
    ImGui::DragFloat("Light radius", &m_light_radius);
    ImGui::DragFloat("Light length", &m_light_length);
    ImGui::DragFloat("Light intensity", &m_uniforms.lightIntensity, 0.1f, 0.f, FLT_MAX);
    ImGui::DragFloat("IBL in shadow", &m_uniforms.IBLShadowedRatio, 0.05f, 0.f, 1.f);
    ImGui::DragFloat("Envmap angle", &m_uniforms.envMapRotation, 1.f, -180.f, 180.f);
    ImGui::DragFloat("Exposure", &m_uniforms.exposure, 0.1f);
    ImGui::PopItemWidth();
    ImGui::EndGroup();

    ImGui::Text("(%.2f, %.2f, %.2f)", m_light_direction.x, m_light_direction.y, m_light_direction.z);


    ImGui::Text("Debug flags:"); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag1", &m_uniforms.debugFlags, 1 << 0); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag2", &m_uniforms.debugFlags, 1 << 1); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag3", &m_uniforms.debugFlags, 1 << 2); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag4", &m_uniforms.debugFlags, 1 << 3); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag5", &m_uniforms.debugFlags, 1 << 4); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag6", &m_uniforms.debugFlags, 1 << 5); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag7", &m_uniforms.debugFlags, 1 << 6); ImGui::SameLine();
    ImGui::CheckboxFlags("##debugflag8", &m_uniforms.debugFlags, 1 << 7);

    {
      ImGui::SameLine();
      ImGui::Text("VSync:"); ImGui::SameLine();
      bool old = m_vsync;
      ImGui::Checkbox("##vsync", &m_vsync);
      if (old != m_vsync)
        RecreateSwapChain();
    }

    ImGui::PushItemWidth(ImGui::CalcItemWidth() / 2);
    ImGui::SliderFloat("Metallic", &m_uniforms.debugMetallic, 0., 1., "%.2f"); ImGui::SameLine();
    ImGui::SliderFloat("Roughness", &m_uniforms.debugRoughness, 0., 1., "%.2f");
    ImGui::PopItemWidth();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::End();
  }

  //ImGui::ShowDemoWindow();

  // Rendering
  ImGui::Render();
}

void SimpleRender::DrawFrameWithGUI()
{
  // for (size_t i = 0; i < m_framesInFlight; ++i)
  //   std::cout << vkWaitForFences(m_device, 1, &m_frameFences[(m_presentationResources.currentFrame + i + 1) % m_framesInFlight], VK_TRUE, 0);
  // std::cout << std::endl;
  vkWaitForFences(m_device, 1, &m_frameFences[m_presentationResources.currentFrame], VK_TRUE, UINT64_MAX);
  vkResetFences(m_device, 1, &m_frameFences[m_presentationResources.currentFrame]);

  uint32_t imageIdx;
  auto result = m_swapchain.AcquireNextImage(m_presentationResources.imageAvailable, &imageIdx);
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    RecreateSwapChain();
    return;
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    RUN_TIME_ERROR("Failed to acquire the next swapchain image!");
  }

  auto currentCmdBuf = m_cmdBuffersDrawMain[m_presentationResources.currentFrame];

  BuildCommandBufferSimple(currentCmdBuf, imageIdx);

  ImDrawData* pDrawData = ImGui::GetDrawData();
  auto currentGUICmdBuf = m_pGUIRender->BuildGUIRenderCommand(imageIdx, pDrawData);

  std::vector<VkCommandBuffer> submitCmdBufs = { currentCmdBuf, currentGUICmdBuf};

  VkSemaphore waitSemaphores[] = {m_presentationResources.imageAvailable};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = waitSemaphores;
  submitInfo.pWaitDstStageMask = waitStages;
  submitInfo.commandBufferCount = (uint32_t)submitCmdBufs.size();
  submitInfo.pCommandBuffers = submitCmdBufs.data();

  VkSemaphore signalSemaphores[] = {m_presentationResources.renderingFinished};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  VK_CHECK_RESULT(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_frameFences[m_presentationResources.currentFrame]));

  VkResult presentRes = m_swapchain.QueuePresent(m_presentationResources.queue, imageIdx,
    m_presentationResources.renderingFinished);

  if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR)
  {
    RecreateSwapChain();
  }
  else if (presentRes != VK_SUCCESS)
  {
    RUN_TIME_ERROR("Failed to present swapchain image");
  }

  m_presentationResources.currentFrame = (m_presentationResources.currentFrame + 1) % m_framesInFlight;

  // vkQueueWaitIdle(m_presentationResources.queue);
}

void SimpleRender::ClearGBuffer()
{
  if (m_gbuffer.renderpass_gbuffer != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(m_device, m_gbuffer.renderpass_gbuffer, nullptr);
    m_gbuffer.renderpass_gbuffer = VK_NULL_HANDLE;
  }

  if (m_gbufferPassFrameBuffer != VK_NULL_HANDLE)
  {
    vkDestroyFramebuffer(m_device, m_gbufferPassFrameBuffer, nullptr);
    m_gbufferPassFrameBuffer = VK_NULL_HANDLE;
  }

  if (m_gbuffer.renderpass_resolve != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(m_device, m_gbuffer.renderpass_resolve, nullptr);
    m_gbuffer.renderpass_resolve = VK_NULL_HANDLE;
  }

  if (m_resolvePassFrameBuffer != VK_NULL_HANDLE)
  {
    vkDestroyFramebuffer(m_device, m_resolvePassFrameBuffer, nullptr);
    m_resolvePassFrameBuffer = VK_NULL_HANDLE;
  }

  auto clearLayer = [this](GBufferLayer& layer)
  {
    vk_utils::deleteImg(m_device, &layer.image);

    if (layer.sampler != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, layer.sampler, VK_NULL_HANDLE);
      layer.sampler = VK_NULL_HANDLE;
    }
  };

  for (auto& layer : m_gbuffer.color_layers)
  {
    clearLayer(layer);
  }

  m_gbuffer.color_layers.clear();

  clearLayer(m_gbuffer.depth_stencil_layer);
  clearLayer(m_gbuffer.finalImage);
}

void SimpleRender::CreateGBuffer(bool uv_buffer)
{
  auto makeLayer = [this](VkFormat format, VkImageUsageFlagBits usage) {
    GBufferLayer result{};

    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
      result.image.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      result.image.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    vk_utils::createImgAllocAndBind(m_device, m_physicalDevice, m_width, m_height,
      format, usage, &result.image);

    return result;
  };

  std::vector<std::tuple<VkFormat, VkImageUsageFlagBits>> layers;
  if (!uv_buffer) {
    layers = {
      // Normal
      { VK_FORMAT_R16G16_SFLOAT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // Albedo
      { VK_FORMAT_R8G8B8A8_UNORM,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // MetallicRoughnessOcclusion
      { VK_FORMAT_R8G8B8A8_UNORM,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // Emissive
      { VK_FORMAT_R8G8B8A8_UNORM,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
    };
  } else {
    layers = {
      // Normal
      { VK_FORMAT_R16G16_SFLOAT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // UV
      { VK_FORMAT_R16G16_UNORM,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // MaterialID
      { VK_FORMAT_R16_UINT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // UVGradients
      { VK_FORMAT_R8G8B8A8_SNORM,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
#ifdef DRIST
      // FullUVGradients
      { VK_FORMAT_R32G32B32A32_SFLOAT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // FullPosXGradients
      { VK_FORMAT_R32G32B32A32_SFLOAT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // FullPosYGradients
      { VK_FORMAT_R32G32B32A32_SFLOAT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
      // PerfectNormals
      { VK_FORMAT_R32G32B32A32_SFLOAT,
        VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) },
#endif
    };
  }

  m_gbuffer.color_layers.reserve(layers.size());

  for (auto[format, usage] : layers) {
    m_gbuffer.color_layers.push_back(makeLayer(format, usage));
  }


  std::vector<VkFormat> depthFormats = {
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM,
  };
  VkFormat dformat;
  vk_utils::getSupportedDepthFormat(m_physicalDevice, depthFormats, &dformat);

  m_gbuffer.depth_stencil_layer = makeLayer(dformat,
    VkImageUsageFlagBits(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT));

  m_gbuffer.finalImage = makeLayer(VK_FORMAT_R8G8B8A8_UNORM, VkImageUsageFlagBits(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT));
  VkSamplerCreateInfo samplerCreateInfo = vk_utils::defaultSamplerCreateInfo(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  VK_CHECK_RESULT(vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_gbuffer.finalImage.sampler))

  // Renderpass gbuffer fill
  {
    std::vector<VkAttachmentDescription> attachmentDescs(layers.size() + 1,
      {
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      });

    // Depth layer
    {
      auto& depth = attachmentDescs[0];
      depth.format = m_gbuffer.depth_stencil_layer.image.format;
    }

    // Color GBuffer layers
    for (std::size_t i = 1; i <= layers.size(); ++i) {
      attachmentDescs[i].format = m_gbuffer.color_layers[i - 1].image.format;
    }

    std::vector<VkAttachmentReference> gBufferColorRefs(layers.size());
    for (std::size_t i = 1; i <= layers.size(); ++i)
    {
      gBufferColorRefs[i - 1] = VkAttachmentReference
        {static_cast<uint32_t>(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }

    VkAttachmentReference depthRef
      {static_cast<uint32_t>(0), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};


    std::array subpasses {
      VkSubpassDescription {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = static_cast<uint32_t>(gBufferColorRefs.size()),
        .pColorAttachments = gBufferColorRefs.data(),
        .pDepthStencilAttachment = &depthRef,
      }
    };

    // Use subpass dependencies for attachment layout transitions
    std::array dependencies {
      VkSubpassDependency {
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        // Source is gbuffer being written
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        // Destination is reading gbuffer as input attachments in fragment shader
        .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      }
    };

    VkRenderPassCreateInfo renderPassInfo {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = static_cast<uint32_t>(attachmentDescs.size()),
      .pAttachments = attachmentDescs.data(),
      .subpassCount = static_cast<uint32_t>(subpasses.size()),
      .pSubpasses = subpasses.data(),
      .dependencyCount = static_cast<uint32_t>(dependencies.size()),
      .pDependencies = dependencies.data(),
    };

    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_gbuffer.renderpass_gbuffer));
  }

  // Framebuffer gbuffer fill
  {
    std::vector<VkImageView> attachments(layers.size() + 1);
    {
      attachments[0] = m_gbuffer.depth_stencil_layer.image.view;

      for (std::size_t j = 1; j <= layers.size(); ++j)
      {
        attachments[j] = m_gbuffer.color_layers[j - 1].image.view;
      }
    }

    VkFramebufferCreateInfo fbufCreateInfo{
      .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .pNext           = nullptr,
      .renderPass      = m_gbuffer.renderpass_gbuffer,
      .attachmentCount = static_cast<uint32_t>(attachments.size()),
      .pAttachments    = attachments.data(),
      .width           = m_width,
      .height          = m_height,
      .layers          = 1,
    };
    VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &fbufCreateInfo, nullptr, &m_gbufferPassFrameBuffer));
  }

  // Renderpass resolve
  {
    std::vector<VkAttachmentDescription> attachmentDescs(1,
      {
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      });

    // Final image
    {
      auto& final = attachmentDescs[0];
      final.format = m_gbuffer.finalImage.image.format;
      final.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      final.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      final.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    }


    std::array resolveColorRefs {
      VkAttachmentReference
      {static_cast<uint32_t>(0), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };


    std::array subpasses {
      VkSubpassDescription {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = static_cast<uint32_t>(resolveColorRefs.size()),
        .pColorAttachments = resolveColorRefs.data(),
      }
    };

    // Use subpass dependencies for attachment layout transitions
    std::array dependencies {
      VkSubpassDependency {
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      },
    };

    VkRenderPassCreateInfo renderPassInfo {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = static_cast<uint32_t>(attachmentDescs.size()),
      .pAttachments = attachmentDescs.data(),
      .subpassCount = static_cast<uint32_t>(subpasses.size()),
      .pSubpasses = subpasses.data(),
      .dependencyCount = static_cast<uint32_t>(dependencies.size()),
      .pDependencies = dependencies.data(),
    };

    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_gbuffer.renderpass_resolve));
  }

  // Framebuffer resolve
  {
    std::vector<VkImageView> attachments(1);
    {
      attachments.back() = m_gbuffer.finalImage.image.view;
    }

    VkFramebufferCreateInfo fbufCreateInfo{
      .sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .pNext           = nullptr,
      .renderPass      = m_gbuffer.renderpass_resolve,
      .attachmentCount = static_cast<uint32_t>(attachments.size()),
      .pAttachments    = attachments.data(),
      .width           = m_width,
      .height          = m_height,
      .layers          = 1,
    };
    VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &fbufCreateInfo, nullptr, &m_resolvePassFrameBuffer));
  }
}

void SimpleRender::ClearShadowmap()
{
  if (m_shadowMapRenderPass != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(m_device, m_shadowMapRenderPass, nullptr);
    m_shadowMapRenderPass = VK_NULL_HANDLE;
  }

  if (m_shadowMapFrameBuffer != VK_NULL_HANDLE)
  {
    vkDestroyFramebuffer(m_device, m_shadowMapFrameBuffer, nullptr);
    m_shadowMapFrameBuffer = VK_NULL_HANDLE;
  }

  auto clearLayer = [this](GBufferLayer& layer)
  {
    vk_utils::deleteImg(m_device, &layer.image);

    if (layer.sampler != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, layer.sampler, VK_NULL_HANDLE);
      layer.sampler = VK_NULL_HANDLE;
    }
  };

  clearLayer(m_shadow_map);
}

void SimpleRender::CreateShadowmap()
{
  auto makeLayer = [this](VkFormat format, VkImageUsageFlagBits usage) {
    GBufferLayer result {};

    VkSamplerCreateInfo samplerCreateInfo = vk_utils::defaultSamplerCreateInfo(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
    samplerCreateInfo.compareEnable = VK_TRUE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_LESS;
    VK_CHECK_RESULT(vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &result.sampler));


    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) {
      result.image.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      result.image.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    vk_utils::createImgAllocAndBind(m_device, m_physicalDevice, m_shadowMapSize.x, m_shadowMapSize.y,
      format, usage, &result.image);

    return result;
  };

  std::vector<VkFormat> depthFormats = {
    VK_FORMAT_D32_SFLOAT,
    VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_FORMAT_D24_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM_S8_UINT,
    VK_FORMAT_D16_UNORM,
  };
  VkFormat dformat;
  vk_utils::getSupportedDepthFormat(m_physicalDevice, depthFormats, &dformat);

  m_shadow_map = makeLayer(dformat,
    VkImageUsageFlagBits(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT));

  set_debug_name((uint64_t)m_shadow_map.image.view, VK_OBJECT_TYPE_IMAGE_VIEW, "shadowmap");

  // Renderpass
  {
    VkAttachmentDescription depthAttachmentDesc = {
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    // Depth layer
    {
      auto& depth = depthAttachmentDesc;
      depth.format = m_shadow_map.image.format;
      depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
      depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }

    VkAttachmentReference depthRef
      {static_cast<uint32_t>(0), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};


    std::array subpasses {
      VkSubpassDescription {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 0,
        .pDepthStencilAttachment = &depthRef
      }
    };

    // Use subpass dependencies for attachment layout transitions
    std::array dependencies {
      VkSubpassDependency {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        // Semaphore waiting doesn't do any memory ops
        .srcAccessMask = {},
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      },
      VkSubpassDependency {
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      }
    };

    VkRenderPassCreateInfo renderPassInfo {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &depthAttachmentDesc,
      .subpassCount = static_cast<uint32_t>(subpasses.size()),
      .pSubpasses = subpasses.data(),
      .dependencyCount = static_cast<uint32_t>(dependencies.size()),
      .pDependencies = dependencies.data(),
    };

    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_shadowMapRenderPass));
  }

  // Framebuffer
  std::array<VkImageView, 1> attachments;

  attachments[0] = m_shadow_map.image.view;

  VkFramebufferCreateInfo fbufCreateInfo {
    .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
    .pNext = nullptr,
    .renderPass = m_shadowMapRenderPass,
    .attachmentCount = static_cast<uint32_t>(attachments.size()),
    .pAttachments = attachments.data(),
    .width = m_shadowMapSize.x,
    .height = m_shadowMapSize.y,
    .layers = 1,
  };
  VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &fbufCreateInfo, nullptr, &m_shadowMapFrameBuffer));
}

void SimpleRender::ClearPostFx()
{
  for (auto framebuf : m_frameBuffers)
  {
    vkDestroyFramebuffer(m_device, framebuf, nullptr);
  }

  m_frameBuffers.clear();

  if (m_postFxRenderPass != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(m_device, m_postFxRenderPass, nullptr);
    m_postFxRenderPass = VK_NULL_HANDLE;
  }
}

void SimpleRender::CreatePostFx()
{
  // Renderpass
  {
    std::array attachmentDescs{
      VkAttachmentDescription {
        .format = m_swapchain.GetFormat(),
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        // no stencil in present img
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      },
    };

    std::array postfxColorRefs{
      VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    };

    std::array subpasses {
      VkSubpassDescription {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = static_cast<uint32_t>(postfxColorRefs.size()),
        .pColorAttachments = postfxColorRefs.data(),
      },
    };

    // Use subpass dependencies for attachment layout transitions
    std::array dependencies {
      VkSubpassDependency {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        // Source is THE PRESENT SEMAPHORE BEING SIGNALED ON THIS PRECISE STAGE!!!!!
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        // Destination is swapchain image being filled with gbuffer resolution
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        // Semaphore waiting doesn't do any memory ops
        .srcAccessMask = {},
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      },
    };

    VkRenderPassCreateInfo renderPassInfo {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = static_cast<uint32_t>(attachmentDescs.size()),
      .pAttachments = attachmentDescs.data(),
      .subpassCount = static_cast<uint32_t>(subpasses.size()),
      .pSubpasses = subpasses.data(),
      .dependencyCount = static_cast<uint32_t>(dependencies.size()),
      .pDependencies = dependencies.data(),
    };

    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_postFxRenderPass));
  }

  // Framebuffer
  m_frameBuffers.resize(m_swapchain.GetImageCount());
  for (uint32_t i = 0; i < m_frameBuffers.size(); ++i)
  {
    std::array attachments{
      m_swapchain.GetAttachment(i).view,
    };

    VkFramebufferCreateInfo fbufCreateInfo {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .pNext = NULL,
      .renderPass = m_postFxRenderPass,
      .attachmentCount = static_cast<uint32_t>(attachments.size()),
      .pAttachments = attachments.data(),
      .width = m_windowWidth,
      .height = m_windowHeight,
      .layers = 1,
    };

    VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &fbufCreateInfo, nullptr, &m_frameBuffers[i]));
  }
}

void SimpleRender::generateBRDFLUT()
{
  auto tStart = std::chrono::high_resolution_clock::now();

  const VkFormat format = VK_FORMAT_R16G16_SFLOAT;
  const int32_t dim = 512;

  // Image
  VkImageCreateInfo imageCI{};
  imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCI.imageType = VK_IMAGE_TYPE_2D;
  imageCI.format = format;
  imageCI.extent.width = dim;
  imageCI.extent.height = dim;
  imageCI.extent.depth = 1;
  imageCI.mipLevels = 1;
  imageCI.arrayLayers = 1;
  imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

  // View
  VkImageViewCreateInfo viewCI{};
  viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewCI.format = format;
  viewCI.subresourceRange = {};
  viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewCI.subresourceRange.levelCount = 1;
  viewCI.subresourceRange.layerCount = 1;

  vk_utils::createImgAllocAndBind(m_device, m_physicalDevice, dim, dim, format, imageCI.usage, &m_brdf_lut.image, &imageCI, &viewCI);

  // Sampler
  VkSamplerCreateInfo samplerCI{};
  samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCI.magFilter = VK_FILTER_LINEAR;
  samplerCI.minFilter = VK_FILTER_LINEAR;
  samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerCI.minLod = 0.0f;
  samplerCI.maxLod = 1.0f;
  samplerCI.maxAnisotropy = 1.0f;
  samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
  VK_CHECK_RESULT(vkCreateSampler(m_device, &samplerCI, nullptr, &m_brdf_lut.sampler));

  // FB, Att, RP, Pipe, etc.
  VkAttachmentDescription attDesc{};
  // Color attachment
  attDesc.format = format;
  attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
  attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  attDesc.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

  VkSubpassDescription subpassDescription{};
  subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpassDescription.colorAttachmentCount = 1;
  subpassDescription.pColorAttachments = &colorReference;

  // Use subpass dependencies for layout transitions
  std::array<VkSubpassDependency, 2> dependencies;
  dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[0].dstSubpass = 0;
  dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
  dependencies[1].srcSubpass = 0;
  dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

  // Create the actual renderpass
  VkRenderPassCreateInfo renderPassCI{};
  renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCI.attachmentCount = 1;
  renderPassCI.pAttachments = &attDesc;
  renderPassCI.subpassCount = 1;
  renderPassCI.pSubpasses = &subpassDescription;
  renderPassCI.dependencyCount = 2;
  renderPassCI.pDependencies = dependencies.data();

  VkRenderPass renderpass;
  VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassCI, nullptr, &renderpass));

  VkFramebufferCreateInfo framebufferCI{};
  framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferCI.renderPass = renderpass;
  framebufferCI.attachmentCount = 1;
  framebufferCI.pAttachments = &m_brdf_lut.image.view;
  framebufferCI.width = dim;
  framebufferCI.height = dim;
  framebufferCI.layers = 1;

  VkFramebuffer framebuffer;
  VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferCI, nullptr, &framebuffer));


  // Desriptors
  VkDescriptorSetLayout descriptorsetlayout;
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
  descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCI, nullptr, &descriptorsetlayout));

  auto make_pipeline = [&](const std::unordered_map<VkShaderStageFlagBits, std::string>& shader_paths)
  {
         vk_utils::GraphicsPipelineMaker maker;
         maker.LoadShaders(m_device, shader_paths);
         pipeline_data_t result;

         // Pipeline layout
         result.layout = maker.MakeLayout(m_device,
           { descriptorsetlayout }, 4);

         maker.SetDefaultState(dim, dim);

         // Pipeline
         maker.depthStencilTest.depthTestEnable = VK_FALSE;
         maker.depthStencilTest.depthWriteEnable = VK_FALSE;

         VkPipelineVertexInputStateCreateInfo emptyInputStateCI{};
         emptyInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

         result.pipeline = maker.MakePipeline(m_device, emptyInputStateCI,
                                              renderpass, {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});

         return result;
  };

  pipeline_data_t pipeline = make_pipeline(
    std::unordered_map<VkShaderStageFlagBits, std::string> {
      {VK_SHADER_STAGE_VERTEX_BIT, "../resources/shaders/genbrdflut.vert.spv"},
      {VK_SHADER_STAGE_FRAGMENT_BIT, "../resources/shaders/genbrdflut.frag.spv"}
    });

  // Render
  VkClearValue clearValues[1];
  clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };

  VkRenderPassBeginInfo renderPassBeginInfo{};
  renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassBeginInfo.renderPass = renderpass;
  renderPassBeginInfo.renderArea.extent.width = dim;
  renderPassBeginInfo.renderArea.extent.height = dim;
  renderPassBeginInfo.clearValueCount = 1;
  renderPassBeginInfo.pClearValues = clearValues;
  renderPassBeginInfo.framebuffer = framebuffer;

  VkCommandBuffer cmdBuf = vk_utils::createCommandBuffer(m_device, m_commandPool);
  VkCommandBufferBeginInfo commandBufferBI{};
  commandBufferBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &commandBufferBI));
  vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

  VkViewport viewport{};
  viewport.width = (float)dim;
  viewport.height = (float)dim;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.extent.width = dim;
  scissor.extent.height = dim;

  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);
  vkCmdEndRenderPass(cmdBuf);

  VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf));

  // Flush cmd buffer
  {
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmdBuf;

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK_RESULT(vkCreateFence(m_device, &fenceInfo, nullptr, &fence));

    // Submit to the queue
    VK_CHECK_RESULT(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK_RESULT(vkWaitForFences(m_device, 1, &fence, VK_TRUE, 100000000000));

    vkDestroyFence(m_device, fence, nullptr);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);
  }

  vkQueueWaitIdle(m_graphicsQueue);

  vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, pipeline.layout, nullptr);
  vkDestroyRenderPass(m_device, renderpass, nullptr);
  vkDestroyFramebuffer(m_device, framebuffer, nullptr);
  vkDestroyDescriptorSetLayout(m_device, descriptorsetlayout, nullptr);

  // TODO: THIS SHIT MAY MATTER LATER
  // textures.lutBrdf.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  auto tEnd = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  std::cout << "Generating BRDF LUT took " << tDiff << " ms" << std::endl;
}

GBufferLayer SimpleRender::loadEnvMap(const std::string& filename,
  VkFormat format,
  VkImageUsageFlags imageUsageFlags,
  VkImageLayout imageLayout)
{
  auto info = getImageInfo(filename);
  std::vector<float> image_data = loadImageHDR(info);
  if (false) {

    // Image
    VkImageCreateInfo imageCI{};
    imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCI.imageType = VK_IMAGE_TYPE_2D;
    imageCI.format = format;
    imageCI.extent.width = info.width;
    imageCI.extent.height = info.height;
    imageCI.extent.depth = 1;
    imageCI.mipLevels = 1;
    imageCI.arrayLayers = 1;
    imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCI.usage = imageUsageFlags | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    // View
    VkImageViewCreateInfo viewCI{};
    viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCI.format = format;
    viewCI.subresourceRange = {};
    viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCI.subresourceRange.levelCount = 1;
    viewCI.subresourceRange.layerCount = 1;

    vk_utils::createImgAllocAndBind(m_device, m_physicalDevice, info.width, info.height, format, imageCI.usage, &m_env_map.image, &imageCI, &viewCI);

    // Sampler
    VkSamplerCreateInfo samplerCI{};
    samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCI.magFilter = VK_FILTER_LINEAR;
    samplerCI.minFilter = VK_FILTER_LINEAR;
    samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.minLod = 0.0f;
    samplerCI.maxLod = static_cast<float>(1);
    samplerCI.maxAnisotropy = 1.0f;
    samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(m_device, &samplerCI, nullptr, &m_env_map.sampler))

    m_pScnMgr->GetCopyHelper()->UpdateImage(m_env_map.image.image, image_data.data(), info.width, info.height, 4, imageLayout);
  }

  int mipLvls = 1 + (int) floor(log2(info.height));

  m_env_map.image = allocateColorTextureFromDataLDR(m_device, m_physicalDevice, reinterpret_cast<unsigned char*>(image_data.data()), info.width, info.height, mipLvls,
    VK_FORMAT_R32G32B32A32_SFLOAT, m_pScnMgr->GetCopyHelper(), 16);

  auto cmdBuf = vk_utils::createCommandBuffer(m_device, m_commandPool);
  vk_utils::generateMipChainCmd(cmdBuf, m_env_map.image.image, info.width, info.height, mipLvls);
  vk_utils::executeCommandBufferNow(cmdBuf, m_graphicsQueue, m_device);
  vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);

  m_env_map.sampler = vk_utils::createSampler(m_device, VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT,
    VK_BORDER_COLOR_INT_TRANSPARENT_BLACK, mipLvls);

  return {};
}

/*
  Offline generation for the cube maps used for PBR lighting
  - Irradiance cube map
  - Pre-filterd environment cubemap
*/

void SimpleRender::generateCubemaps()
{
  enum Target { IRRADIANCE = 0, PREFILTEREDENV = 1 };

  auto flushCmdBuffers = [this](const auto& cmdBuf, bool free = true) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmdBuf;

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK_RESULT(vkCreateFence(m_device, &fenceInfo, nullptr, &fence));

    // Submit to the queue
    VK_CHECK_RESULT(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK_RESULT(vkWaitForFences(m_device, 1, &fence, VK_TRUE, 100000000000));

    vkDestroyFence(m_device, fence, nullptr);
    if (free)
      vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmdBuf);
  };

  for (uint32_t target = 0; target < PREFILTEREDENV + 1; target++) {

    auto& cubemap = target ? m_prefiltered_map : m_irradiance_map;

    auto tStart = std::chrono::high_resolution_clock::now();

    VkFormat format;
    int32_t dim;

    switch (target) {
    case IRRADIANCE:
      format = VK_FORMAT_R32G32B32A32_SFLOAT;
      dim = 128;
      break;
    case PREFILTEREDENV:
      format = VK_FORMAT_R16G16B16A16_SFLOAT;
      dim = 2048;
      break;
    };

    const uint32_t numMips = static_cast<uint32_t>(floor(log2(dim))) + 1;

    // Create target cubemap
    {
      // Image
      VkImageCreateInfo imageCI{};
      imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      imageCI.imageType = VK_IMAGE_TYPE_2D;
      imageCI.format = format;
      imageCI.extent.width = dim;
      imageCI.extent.height = dim;
      imageCI.extent.depth = 1;
      imageCI.mipLevels = numMips;
      imageCI.arrayLayers = 6;
      imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
      imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
      imageCI.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

      // View
      VkImageViewCreateInfo viewCI{};
      viewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      viewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      viewCI.format = format;
      viewCI.subresourceRange = {};
      viewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      viewCI.subresourceRange.levelCount = numMips;
      viewCI.subresourceRange.layerCount = 6;

      vk_utils::createImgAllocAndBind(m_device, m_physicalDevice, dim, dim, format, imageCI.usage, &cubemap.image, &imageCI, &viewCI);

      // Sampler
      VkSamplerCreateInfo samplerCI{};
      samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      samplerCI.magFilter = VK_FILTER_LINEAR;
      samplerCI.minFilter = VK_FILTER_LINEAR;
      samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerCI.minLod = 0.0f;
      samplerCI.maxLod = static_cast<float>(numMips);
      samplerCI.maxAnisotropy = 1.0f;
      samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      VK_CHECK_RESULT(vkCreateSampler(m_device, &samplerCI, nullptr, &cubemap.sampler));
    }

    // FB, Att, RP, Pipe, etc.
    VkAttachmentDescription attDesc{};
    // Color attachment
    attDesc.format = format;
    attDesc.samples = VK_SAMPLE_COUNT_1_BIT;
    attDesc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attDesc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attDesc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attDesc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attDesc.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorReference = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpassDescription{};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;

    // Use subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Renderpass
    VkRenderPassCreateInfo renderPassCI{};
    renderPassCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCI.attachmentCount = 1;
    renderPassCI.pAttachments = &attDesc;
    renderPassCI.subpassCount = 1;
    renderPassCI.pSubpasses = &subpassDescription;
    renderPassCI.dependencyCount = 2;
    renderPassCI.pDependencies = dependencies.data();
    VkRenderPass renderpass;
    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassCI, nullptr, &renderpass));

    struct Offscreen {
      VkImage image;
      VkImageView view;
      VkDeviceMemory memory;
      VkFramebuffer framebuffer;
    } offscreen;

    // Create offscreen framebuffer
    {
      {
        vk_utils::VulkanImageMem temp{};
        // Image
        VkImageCreateInfo imageCI{};
        imageCI.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCI.imageType     = VK_IMAGE_TYPE_2D;
        imageCI.format        = format;
        imageCI.extent.width  = dim;
        imageCI.extent.height = dim;
        imageCI.extent.depth  = 1;
        imageCI.mipLevels     = 1;
        imageCI.arrayLayers   = 1;
        imageCI.samples       = VK_SAMPLE_COUNT_1_BIT;
        imageCI.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageCI.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

        // View
        VkImageViewCreateInfo viewCI{};
        viewCI.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCI.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewCI.format                          = format;
        viewCI.flags                           = 0;
        viewCI.subresourceRange                = {};
        viewCI.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCI.subresourceRange.baseMipLevel   = 0;
        viewCI.subresourceRange.levelCount     = 1;
        viewCI.subresourceRange.baseArrayLayer = 0;
        viewCI.subresourceRange.layerCount     = 1;

        vk_utils::createImgAllocAndBind(m_device, m_physicalDevice, dim, dim, format, imageCI.usage, &temp, &imageCI, &viewCI);
        offscreen.image  = temp.image;
        offscreen.view   = temp.view;
        offscreen.memory = temp.mem;
      }

      // Framebuffer
      VkFramebufferCreateInfo framebufferCI{};
      framebufferCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferCI.renderPass = renderpass;
      framebufferCI.attachmentCount = 1;
      framebufferCI.pAttachments = &offscreen.view;
      framebufferCI.width = dim;
      framebufferCI.height = dim;
      framebufferCI.layers = 1;
      VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferCI, nullptr, &offscreen.framebuffer));

      VkCommandBuffer layoutCmd = vk_utils::createCommandBuffer(m_device, m_commandPool);

      VkCommandBufferBeginInfo commandBufferBI{};
      commandBufferBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      VK_CHECK_RESULT(vkBeginCommandBuffer(layoutCmd, &commandBufferBI));

      VkImageMemoryBarrier imageMemoryBarrier{};
      imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageMemoryBarrier.image = offscreen.image;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      imageMemoryBarrier.srcAccessMask = 0;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      vkCmdPipelineBarrier(layoutCmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
      VK_CHECK_RESULT(vkEndCommandBuffer(layoutCmd));
      flushCmdBuffers(layoutCmd);
    }

    // Descriptors
    VkDescriptorSetLayout descriptorsetlayout;
    VkDescriptorSetLayoutBinding setLayoutBinding = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{};
    descriptorSetLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCI.pBindings = &setLayoutBinding;
    descriptorSetLayoutCI.bindingCount = 1;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &descriptorSetLayoutCI, nullptr, &descriptorsetlayout));

    // Descriptor Pool
    VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo descriptorPoolCI{};
    descriptorPoolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCI.poolSizeCount = 1;
    descriptorPoolCI.pPoolSizes = &poolSize;
    descriptorPoolCI.maxSets = 2;
    VkDescriptorPool descriptorpool;
    VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &descriptorPoolCI, nullptr, &descriptorpool));

    // Env_tex descriptor
    VkDescriptorImageInfo envcubeDesc{};
    envcubeDesc.sampler = m_env_map.sampler;
    envcubeDesc.imageView = m_env_map.image.view;
    envcubeDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Descriptor sets
    VkDescriptorSet descriptorset;
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
    descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocInfo.descriptorPool = descriptorpool;
    descriptorSetAllocInfo.pSetLayouts = &descriptorsetlayout;
    descriptorSetAllocInfo.descriptorSetCount = 1;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device, &descriptorSetAllocInfo, &descriptorset));
    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeDescriptorSet.descriptorCount = 1;
    writeDescriptorSet.dstSet = descriptorset;
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.pImageInfo = &envcubeDesc;
    vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSet, 0, nullptr);

    struct PushBlockIrradiance {
      LiteMath::float4x4 mvp;
      float deltaPhi = (2.0f * float(M_PI)) / 180.0f;
      float deltaTheta = (0.5f * float(M_PI)) / 64.0f;
    } pushBlockIrradiance;

    struct PushBlockPrefilterEnv {
      LiteMath::float4x4 mvp;
      float roughness;
      uint32_t numSamples = 256u;
    } pushBlockPrefilterEnv;






    auto make_pipeline = [&](const std::unordered_map<VkShaderStageFlagBits, std::string>& shader_paths)
    {
      vk_utils::GraphicsPipelineMaker maker;
      maker.LoadShaders(m_device, shader_paths);
      pipeline_data_t result;

      // Pipeline layout
      uint32_t pcRangeSize = 0;
      switch (target) {
      case IRRADIANCE:
        pcRangeSize = sizeof(PushBlockIrradiance);
        break;
      case PREFILTEREDENV:
        pcRangeSize = sizeof(PushBlockPrefilterEnv);
        break;
      };

      // Pipeline layout
      result.layout = maker.MakeLayout(m_device,
        { descriptorsetlayout }, pcRangeSize);

      maker.SetDefaultState(dim, dim);


      // Pipeline
      maker.depthStencilTest.depthTestEnable = VK_FALSE;
      maker.depthStencilTest.depthWriteEnable = VK_FALSE;

      // maker.rasterizer.cullMode = VK_CULL_MODE_NONE;

      VkPipelineVertexInputStateCreateInfo vertexInputStateCI{};
      vertexInputStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

      result.pipeline = maker.MakePipeline(m_device, vertexInputStateCI,
        renderpass, {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR});

      return result;
    };

    const std::string FRAG_SHADER_NAME = target == IRRADIANCE ? "irradiancecube.frag.spv" : "prefilterenvmap.frag.spv";
    pipeline_data_t pipeline = make_pipeline(
      std::unordered_map<VkShaderStageFlagBits, std::string> {
        {VK_SHADER_STAGE_VERTEX_BIT, "../resources/shaders/filtercube.vert.spv"},
        {VK_SHADER_STAGE_FRAGMENT_BIT, "../resources/shaders/" + FRAG_SHADER_NAME}
      });

    // Render cubemap
    VkClearValue clearValues[1];
    clearValues[0].color = { { 0.0f, 0.0f, 0.2f, 0.0f } };

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = renderpass;
    renderPassBeginInfo.framebuffer = offscreen.framebuffer;
    renderPassBeginInfo.renderArea.extent.width = dim;
    renderPassBeginInfo.renderArea.extent.height = dim;
    renderPassBeginInfo.clearValueCount = 1;
    renderPassBeginInfo.pClearValues = clearValues;

    std::vector<LiteMath::float4x4> matrices = {
      LiteMath::rotate4x4X(180.0f * DEG_TO_RAD) * LiteMath::rotate4x4Y(-90.0f * DEG_TO_RAD),
      LiteMath::rotate4x4X(180.0f * DEG_TO_RAD) * LiteMath::rotate4x4Y(90.0f * DEG_TO_RAD),
      LiteMath::rotate4x4X(-90.0f * DEG_TO_RAD),
      LiteMath::rotate4x4X(90.0f * DEG_TO_RAD),
      LiteMath::rotate4x4X(180.0f * DEG_TO_RAD),
      LiteMath::rotate4x4Z(180.0f * DEG_TO_RAD),
    };

    VkCommandBuffer cmdBuf = vk_utils::createCommandBuffer(m_device, m_commandPool);

    VkViewport viewport{};
    viewport.width = (float)dim;
    viewport.height = (float)dim;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent.width = dim;
    scissor.extent.height = dim;

    VkImageSubresourceRange subresourceRange{};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = numMips;
    subresourceRange.layerCount = 6;

    // Change image layout for all cubemap faces to transfer destination
    {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &beginInfo));
      VkImageMemoryBarrier imageMemoryBarrier{};
      imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageMemoryBarrier.image = cubemap.image.image;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      imageMemoryBarrier.srcAccessMask = 0;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      imageMemoryBarrier.subresourceRange = subresourceRange;
      vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
      VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf));
      flushCmdBuffers(cmdBuf, false);
    }

    for (uint32_t m = 0; m < numMips; m++) {
      for (uint32_t f = 0; f < 6; f++) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &beginInfo));

        viewport.width = static_cast<float>(dim * std::pow(0.5f, m));
        viewport.height = static_cast<float>(dim * std::pow(0.5f, m));
        vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
        vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

        // Render scene from cube face's point of view
        vkCmdBeginRenderPass(cmdBuf, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Pass parameters for current pass using a push constant block
        switch (target) {
        case IRRADIANCE:
          pushBlockIrradiance.mvp = perspectiveMatrix(90.f, 1.0f, 0.1f, 512.0f) * matrices[f];
          vkCmdPushConstants(cmdBuf, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushBlockIrradiance), &pushBlockIrradiance);
          break;
        case PREFILTEREDENV:
          pushBlockPrefilterEnv.mvp = perspectiveMatrix(90.f, 1.0f, 0.1f, 512.0f) * matrices[f];
          pushBlockPrefilterEnv.roughness = (float)m / (float)(numMips - 1);
          vkCmdPushConstants(cmdBuf, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushBlockPrefilterEnv), &pushBlockPrefilterEnv);
          break;
        };

        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0, 1, &descriptorset, 0, nullptr);

        VkDeviceSize offsets[1] = { 0 };

        vkCmdDraw(cmdBuf, 3 * 4, 1, 0, 0);

        vkCmdEndRenderPass(cmdBuf);

        VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = numMips;
        subresourceRange.layerCount = 6;

        {
          VkImageMemoryBarrier imageMemoryBarrier{};
          imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
          imageMemoryBarrier.image = offscreen.image;
          imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
          imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
          imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
          imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
          vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
        }

        // Copy region for transfer from framebuffer to cube face
        VkImageCopy copyRegion{};

        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.srcSubresource.baseArrayLayer = 0;
        copyRegion.srcSubresource.mipLevel = 0;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.srcOffset = { 0, 0, 0 };

        copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.dstSubresource.baseArrayLayer = f;
        copyRegion.dstSubresource.mipLevel = m;
        copyRegion.dstSubresource.layerCount = 1;
        copyRegion.dstOffset = { 0, 0, 0 };

        copyRegion.extent.width = static_cast<uint32_t>(viewport.width);
        copyRegion.extent.height = static_cast<uint32_t>(viewport.height);
        copyRegion.extent.depth = 1;

        vkCmdCopyImage(
          cmdBuf,
          offscreen.image,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          cubemap.image.image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          1,
          &copyRegion);

        {
          VkImageMemoryBarrier imageMemoryBarrier{};
          imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
          imageMemoryBarrier.image = offscreen.image;
          imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
          imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
          imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
          imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
          imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
          vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
        }
        VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf));
        flushCmdBuffers(cmdBuf, false);
      }
    }

    {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &beginInfo));

      VkImageMemoryBarrier imageMemoryBarrier{};
      imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageMemoryBarrier.image = cubemap.image.image;
      imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      imageMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      imageMemoryBarrier.subresourceRange = subresourceRange;
      vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);

      VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf));
      flushCmdBuffers(cmdBuf, false);
    }


    vkDestroyRenderPass(m_device, renderpass, nullptr);
    vkDestroyFramebuffer(m_device, offscreen.framebuffer, nullptr);
    vkFreeMemory(m_device, offscreen.memory, nullptr);
    vkDestroyImageView(m_device, offscreen.view, nullptr);
    vkDestroyImage(m_device, offscreen.image, nullptr);
    vkDestroyDescriptorPool(m_device, descriptorpool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, descriptorsetlayout, nullptr);
    vkDestroyPipeline(m_device, pipeline.pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, pipeline.layout, nullptr);

    if (target == PREFILTEREDENV) {
      m_uniforms.prefilteredCubeMipLevels = static_cast<float>(numMips);
    };

    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "Generating cube map with " << numMips << " mip levels took " << tDiff << " ms" << std::endl;
  }
}
