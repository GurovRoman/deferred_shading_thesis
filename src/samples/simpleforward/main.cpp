#include "simple_render.h"
#include "create_render.h"
#include "utils/glfw_window.h"

void initVulkanGLFW(std::shared_ptr<IRender> &app, GLFWwindow* window, int deviceID, bool showGUI)
{
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions  = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  if(glfwExtensions == nullptr)
  {
    std::cout << "WARNING. Can't connect Vulkan to GLFW window (glfwGetRequiredInstanceExtensions returns NULL)" << std::endl;
  }

  app->InitVulkan(glfwExtensions, glfwExtensionCount, deviceID);

  if(glfwExtensions != nullptr)
  {
    VkSurfaceKHR surface;
    VK_CHECK_RESULT(glfwCreateWindowSurface(app->GetVkInstance(), window, nullptr, &surface));

    if(showGUI)
      setupImGuiContext(window);

    app->InitPresentation(surface, showGUI);
  }
}

int main()
{
  constexpr int WIDTH = 1024;
  constexpr int HEIGHT = 1024;
  constexpr int VULKAN_DEVICE_ID = 0;

  bool showGUI = true;

  std::shared_ptr<IRender> app = CreateRender(WIDTH, HEIGHT);

  if(app == nullptr)
  {
    std::cout << "Can't create render of specified type" << std::endl;
    return 1;
  }

  auto* window = initWindow(WIDTH, HEIGHT);

  initVulkanGLFW(app, window, VULKAN_DEVICE_ID, showGUI);

  const std::vector<std::string> scenes = {
    "../resources/scenes_gltf/sponza-gltf-pbr/sponza.glb",
    "../resources/scenes_gltf/sponza_blendered/sponza.gltf",
    "../resources/scenes_gltf/sponza_blendered_low/sponza_low.gltf",
    "../resources/scenes/04_hair_balls/statex_00001.xml"
  };

  app->LoadScene(scenes[2].c_str(), false);

  mainLoop(app, window, showGUI);

  return 0;
}
