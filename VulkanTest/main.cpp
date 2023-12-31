//
//  main.cpp
//  VulkanTest
//
//  Created by Sheridan on 2023-08-21.
//

#include "main.hpp"
#include "glsl.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#include <glslang/Include/glslang_c_interface.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // required for vulkan
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

// C headers
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
#include <unistd.h>

// C++ headers
#include <thread>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <set>

#pragma clang diagnostic pop

int printlog(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int result = vprintf(fmt, args);
    va_end(args);
    const auto len = strlen(fmt);
    if (len > 0) {
        if (fmt[len - 1] != '\n') {
            printf("\n");
        }
    }
    return result;
}

struct QueueFamilyInfo {
    uint32_t index = 0;
    uint32_t queues = 0;
    bool hasPresent = false;
    bool hasGraphics = false;
    bool hasTransfer = false;
};
using QueueFamilies = std::vector<QueueFamilyInfo>;

struct CommandPool {
    VkCommandPool handle;
    std::vector<VkCommandBuffer> buffers;
};
    
struct Vertex {
    alignas(16) glm::vec4 position;
    alignas(16) glm::vec4 color;
    alignas(8) glm::vec2 texture;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }
    
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texture);

        return attributeDescriptions;
    }
};

struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    
    void destroy(VkDevice device) {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
};

struct Image {
    VkImage image{};
    VkImageView view{};
    VkDeviceMemory memory{};
    uint32_t mipLevels{};
    uint32_t width{};
    uint32_t height{};
    uint32_t depth{};
    VkFormat format{};
    
    void destroy(VkDevice device) {
        format = VK_FORMAT_UNDEFINED;
        width = 0;
        height = 0;
        depth = 0;
        mipLevels = 0;
        if (view) {
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
        if (image) {
            vkDestroyImage(device, image, nullptr);
            image = VK_NULL_HANDLE;
        }
        if (memory) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
};

struct Texture {
    Image image{};
    VkSampler sampler{};
    
    void destroy(VkDevice device) {
        if (sampler) {
            vkDestroySampler(device, sampler, nullptr);
            sampler = VK_NULL_HANDLE;
        }
        image.destroy(device);
    }
};

struct Mesh {
    Buffer vertexBuffer{};
    Buffer indexBuffer{};
    
    void destroy(VkDevice device) {
        indexBuffer.destroy(device);
        vertexBuffer.destroy(device);
    }
};

struct UniformBuffer {
    Buffer buffer{};
    void* data{};
    
    void destroy(VkDevice device) {
        buffer.destroy(device);
        // data pointer is now invalid
        data = nullptr;
    }
};

struct UniformBufferObject {
    // scalars have to be aligned by N (= 4 bytes given 32 bit floats).
    // vec2 must be aligned by 2N (= 8 bytes)
    // vec3 or vec4 must be aligned by 4N (= 16 bytes)
    // nested structure must be aligned by the base alignment of its members rounded up to a multiple of 16.
    // mat4 matrix must have the same alignment as a vec4.
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap15.html#interfaces-resources-layout
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 model;
};

struct DescriptorPool {
    VkDescriptorPool pool{};
    std::vector<VkDescriptorSet> sets{};
    
    void destroy(VkDevice device) {
        vkDestroyDescriptorPool(device, pool, nullptr);
        pool = VK_NULL_HANDLE;
        sets.clear();
    }
};

const char glsl_vert[] = 
    "layout(location = 0) in vec4 inPosition;"
    "layout(location = 1) in vec4 inColor;"
    "layout(location = 2) in vec2 inTexCoord;"
    
    "layout(location = 0) out vec4 outColor;"
    "layout(location = 1) out vec2 outTexCoord;"
    
    "layout(binding = 0) uniform UniformBufferObject {"
    "    mat4 proj;"
    "    mat4 view;"
    "    mat4 model;"
    "} ubo;"
    
    "void main() {"
    "    gl_Position = ubo.proj * ubo.view * ubo.model * inPosition;"
    "    outColor = inColor;"
    "    outTexCoord = inTexCoord;"
    "}";

const char glsl_frag[] = 
    "layout(location = 0) in vec4 inColor;"
    "layout(location = 1) in vec2 inTexCoord;"
    
    "layout(location = 0) out vec4 outColor;"
    
    "layout(binding = 1) uniform sampler2D texSampler;"
    
    "void main() {"
    "    outColor = vec4(texture(texSampler, inTexCoord).rgb * inColor.rgb, 1.0);"
    "}";

namespace app {
    static bool running{};
    static double time{};
    static uint64_t ticks{};
    
    // uniform data
    static UniformBufferObject ubo;
    
    // mesh data
    static std::vector<Vertex> vertices;
    static std::vector<uint16_t> indices;
    
    // resource paths
    static const char* objPath = "mesh/cube/cube.obj";
    static const char* imgPath = "mesh/cube/cube.png";
};

namespace vk {
    static GLFWwindow* window{};                                    // handle to the window in the window manager (desktop)
    static VkInstance instance{};                                   // handle to the vulkan instance (mother of it all)
    static VkPhysicalDevice physicalDevice{};                       // handle to physical device we want to use
    static VkPhysicalDeviceProperties physicalDeviceProperties{};   // properties of the physical device we've chosen for rendering
    static VkDevice device{};                                       // handle to the logical device that does anything (GPU components we want to use)
    static VkSurfaceKHR surface{};                                  // the visible pixel data in the window, essentially
    
    // according to vulkan-tutorial.com, queue families are divided by their capabilities.
    // some can't do Graphics (making pixels), some can't do Presentation (putting pixels on screen), and some are seemingly identical, but differentiated by less-obvious characteristics in the hardware.
    // each Queue Family has a number of Queues, which do Work. To instruct them to work, we create Command Pools that contain Command Buffers, that contain instructions like draw commands.
    // for graphics, we need a Framebuffer (with its Images), Render Pass, and Pipeline. These all instruct a GPU _how_ to complete work.
    // Lastly, we need a Swap Chain that is used during Graphics and Presentation. The Swap Chain just decides which Images we're generating with Graphics and which we're displaying with Presentation at any moment.
    // With all of these objects in place, we can submit Queues to Work, and get stuff on the screen.
    
    static QueueFamilies families{};                                // work horses made available by our device (GPU) each queue families owns queues -> command pools -> command buffers that do work
    static std::vector<CommandPool> commandPools{};                 // command pools own common command buffers, so they don't need to be individually deleted when it's time to clean a command pool.
    
    // Every command pool corresponds to one queue family on creation.
    // In my app the index of a family associates directly to its command pool in the above vector.
    
    // we can generate more than one image at a time by writing to different images in thes wap chain.
    // semaphores and fences enable us to signal the GPU/CPU when frames are available on the swapchain.
    constexpr int maxImagesInFlight = 2;
    
    static VkSwapchainKHR swapChain{};                              // the mechanism by which images get swapped and displayed on the surface
    static VkFormat swapChainFormat{};                              // pixel format of the images in the swap chain
    static VkExtent2D swapChainExtent{};                            // dimensions of the images in the chain (matches the surface size essentially)
    static std::vector<VkImage> swapChainImages{};                  // pixel data for every image in the swap chain
    static std::vector<VkImageView> swapChainImageViews{};          // interface to the swap chain images which framebuffers need
    
    // in an app without multisampling, depth buffering, or stencil operations,
    // you can actually perform all rendering directly on the swap chain images themselves.
    // if you want any of the above features however, you must create additional buffers that will act as attachments during rendering:
    
    static std::vector<Image> colorBuffers{};                       // every image in the swapchain gets a color buffer
    static std::vector<Image> depthBuffers{};                       // every image in the swapchain gets a depth buffer
    
    // framebuffers are very simple objects. they are the output of drawing commands.
    // all they do is define the relationship between a renderpass and one or more image views.
    // if you want a renderpass to output to more than one image, configuring a framebuffer
    // to have more than one attachment is the way to do it.
    static std::vector<VkFramebuffer> swapChainFramebuffers{};
    
    static VkDescriptorSetLayout descriptorSetLayout{};             // defines uniform bindings between shader and pipeline layout
    static VkPipelineLayout pipelineLayout{};                       // binds descriptor set layout to pipeline
    static VkPipeline pipeline{};                                   // a structure that defines the pipeline for rendering, eg: assembly, vertex shading, tesselation and geometry, fragmentation, rasterization, all the way to drawn pixels
    static VkRenderPass renderPass{};                               // a structure that defines the framebuffer attachment points for a shader
    
    static std::vector<VkSemaphore> imageAvailableSemaphores{};     // image ready to present
    static std::vector<VkSemaphore> renderFinishedSemaphores{};     // image presented to swap chain
    static std::vector<VkFence> inFlightFences{};                   // waiting for an image from GPU
    
    static uint32_t currentFrame = 0;
    
    // if true, the whole swapchain needs to be recreated.
    // this happens when the window is resized, for instance.
    // recreating the swapchain (and its images) necessitates recreating the
    // renderpass, pipeline, and framebuffer as well.
    static bool windowResized = false;
    
    // Additional topics
    // * Vertex buffers, which are lists of geometry/mesh data (3D models).
    //     * Loading models from disk happens on the CPU, the vertex buffer is then created on the GPU to store the mesh data
    //     * Complete models typically contain a tree of submeshes, meaning multiple draw calls often need to be issued to render a single object. 
    // * Uniform buffers, which contain variables the CPU can push (things like transforms for actor positions and so forth), that are read by the GPU when executing shaders on a pipeline.
    // * Texture mapping, where Images are uploaded directly to the GPU, then a Sampler can be used to access an ImageView in a shader in the midst of a Pipeline
    //     * Generating mipmaps happens after the original image gets uploaded to the GPU
    // * Depth and stencil buffering require extra images and attachments to the shaders executing in the pipeline, they are setup when a pipeline is created.
    // * Multisampling requires a device that supports it and a render pass + pipeline that implements it
    // * Compute shaders, ray-tracing, video encoding/decoding, presentation, transferring, etc.
    
    // test data
    static Mesh mesh{};
    static Texture texture{};
    
    static std::vector<UniformBuffer> uniformBuffers{}; // uniform buffers; one buffer is needed for each frame in flight
    static DescriptorPool descriptorPool{}; // descriptor pools contain sets. sets declare uniform bindings in a pipeline
};

static bool getTransferQueue(VkQueue& queue, VkCommandBuffer& commandBuffer) {
    // for now: pick the first capable queue to work with.
    int familyIndex = 0;
    queue = VK_NULL_HANDLE;
    for (auto& family : vk::families) {
        if (family.hasTransfer && family.queues > 0) {
            vkGetDeviceQueue(vk::device, family.index, 0, &queue);
            break;
        }
        ++familyIndex;
    }
    if (!queue) {
        printlog("no suitable queue for transferring");
        return false;
    }
    
    // pick the command buffer associated with the command pool / queue family
    assert(familyIndex < vk::commandPools.size());
    commandBuffer = vk::commandPools[familyIndex].buffers[vk::currentFrame];
    return true;
}

static bool getGraphicsQueue(VkQueue& queue, VkCommandBuffer& commandBuffer) {
    // for now: pick the first capable queue to work with.
    int familyIndex = 0;
    queue = VK_NULL_HANDLE;
    for (auto& family : vk::families) {
        if (family.hasGraphics && family.queues > 0) {
            vkGetDeviceQueue(vk::device, family.index, 0, &queue);
            break;
        }
        ++familyIndex;
    }
    if (!queue) {
        printlog("no suitable queue for transferring");
        return false;
    }
    
    // pick the command buffer associated with the command pool / queue family
    assert(familyIndex < vk::commandPools.size());
    commandBuffer = vk::commandPools[familyIndex].buffers[vk::currentFrame];
    return true;
}

static bool getPresentQueue(VkQueue& queue, VkCommandBuffer& commandBuffer) {
    // for now: pick the first capable queue to work with.
    int familyIndex = 0;
    queue = VK_NULL_HANDLE;
    for (auto& family : vk::families) {
        if (family.hasPresent && family.queues > 0) {
            vkGetDeviceQueue(vk::device, family.index, 0, &queue);
            break;
        }
        ++familyIndex;
    }
    if (!queue) {
        printlog("no suitable queue for transferring");
        return false;
    }
    
    // pick the command buffer associated with the command pool / queue family
    assert(familyIndex < vk::commandPools.size());
    commandBuffer = vk::commandPools[familyIndex].buffers[vk::currentFrame];
    return true;
}

static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    printlog("failed to find supported format!");
    return VK_FORMAT_UNDEFINED;
}

static VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
    return findSupportedFormat(
        physicalDevice,
        {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

/*static bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}*/

static QueueFamilies findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilies result;
    
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);

    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());
    
    for (int c = 0; c < properties.size(); ++c) {
        const auto& property = properties[c];
        
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, c, surface, &presentSupport);
        
        QueueFamilyInfo family;
        family.index = c;
        family.queues = property.queueCount;
        family.hasGraphics = property.queueFlags & VK_QUEUE_GRAPHICS_BIT;
        family.hasPresent = presentSupport;
        family.hasTransfer = property.queueFlags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
        result.push_back(family);
    }
    
    return result;
}

VkSampleCountFlagBits findMaxUsableSampleCount(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    
    // get capabilities
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    // get formats
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    details.formats.resize(formatCount);
    if (formatCount > 0) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }
    
    // get present modes
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    details.presentModes.resize(presentModeCount);
    if (presentModeCount > 0) {
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats, VkSurfaceFormatKHR preferredFormat) {
    for (auto& fmt : availableFormats) {
        if (fmt.format == preferredFormat.format && fmt.colorSpace == preferredFormat.colorSpace) {
            return fmt;
        }
    }
    return availableFormats[0]; // fail case, just choose the first one available
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availableModes, VkPresentModeKHR preferredMode) {
    for (auto& mode : availableModes) {
        if (mode == preferredMode) {
            return preferredMode;
        }
    }
    
    // according to vulkan-tutorials.com, this is guaranteed to be available
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = {
            (uint32_t)width,
            (uint32_t)height
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

static int deviceSuitability(VkPhysicalDevice device, VkSurfaceKHR surface) {
    int score = 0;
    
    auto families = findQueueFamilies(device, surface);
    
    // check that the device has both graphics and present capabilities
    bool hasPresent = false;
    bool hasGraphics = false;
    for (auto& family : families) {
        if (family.hasPresent) {
            hasPresent = true;
        }
        if (family.hasGraphics) {
            hasGraphics = true;
        }
        if (hasGraphics && hasPresent) {
            break;
        }
    }
    if (!hasGraphics || !hasPresent) {
        return 0;
    }
    
    // search for required extensions
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
    std::set<std::string> extensions;
    for (auto& ext : availableExtensions) {
        extensions.insert(ext.extensionName);
    }
    std::vector<std::string> requiredExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, // swap-chain extension
    };
    for (auto& ext : requiredExtensions) {
        if (!extensions.contains(ext)) {
            return 0;
        }
    }
    
    // check whether the device can support an adequate swap chain
    const SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
    const bool swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    if (!swapChainAdequate) {
        return 0;
    }
    
    // count the features of the device: every feature is worth 1 point
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);
    for (auto* ptr = &features.robustBufferAccess;
        ptr != &features.inheritedQueries; ++ptr) {
        if (*ptr) {
            ++score; // every feature is one point
        }
    }
    
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    // the type of the device has a large effect on its suitability
    switch (properties.deviceType) {
    default:
    case VK_PHYSICAL_DEVICE_TYPE_OTHER: break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: score += 100000; break; // discrete GPUs have a significant performance advantage
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 30000; break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: score += 20000; break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: break;
    }

    // maximum possible size of textures affects graphics quality, so include this in score
    score += properties.limits.maxImageDimension2D;

    return score;
}

static void resizeWindowCallback(GLFWwindow* window, int width, int height) {
    vk::windowResized = true;
}

static GLFWwindow* initGlfw() {
    constexpr int xres = 1280;
    constexpr int yres = 720;
    
    // initialize glfw
    if (glfwInit() != GLFW_TRUE) {
        printlog("failed to initialize GLFW");
        return nullptr; // error
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(xres, yres, "Vulkan Test", nullptr, nullptr);
    if (!window) {
        printlog("failed to open window");
        return nullptr; // error
    }
    glfwSetFramebufferSizeCallback(window, resizeWindowCallback);
    
    return window;
}

static VkInstance createVulkanInstance() {
    // initialize app info structure (vulkan)
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;
    
    // require specific extensions
    uint32_t glfwExtensionCount;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    printlog("%u extensions required", glfwExtensionCount);
    for (uint32_t c = 0; c < glfwExtensionCount; ++c) {
        printlog(" * %s", glfwExtensions[c]);
    }
    
    // enumerate supported extensions
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<const char*> extensions(extensionCount);
    std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, supportedExtensions.data());
    printlog("%u extensions supported", extensionCount);
    for (uint32_t c = 0; c < extensionCount; ++c) {
        const auto& ext = supportedExtensions[c];
        printlog(" * %s (%u)", ext.extensionName, ext.specVersion);
        extensions[c] = ext.extensionName;
    }
    
    // create vulkan instance
    VkInstance instance = VK_NULL_HANDLE;
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = (uint32_t)extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = 0; // if >0, validation layers enabled
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    auto result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        printlog("failed to create vulkan instance");
        return VK_NULL_HANDLE; // error
    }
    
    return instance;
}

static VkPhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        printlog("no graphics devices found!");
        return VK_NULL_HANDLE; // error
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    std::multimap<int, VkPhysicalDevice> candidates;
    for (const auto& device : devices) {
        int score = deviceSuitability(device, surface);
        candidates.insert(std::make_pair(score, device));
    }

    // check if the best candidate is suitable at all
    if (candidates.rbegin()->first > 0) {
        return candidates.rbegin()->second;
    } else {
        printlog("no suitable GPU found");
        return VK_NULL_HANDLE; // error
    }
}

static VkDevice createLogicalDevice(QueueFamilies& families, VkPhysicalDevice device, VkSurfaceKHR surface) {
    VkDevice result = VK_NULL_HANDLE;
    
    // create new queues for our logical device
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(families.size());
    for (int index = 0; index < queueCreateInfos.size(); ++index) {
        const auto& queueFamily = families[index];
        
        // the first queue in the family gets top priority, every subsequent queue gets less priority
        std::vector<float> priorities(queueFamily.queues);
        for (auto c = 0; c < queueFamily.queues; ++c) {
            priorities[c] = 1.f - (float)c / queueFamily.queues;
        }
        
        auto& queueCreateInfo = queueCreateInfos[index];
        queueCreateInfo = VkDeviceQueueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily.index;
        queueCreateInfo.queueCount = queueFamily.queues;
        queueCreateInfo.pQueuePriorities = priorities.data();
    }
    
    // logical device features
    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE; // for anisotropic sampling support.
    deviceFeatures.sampleRateShading = VK_TRUE; // for smoother texture sampling.
    
    // logical device extensions
    std::vector<const char*> requiredExtensions{
        VK_KHR_SWAPCHAIN_EXTENSION_NAME, // swap-chain extension
    };
    
    // logical device creation info
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledLayerCount = 0; // device-level validation layers are deprecated in modern Vulkan
    createInfo.ppEnabledLayerNames = nullptr;
    createInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();
    
    // create logical device
    if (vkCreateDevice(device, &createInfo, nullptr, &result) != VK_SUCCESS) {
        printlog("failed to create logical vulkan device");
        return VK_NULL_HANDLE;
    }

    return result;
}

VkSurfaceKHR createVulkanSurface(VkInstance instance, GLFWwindow* window) {
    if (glfwVulkanSupported() != GLFW_TRUE) {
        printlog("vulkan not supported.");
        return VK_NULL_HANDLE;
    }
    VkSurfaceKHR surface;
    auto result = glfwCreateWindowSurface(instance, window, nullptr, &surface); 
    if (result != VK_SUCCESS) {
        printlog("failed to create window surface");
        return VK_NULL_HANDLE;
    }
    return surface;
}

static VkSwapchainKHR createSwapChain(VkSwapchainKHR oldSwapChain, GLFWwindow* window, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface, const QueueFamilies& families) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice, surface);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats,
        {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}); // preferred format
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes,
        VK_PRESENT_MODE_FIFO_RELAXED_KHR); // preferred present mode (relaxed vsync)
    VkExtent2D extent = chooseSwapExtent(window, swapChainSupport.capabilities);
    
    // request one more image than necessary (if possible),
    // hopefully to prevent possible stalls in the swap chain
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }
    
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1; // for stereoscopic vision: layers = 2
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    
    // in the case of images destined for post-processing or other uses,
    // change imageUsage (eg to VK_IMAGE_USAGE_TRANSFER_DST_BIT)
    
    // set image sharing properties for the swapchain
    if (families.size() == 1) {
        // VK_SHARING_MODE_EXCLUSIVE is more performant,
        // but requires explicit transfer of images between queue families.
        // that's a hassle, so only use it if we have one family.
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0;
        createInfo.pQueueFamilyIndices = nullptr;
    } else {
        // if there are multiple families, share the images between all queue families.
        // this is less performant, but also less complex than transferring ownership.
        std::vector<uint32_t> familyIndices(families.size());
        for (int c = 0; c < familyIndices.size(); ++c) {
            familyIndices[c] = families[c].index;
        }
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = (uint32_t)familyIndices.size();
        createInfo.pQueueFamilyIndices = familyIndices.data();
    }
    
    // blending with other windows on the desktop is possible by adjusting this bit
    // could be useful for tooltips, pop-up dialogs or menus
    //createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
    createInfo.presentMode = presentMode;
    
    // note: this causes written pixels to be unreliable if obscured by the rest of the windowing system!!
    // VK_TRUE improves performance at the cost of undefined behavior if reading data back from the
    // window surface for a later stage.
    createInfo.clipped = VK_TRUE;
    
    createInfo.oldSwapchain = oldSwapChain;
    
    VkSwapchainKHR swapChain;
    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        printlog("failed to create swap chain");
        return VK_NULL_HANDLE;
    }
    
    // store some info about the swap chain
    vk::swapChainFormat = surfaceFormat.format;
    vk::swapChainExtent = extent;
    
    return swapChain;
}

static VkImageView createImageView(VkDevice device, VkImage image, uint32_t mipLevels, VkImageViewType type, VkFormat format, VkImageAspectFlags aspectFlags) {
    assert(device);
    assert(image);

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image;
    createInfo.viewType = type;
    createInfo.format = format;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = mipLevels;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    
    VkImageView imageView{};
    if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
        printlog("failed to create vulkan image view");
        return VK_NULL_HANDLE;
    }
    
    return imageView;
}

static bool createSwapChainImageViews(VkDevice device, VkSwapchainKHR swapChain) {
    uint32_t swapChainImageCount;
    vkGetSwapchainImagesKHR(device, swapChain, &swapChainImageCount, nullptr);
    vk::swapChainImages.resize(swapChainImageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &swapChainImageCount, vk::swapChainImages.data());
    vk::swapChainImageViews.resize(swapChainImageCount);
    for (size_t i = 0; i < vk::swapChainImages.size(); i++) {
        vk::swapChainImageViews[i] = createImageView(device, vk::swapChainImages[i], 1,
            VK_IMAGE_VIEW_TYPE_2D, vk::swapChainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        if (!vk::swapChainImageViews[i]) {
            printlog("failed to create image view for swap chain!");
            return false;
        }
    }
    return true;
}

enum ShaderType {
    INVALID,
    VERTEX,
    FRAGMENT,
};

VkShaderModule createShaderModule(VkDevice device, ShaderType type, const char* source) {
    glslang_stage_t glslType;
    switch (type) {
    default:
    case ShaderType::INVALID:
        printlog("invalid GLSL shader type");
        return VK_NULL_HANDLE;
    case ShaderType::VERTEX: glslType = GLSLANG_STAGE_VERTEX; break;
    case ShaderType::FRAGMENT: glslType = GLSLANG_STAGE_FRAGMENT; break;
    }
    auto bin = compileGLSLToSPIRV(glslType, source);
    
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = bin.code.size() * sizeof(uint32_t);
    createInfo.pCode = bin.code.data();
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        printlog("failed to create shader module!");
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

// Creates the layout that binds shader uniforms to a pipeline
static VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        printlog("failed to create descriptor set layout!");
        return VK_NULL_HANDLE;
    }

    return descriptorSetLayout;
}

// Defines a whole graphics pipeline (vertex assembly, vertex shader, tesselation, geometry, fragment shader, etc)
static VkPipeline createGraphicsPipeline(VkDevice device, VkSampleCountFlagBits numSamples, const std::vector<VkDescriptorSetLayout>& layouts, VkExtent2D extent, VkRenderPass renderPass) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = (uint32_t)layouts.size();
    pipelineLayoutInfo.pSetLayouts = layouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &vk::pipelineLayout) != VK_SUCCESS) {
        printlog("failed to create pipeline layout!");
        return VK_NULL_HANDLE;
    }
    
    // set parameters for the input assembler (which is a fixed function)
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    // compile GLSL to SPIRV bytecode format
    VkShaderModule vertShaderModule = createShaderModule(device, ShaderType::VERTEX, glsl_vert);
    VkShaderModule fragShaderModule = createShaderModule(device, ShaderType::FRAGMENT, glsl_frag);
    
    // vertex stage
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    
    // vertex data binding
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    
    // no tesselation stage
    // no geometry stage
    
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)extent.width;
    viewport.height = (float)extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    // set parameters for rasterization (another fixed function)
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE; // clamps depth values to [0.0 - 1.0], requires a device feature to enable
    rasterizer.rasterizerDiscardEnable = VK_FALSE; // if true, all fragments will be discarded
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL; // other modes require device specific features
    rasterizer.lineWidth = 1.0f; // higher values than 1.0 requires a specific device feature
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // automatically cull back faces (no extra features required)
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // determine which wind order defines the front of a polygon
    rasterizer.depthBiasEnable = VK_FALSE; // requires no extra features
    rasterizer.depthBiasConstantFactor = 0.0f; // optional if disabled
    rasterizer.depthBiasClamp = 0.0f; // optional if disabled
    rasterizer.depthBiasSlopeFactor = 0.0f; // optional if disabled
    
    // multisample configuration
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_TRUE; // make sure deviceFeatures.sampleRateShading = VK_TRUE
    multisampling.rasterizationSamples = numSamples;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
    
    // depth + stencil configuration
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = 0.0f; // Optional
    depthStencil.maxDepthBounds = 1.0f; // Optional
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {}; // Optional
    depthStencil.back = {}; // Optional
    
    // color blending configuration
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE; // we are not blending by bits, so this should be disabled
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // optional if disabled
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // optional if disabled
    colorBlending.blendConstants[1] = 0.0f; // optional if disabled
    colorBlending.blendConstants[2] = 0.0f; // optional if disabled
    colorBlending.blendConstants[3] = 0.0f; // optional if disabled
    
    // fragment stage
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    // allow for viewport and scissor settings to be updated dynamically
    // without recreating the entire pipeline
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = (uint32_t)dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();
    
    // pipeline creation properties!
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = vk::pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    
    // if our pipeline is derivative, set the "parent" pipeline here
    // if the "parent" hasn't been created yet, set the index of the new one
    // instead of the handle
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;
    
    VkPipeline pipeline = VK_NULL_HANDLE;
    auto result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        printlog("failed to create graphics pipeline!");
        vkDestroyPipelineLayout(device, vk::pipelineLayout, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        return VK_NULL_HANDLE;
    }
    
    // destroy compiled shader modules, they are not needed any longer
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    
    return pipeline;
}

static VkRenderPass createRenderPass(VkDevice device, VkSampleCountFlagBits numSamples, VkFormat colorFormat, VkFormat depthFormat) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = colorFormat;
    colorAttachment.samples = numSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    // if you were rendering directly to the window surface, you would use this layout instead:
    //colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = depthFormat;
    depthAttachment.samples = numSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    // used to "downsample" a multisampled image to a single-sampled image (ie the window surface)
    VkAttachmentDescription presentAttachment{};
    presentAttachment.format = colorFormat;
    presentAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    presentAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    presentAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    presentAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    presentAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    presentAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    presentAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // we are presenting this image
    
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0; // "layout(location = 0) out vec4 outColor"
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    VkAttachmentReference presentAttachmentRef{};
    presentAttachmentRef.attachment = 2;
    presentAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &presentAttachmentRef;
    
    std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, presentAttachment};
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = (uint32_t)attachments.size();
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    
    // this dependency prevents this render pass from executing until the
    // color attachment in the framebuffer has been freed from its last write.
    // TODO: study what subpasses are and why they really exist.
    // Apparently certain implicit subpasses adjust memory layouts of inputs
    // just to achieve coherency with upcoming operations.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkRenderPass renderPass;
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        printlog("failed to create render pass!");
        return VK_NULL_HANDLE;
    }

    return renderPass;
}

static bool createFramebuffers(
    VkDevice device,
    std::vector<VkFramebuffer>& framebuffers,
    const std::vector<VkImageView>& colorViews,
    const std::vector<VkImageView>& depthViews,
    const std::vector<VkImageView>& swapChainImageViews,
    VkRenderPass renderPass,
    VkExtent2D extent
    ) {
    assert(colorViews.size() == swapChainImageViews.size());
    assert(depthViews.size() == swapChainImageViews.size());
    framebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < framebuffers.size(); i++) {
        std::array<VkImageView, 3> attachments = {
            colorViews[i],
            depthViews[i],
            swapChainImageViews[i],
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = (uint32_t)attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = extent.width;
        framebufferInfo.height = extent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
            printlog("failed to create framebuffer!");
            return false;
        }
    }
    return true;
}

DescriptorPool createDescriptorPool(VkDevice device, VkDescriptorSetLayout layout, const std::vector<UniformBuffer>& uniformBuffers, const Texture& texture) {
    DescriptorPool result{};
    
    const auto numDescriptorSets = (uint32_t)uniformBuffers.size();
    assert(numDescriptorSets != 0);
    
    // first step is to create the descriptor pool:
    
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = numDescriptorSets;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = numDescriptorSets;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = numDescriptorSets;
    
    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        printlog("failed to create descriptor pool!");
        return result;
    }
    
    // then create the descriptor sets:
    
    std::vector<VkDescriptorSetLayout> layouts(numDescriptorSets, layout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = numDescriptorSets;
    allocInfo.pSetLayouts = layouts.data();
    
    result.sets.resize(numDescriptorSets);
    if (vkAllocateDescriptorSets(device, &allocInfo, result.sets.data()) != VK_SUCCESS) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        printlog("failed to allocate descriptor sets!");
        return result;
    }
    
    // now populate the descriptor sets:
    
    for (uint32_t c = 0; c < numDescriptorSets; ++c) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[c].buffer.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);
        
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = texture.image.view;
        imageInfo.sampler = texture.sampler;
        
        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
        
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = result.sets[c];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;
        
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = result.sets[c];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;
        
        vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }
    
    result.pool = descriptorPool;
    return result;
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

static VkDeviceMemory allocateBufferMemory(VkPhysicalDevice physicalDevice, VkDevice device, VkBuffer buffer, VkMemoryPropertyFlags flags) {
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    
    // find correct memory type
    auto memTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, flags);
    if (memTypeIndex == UINT32_MAX) {
        printlog("unable to find memory heap for vulkan buffer");
        return VK_NULL_HANDLE;
    }
    
    // allocation info
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    
    // allocate memory
    // NOTE the number of pages you can allocate with vkAllocateMemory is actually very low!
    // on a GTX 1080 you will run out at 4096 calls. It is better to use a custom allocator eg:
    // https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    VkDeviceMemory deviceMemory;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &deviceMemory) != VK_SUCCESS) {
        printlog("failed to allocate vertex buffer memory!");
        return VK_NULL_HANDLE;
    }
    
    // bind memory to buffer
    vkBindBufferMemory(device, buffer, deviceMemory, 0);
    
    return deviceMemory;
}

static VkBuffer createBuffer(VkDevice device, const QueueFamilies& families, VkDeviceSize bufferSize, VkBufferUsageFlags usage) {
    assert(!families.empty());
    assert(device != VK_NULL_HANDLE);
    assert(bufferSize > 0);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = usage;
    
    // set sharing mode
    if (families.size() == 1) {
        // VK_SHARING_MODE_EXCLUSIVE is more performant,
        // but requires explicit transfer of images between queue families.
        // that's a hassle, so only use it if we have one family.
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.queueFamilyIndexCount = 0;
        bufferInfo.pQueueFamilyIndices = nullptr;
    } else {
        // if there are multiple families, share the images between all queue families.
        // this is less performant, but also less complex than transferring ownership.
        std::vector<uint32_t> familyIndices(families.size());
        for (int c = 0; c < familyIndices.size(); ++c) {
            familyIndices[c] = families[c].index;
        }
        bufferInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = (uint32_t)familyIndices.size();
        bufferInfo.pQueueFamilyIndices = familyIndices.data();
    }

    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        printlog("failed to create vulkan buffer!");
        return VK_NULL_HANDLE;
    }
    
    return buffer;
}

static bool copyBuffer(VkDevice device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer{};
    VkQueue transferQueue{};
    getTransferQueue(transferQueue, commandBuffer);
    
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // if this were a temporary buffer we could use VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT. but its not
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(transferQueue, 1, &submitInfo, VK_NULL_HANDLE);
    
    // stall CPU until queue is finished
    vkQueueWaitIdle(transferQueue);
    
    // clearly stalling is not the only choice here.
    // if no other queue needs access to the vertex buffer, and we have other
    // queues available, we can easily put the other queues to work
    // and only pause further execution when we know we need access to this
    // vertex buffer.
    
    // this can be a powerful mechanism by which we stream data to the GPU
    // without stalling execution of any pipeline.
    
    // note: even waiting for the queue to finish with vkQueueWaitIdle may not be most performant,
    // because a _fence_ allows us to schedule multiple transfers and allow the driver
    // to complete them in the order that is most optimal for it, rather than for my
    // application.
    
    return true;
}

static void transitionImageLayout(Image image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkQueue queue{};
    VkCommandBuffer commandBuffer{};
    getTransferQueue(queue, commandBuffer);
    
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // if this were a temporary buffer we could use VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT. but its not
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    
    // if transferring an image to another queue family, use these:
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = image.mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    // work out stages and access flags (valid combinations are as follows):
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported
    
    VkPipelineStageFlags sourceStage{};
    VkPipelineStageFlags destinationStage{};
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        printlog("unsupported layout transition!");
    }
    
    // vkCmdPipelineBarrier transitions the memory layout of the image.
    // The fourth parameter is either 0 or VK_DEPENDENCY_BY_REGION_BIT.
    // The latter turns the barrier into a per-region condition.
    // That means that the implementation is allowed to already begin
    // reading from the parts of a resource that were written so far, for example.
    vkCmdPipelineBarrier(commandBuffer,
        sourceStage, destinationStage,
        0,              // VK_DEPENDENCY_BY_REGION_BIT
        0, nullptr,     // memory layout transfer
        0, nullptr,     // buffer layout transfer
        1, &barrier);   // image layout transfer
    
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    
    // stall CPU until queue is finished
    // TODO: find a better way to synchronize this than stalling the queue
    vkQueueWaitIdle(queue);
}

static bool generateMipmaps(VkPhysicalDevice physicalDevice, Image image) {
    // check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, image.format, &formatProperties);
    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        printlog("texture image format does not support linear blitting!");
        return false;
    }

    // begin command queue
    VkQueue queue{};
    VkCommandBuffer commandBuffer{};
    getGraphicsQueue(queue, commandBuffer);
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // if this were a temporary buffer we could use VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT. but its not
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    // begin memory layout transfer structure
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image.image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    
    // begin generating mipmaps
    int32_t mipWidth = (int32_t)image.width;
    int32_t mipHeight = (int32_t)image.height;
    for (uint32_t i = 1; i < image.mipLevels; ++i) {
        // transition the previous mip level to a src optimal layout.
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
        
        // blit the previous mip level to the current one, performing scaling in the process.
        VkImageBlit blit{};
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;
        vkCmdBlitImage(commandBuffer,
            image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit,
            VK_FILTER_LINEAR);
    
        // transition the last mip level to a shader read-only layout.
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
        
        // reduce the mipmap size
        if (mipWidth > 1) {
            mipWidth /= 2;
        }
        if (mipHeight > 1) {
            mipHeight /= 2;
        }
    }
    
    // finally, transfer the smallest mipmap to a shader read-only layout
    barrier.subresourceRange.baseMipLevel = image.mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
    
    // submit commands
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    
    // stall CPU until queue is finished
    // TODO: find a better way to synchronize this than stalling the queue
    vkQueueWaitIdle(queue);
    
    return true;
}

static bool copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth) {
    VkQueue queue{};
    VkCommandBuffer commandBuffer{};
    getTransferQueue(queue, commandBuffer);
    
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // if this were a temporary buffer we could use VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT. but its not
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, depth};
    vkCmdCopyBufferToImage(commandBuffer, buffer, image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    
    // stall CPU until queue is finished
    // TODO: find a better way to synchronize this than stalling the queue
    vkQueueWaitIdle(queue);
    
    return true;
}

Image createImage(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    uint32_t width, uint32_t height, uint32_t depth,
    uint32_t mipLevels, VkSampleCountFlagBits numSamples,
    VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties, VkImageAspectFlags aspectFlags
) {
    assert(device);
    assert(physicalDevice);
    assert(width);
    assert(height);
    assert(depth);

    Image result{};
    
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = depth;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // TODO: used by one queue family (for graphics as well as transfers)
    imageInfo.samples = numSamples;
    imageInfo.flags = 0; // Optional
    
    VkImage image;
    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        printlog("failed to create image: image creation failed");
        return result;
    }
    
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    VkDeviceMemory imageMemory;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        vkDestroyImage(device, image, nullptr);
        printlog("failed to create image: image memory allocation failed");
        return result;
    }

    vkBindImageMemory(device, image, imageMemory, 0);
    
    auto imageView = createImageView(device, image, mipLevels,
        depth > 1 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
        format, aspectFlags);
    if (!imageView) {
        vkDestroyImage(device, image, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);
        printlog("failed to create image: image view creation failed");
        return result;
    }
    
    result.image = image;
    result.memory = imageMemory;
    result.view = imageView;
    result.mipLevels = mipLevels;
    result.width = width;
    result.height = height;
    result.depth = depth;
    result.format = format;
    return result;
}

static VkSampler createTextureSampler(VkDevice device, uint32_t mipLevels) {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    samplerInfo.anisotropyEnable = vk::physicalDeviceProperties.limits.maxSamplerAnisotropy > 1.f ? VK_TRUE : VK_FALSE;
    samplerInfo.maxAnisotropy = vk::physicalDeviceProperties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = (float)mipLevels;
    
    VkSampler sampler{};
    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        printlog("failed to create texture sampler!");
        return VK_NULL_HANDLE;
    }
    
    return sampler;
}

static Texture createTexture(VkDevice device, VkPhysicalDevice physicalDevice, const QueueFamilies& families, const char* path) {
    assert(path);

    Texture result{};

    int texWidth, texHeight, texChannels;
    auto pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize bufferSize = texWidth * texHeight * texChannels;
    if (!pixels) {
        printlog("failed to load texture from file '%s'", path);
        return result;
    }
    const auto mipLevels = (uint32_t)std::floor(std::log2(std::max(texWidth, texHeight))) + 1;
    
    // create staging buffer
    auto stagingBuffer = createBuffer(device, families, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!stagingBuffer) {
        stbi_image_free(pixels);
        printlog("failed to create texture: staging buffer creation failed");
        return result;
    }
    
    // allocate staging buffer
    auto stagingBufferMemory = allocateBufferMemory(physicalDevice, device, stagingBuffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!stagingBufferMemory) {
        stbi_image_free(pixels);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        printlog("failed to create texture: staging buffer memory allocation failed");
        return result;
    }

    // map staging buffer (move data from CPU to staging buffer)
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, pixels, (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);
    stbi_image_free(pixels);
    
    auto image = createImage(device, physicalDevice,
        texWidth, texHeight, 1, mipLevels, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    if (!image.image) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        printlog("failed to create texture: image creation failed");
        return result;
    }
    
    // transition image to a layout suitable for copying
    transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        
    // copy staging buffer contents to image
    copyBufferToImage(stagingBuffer, image.image,
        (uint32_t)texWidth, (uint32_t)texHeight, 1);
    
    // free the staging buffer and its memory.
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
        
    // generate mipmaps (also transitions all mip-level layouts to ones suitable for sampling)
    if (!generateMipmaps(physicalDevice, image)) {
        printlog("failed to create texture: mipmap generation failed");
        image.destroy(device);
        return result;
    }
    
    // if we were *not* generating mipmaps, we would simply transition
    // the image to a layout suitable for sampling in a shader
    /*transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);*/
    
    auto sampler = createTextureSampler(device, image.mipLevels);
    if (!sampler) {
        printlog("failed to create texture: sampler creation failed");
        image.destroy(device);
        return result;
    }
    
    result.image = image;
    result.sampler = sampler;
    return result;
}

static UniformBuffer createUniformBuffer(VkDevice device, VkPhysicalDevice physicalDevice, const QueueFamilies& families) {
    UniformBuffer result{};
    const auto bufferSize = sizeof(UniformBufferObject);
    
    auto buffer = createBuffer(device, families, bufferSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    if (!buffer) {
        printlog("failed to create uniform buffer: buffer creation failed");
        return result;
    }
    
    auto memory = allocateBufferMemory(physicalDevice, device, buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!memory) {
        vkDestroyBuffer(device, buffer, nullptr);
        printlog("failed to create uniform buffer: memory allocation failed");
        return result;
    }
    
    if (vkMapMemory(device, memory, 0, bufferSize, 0, &result.data) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        printlog("failed to create uniform buffer: failed to map uniform buffer memory");
        return result;
    }
    result.buffer.buffer = buffer;
    result.buffer.memory = memory;
    
    return result;
}

Buffer createVertexBuffer(VkPhysicalDevice physicalDevice, VkDevice device, const QueueFamilies& families, const std::vector<Vertex>& vertices) {
    Buffer result{};
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    // create staging buffer
    auto stagingBuffer = createBuffer(device, families, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!stagingBuffer) {
        printlog("failed to create vertex buffer: staging buffer creation failed");
        return result;
    }
    
    // allocate staging buffer
    auto stagingBufferMemory = allocateBufferMemory(physicalDevice, device, stagingBuffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!stagingBufferMemory) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        printlog("failed to create vertex buffer: staging buffer memory allocation failed");
        return result;
    }

    // map staging buffer (move data from CPU to staging buffer)
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // create vertex buffer
    auto vertexBuffer = createBuffer(device, families, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if (!vertexBuffer) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        printlog("failed to create vertex buffer: vertex buffer creation failed");
        return result;
    }
    
    // allocate vertex buffer memory
    auto vertexBufferMemory = allocateBufferMemory(physicalDevice, device, vertexBuffer,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!vertexBufferMemory) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        printlog("failed to create vertex buffer: vertex buffer memory allocation failed");
        return result;
    }
    
    // copy staging buffer to vertex buffer using GPU commands,
    // then free the staging buffer and its memory.
    copyBuffer(device, stagingBuffer, vertexBuffer, bufferSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    
    result.buffer = vertexBuffer;
    result.memory = vertexBufferMemory;
    return result;
}

Buffer createIndexBuffer(VkPhysicalDevice physicalDevice, VkDevice device, const QueueFamilies& families, const std::vector<uint16_t>& indices) {
    Buffer result{};
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    // create staging buffer
    auto stagingBuffer = createBuffer(device, families, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!stagingBuffer) {
        printlog("failed to create index buffer: staging buffer creation failed");
        return result;
    }
    
    // allocate staging buffer
    auto stagingBufferMemory = allocateBufferMemory(physicalDevice, device, stagingBuffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!stagingBufferMemory) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        printlog("failed to create index buffer: staging buffer memory allocation failed");
        return result;
    }

    // map staging buffer (move data from CPU to staging buffer)
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    // create vertex buffer
    auto indexBuffer = createBuffer(device, families, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    if (!indexBuffer) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        printlog("failed to create index buffer: index buffer creation failed");
        return result;
    }
    
    // allocate vertex buffer memory
    auto indexBufferMemory = allocateBufferMemory(physicalDevice, device, indexBuffer,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!indexBufferMemory) {
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        printlog("failed to create index buffer: index buffer memory allocation failed");
        return result;
    }
    
    // copy staging buffer to vertex buffer using GPU commands,
    // then free the staging buffer and its memory.
    copyBuffer(device, stagingBuffer, indexBuffer, bufferSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
    
    result.buffer = indexBuffer;
    result.memory = indexBufferMemory;
    return result;
}

static bool createCommandBuffers(VkDevice device, CommandPool& commandPool) {
    commandPool.buffers.resize(vk::maxImagesInFlight);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool.handle;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandPool.buffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandPool.buffers.data()) != VK_SUCCESS) {
        printlog("failed to allocate command buffers!");
        return false;
    }
    return true;
}

static VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    
    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        printlog("failed to create command pool!");
        return VK_NULL_HANDLE;
    }
    return commandPool;
}

/*
    All of the command buffers we've used thus far have been primary command buffers,
meaning they can be submitted directly to a Vulkan queue to be executed by the device.
Secondary command buffers are instead executed indirectly by being called from primary
command buffers and may not be submitted to queues.

The usage of secondary command buffers offers two primary advantages:

    1. Secondary command buffers may be allocated and recorded in parallel
    which allows you to better leverage modern hardware with its panoply of CPU cores

    2. The lifetime of secondary command buffers can managed independently of one another
    so you can have a mixture of long-lived or permanent secondary command buffers that
    intermingle with frequently updated secondary command buffers which allows you to
    reduce the number of command buffers you need to create every frame

Both of these points are true for primary command buffers as well, but primary command
buffers have a significant limitation that effectively prevents them from fulfilling
these use cases. Multiple primary command buffers may not be executed within the same
render pass instance meaning that if you wanted to execute multiple primary command
buffers for a frame, each primary command buffer would need to start with
cmd_begin_render_pass and end with cmd_end_render_pass.

This might not sound like a big deal but beginning a render pass instance can be a pretty
heavyweight operation and needing to do this many times per frame can destroy performance
on some hardware. Secondary command buffers avoid this problem by being able to inherit
the render pass instance as well as other state from the primary command buffer it is
called from.

    https://kylemayes.github.io/vulkanalia/dynamic/secondary_command_buffers.html#primary-vs-secondary
*/

static bool recordCommandBuffer(VkCommandBuffer commandBuffer, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extent, VkPipeline pipeline) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // see VkCommandBufferUsageFlags
    
    // for secondary command buffers:
    // allows us to specify what part of the command buffer to inherit
    beginInfo.pInheritanceInfo = nullptr;
    
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        printlog("failed to begin recording command buffer!");
        return false;
    }
    
    // configure the render pass
    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = framebuffer;
    
    // these should match the size of the framebuffer for best performance:
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = extent;
    
    // define clear color for VK_ATTACHMENT_LOAD_OP_CLEAR
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = (uint32_t)clearValues.size();
    renderPassInfo.pClearValues = clearValues.data();
    
    // begin render pass
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    
    // viewport and scissor were declared dynamic on the pipeline previously, so they must be recorded on the command line
    // note: is there any performance advantage to making viewport and scissor static? answer: no, not really.
    
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // bind mesh data
    VkBuffer vertexBuffers[] = {vk::mesh.vertexBuffer.buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, vk::mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
    
    // bind descriptor set for uniforms
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk::pipelineLayout, 0, 1, &vk::descriptorPool.sets[vk::currentFrame], 0, nullptr);
    
    // draw call has 5 parameters (aside from the command buffer)
    // * indexCount: self explanatory
    // * instanceCount: Used for instanced rendering, use 1 if you're not doing that.
    // * firstIndex: Used as an offset into the index buffer.
    // * vertexOffset: the values in the index buffer will be offset by this value.
    // * firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
    vkCmdDrawIndexed(commandBuffer, (uint32_t)app::indices.size(), 1, 0, 0, 0);
    
    // end render pass
    vkCmdEndRenderPass(commandBuffer);
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        printlog("failed to record command buffer!");
        return false;
    }
    return true;
}

static bool createSyncObjects() {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = nullptr; // optional
    semaphoreInfo.flags = 0; // optional
    
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = nullptr; // optional
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // without this, draw() hangs on the first frame
    
    // the above are dummy objects, you really can't customize semaphores and fences on creation.
    
    vk::imageAvailableSemaphores.resize(vk::maxImagesInFlight);
    vk::renderFinishedSemaphores.resize(vk::maxImagesInFlight);
    vk::inFlightFences.resize(vk::maxImagesInFlight);
    
    for (int c = 0; c < vk::maxImagesInFlight; ++c) {
        if (vkCreateSemaphore(vk::device, &semaphoreInfo, nullptr, &vk::imageAvailableSemaphores[c]) != VK_SUCCESS ||
            vkCreateSemaphore(vk::device, &semaphoreInfo, nullptr, &vk::renderFinishedSemaphores[c]) != VK_SUCCESS ||
            vkCreateFence(vk::device, &fenceInfo, nullptr, &vk::inFlightFences[c]) != VK_SUCCESS) {
            printlog("failed to create sync objects!");
            return false;
        }
    }
    return true;
}

static void cleanupSwapChain() {
    if (!vk::swapChainFramebuffers.empty()) {
        for (auto framebuffer : vk::swapChainFramebuffers) {
            vkDestroyFramebuffer(vk::device, framebuffer, nullptr);
        }
        vk::swapChainFramebuffers.clear();
    }
    if (!vk::swapChainImageViews.empty()) {
        for (auto& imageView : vk::swapChainImageViews) {
            vkDestroyImageView(vk::device, imageView, nullptr);
        }
        vk::swapChainImageViews.clear();
    }
    if (vk::swapChain) {
        vkDestroySwapchainKHR(vk::device, vk::swapChain, nullptr);
        vk::swapChain = VK_NULL_HANDLE;
    }
    
    // destroy pipeline and renderpass
    if (vk::pipeline) {
        vkDestroyPipeline(vk::device, vk::pipeline, nullptr);
        vk::pipeline = VK_NULL_HANDLE;
    }
    if (vk::pipelineLayout) {
        vkDestroyPipelineLayout(vk::device, vk::pipelineLayout, nullptr);
        vk::pipelineLayout = VK_NULL_HANDLE;
    }
    if (vk::renderPass) {
        vkDestroyRenderPass(vk::device, vk::renderPass, nullptr);
        vk::renderPass = VK_NULL_HANDLE;
    }
    
    // destroy color buffers
    if (!vk::colorBuffers.empty()) {
        for (auto& buffer : vk::colorBuffers) {
            buffer.destroy(vk::device);
        }
        vk::colorBuffers.clear();
    }
    
    // destroy depth buffers
    if (!vk::depthBuffers.empty()) {
        for (auto& buffer : vk::depthBuffers) {
            buffer.destroy(vk::device);
        }
        vk::depthBuffers.clear();
    }
}

static std::vector<Image> createDepthBuffers(VkDevice device, VkPhysicalDevice physicalDevice, size_t number, VkExtent2D extent) {
    std::vector<Image> result{};
    VkFormat depthFormat = findDepthFormat(physicalDevice);
    if (depthFormat != VK_FORMAT_UNDEFINED) {
        const auto samples = findMaxUsableSampleCount(physicalDevice);
        for (auto c = 0; c < number; ++c) {
            auto image = createImage(device, physicalDevice,
                extent.width, extent.height, 1, 1, samples, depthFormat,
                VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
            result.push_back(image);
        }
    }
    return result;
}

static std::vector<Image> createColorBuffers(VkDevice device, VkPhysicalDevice physicalDevice, size_t number, VkExtent2D extent) {
    std::vector<Image> result{};
    VkFormat colorFormat = vk::swapChainFormat;
    if (colorFormat != VK_FORMAT_UNDEFINED) {
        const auto samples = findMaxUsableSampleCount(physicalDevice);
        for (auto c = 0; c < number; ++c) {
            auto image = createImage(device, physicalDevice,
                extent.width, extent.height, 1, 1, samples, colorFormat,
                VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            result.push_back(image);
        }
    }
    return result;
}

static bool recreateSwapChain() {
    int width, height;
    glfwGetFramebufferSize(vk::window, &width, &height);
    if (width <= 0 || height <= 0) {
        // in this case just early abort, but don't send an error.
        // the point is, we can't create a swapchain of zero size.
        // but we don't want to inform the rest of the app something is wrong.
        // this case happens quite naturally when apps are minimized or scaled to nothing by the user.
        return true;
    }
    vkDeviceWaitIdle(vk::device);
    
    // new swap chain
    auto newSwapChain = createSwapChain(vk::swapChain, vk::window, vk::physicalDevice, vk::device, vk::surface, vk::families);
    cleanupSwapChain();
    vk::swapChain = newSwapChain;
    if (vk::swapChain == VK_NULL_HANDLE) {
        return false;
    }
    if (!createSwapChainImageViews(vk::device, vk::swapChain)) {
        return false;
    }
    
    const auto numSamples = findMaxUsableSampleCount(vk::physicalDevice);
    
    // create color buffers
    vk::colorBuffers = createColorBuffers(vk::device, vk::physicalDevice, vk::swapChainImageViews.size(), vk::swapChainExtent);
    if (vk::colorBuffers.empty()) {
        return false;
    }
    std::vector<VkImageView> colorViews(vk::colorBuffers.size());
    for (auto c = 0; c < vk::colorBuffers.size(); ++c) {
        colorViews[c] = vk::colorBuffers[c].view;
    }
    
    // create depth buffers
    vk::depthBuffers = createDepthBuffers(vk::device, vk::physicalDevice, vk::swapChainImageViews.size(), vk::swapChainExtent);
    if (vk::depthBuffers.empty()) {
        return false;
    }
    std::vector<VkImageView> depthViews(vk::depthBuffers.size());
    for (auto c = 0; c < vk::depthBuffers.size(); ++c) {
        depthViews[c] = vk::depthBuffers[c].view;
    }
    auto depthFormat = findDepthFormat(vk::physicalDevice);
    if (depthFormat == VK_FORMAT_UNDEFINED) {
        return false;
    }
    
    // renderpass and pipeline
    vk::renderPass = createRenderPass(vk::device, numSamples, vk::swapChainFormat, depthFormat);
    if (vk::renderPass == VK_NULL_HANDLE) {
        return false;
    }
    const std::vector<VkDescriptorSetLayout> layouts = {vk::descriptorSetLayout};
    vk::pipeline = createGraphicsPipeline(vk::device, numSamples, layouts, vk::swapChainExtent, vk::renderPass);
    if (vk::pipeline == VK_NULL_HANDLE) {
        return false;
    }
    
    // framebuffer
    const bool framebufferCreationResult = createFramebuffers(
        vk::device,
        vk::swapChainFramebuffers,
        colorViews,
        depthViews,
        vk::swapChainImageViews,
        vk::renderPass,
        vk::swapChainExtent);
    if (!framebufferCreationResult) {
        return false;
    }
    
    return true;
}

static bool loadObj(const char* path, std::vector<Vertex>& out_vertices, std::vector<uint16_t>& out_indices) {
    out_vertices.clear();
    out_indices.clear();
    
    // open file
    FILE* fp = fopen(path, "r");
    if (!fp) {
        return false;
    }
    enum class CoordType : uint32_t {
        Position,
        Texture,
        Normal,
        Color,
        Max
    };
    constexpr auto numCoordTypes = (uint32_t)CoordType::Max;
    
    // count number of elements in file
    char buf[1024];
    std::array<uint32_t, numCoordTypes> num_coords{0};
    std::array<uint32_t, numCoordTypes> num_indices{0};
    while (fgets(buf, sizeof(buf), fp)) {
        if (buf[0] != '#' && buf[0] != '\n') {
            char* type = strtok(buf, " ");
            if (strcmp(type, "v") == 0) {
                ++num_coords[(uint32_t)CoordType::Position];
            }
            else if (strcmp(type, "vt") == 0) {
                ++num_coords[(uint32_t)CoordType::Texture];
            }
            else if (strcmp(type, "vn") == 0) {
                ++num_coords[(uint32_t)CoordType::Normal];
            }
            else if (strcmp(type, "f") == 0) {
                char *token;
                while ((token = strtok(nullptr, " "))) {
                    for (uint32_t c = 0; c < (uint32_t)CoordType::Max; ++c) {
                        num_indices[c] += 3;
                    }
                }
                for (uint32_t c = 0; c < (uint32_t)CoordType::Max; ++c) {
                    num_indices[c] -= 6;
                }
            }
        }
    }
    rewind(fp);
    
    // parse file
    std::vector<glm::vec4> coords[numCoordTypes];
    std::vector<uint16_t> indices[numCoordTypes];
    for (uint32_t c = 0; c < numCoordTypes; ++c) {
        coords[c].resize(num_coords[c]);
        indices[c].resize(num_indices[c]);
    }
    uint32_t indices_count[numCoordTypes] = {0}, coords_count[numCoordTypes] = {0};
    while (fgets(buf, sizeof(buf), fp)) {
        if (buf[0] != '#' && buf[0] != '\n') {
            char* type = strtok(buf, " ");
            if (strcmp(type, "v") == 0) {
                char* c1 = strtok(nullptr, " "); float f1 = strtof(c1, nullptr);
                char* c2 = strtok(nullptr, " "); float f2 = strtof(c2, nullptr);
                char* c3 = strtok(nullptr, " "); float f3 = strtof(c3, nullptr);
                coords[(uint32_t)CoordType::Position][coords_count[(uint32_t)CoordType::Position]++] = glm::vec4{f1, f2, f3, 1.f};
            }
            else if (strcmp(type, "vt") == 0) {
                char* c1 = strtok(nullptr, " "); float f1 = strtof(c1, nullptr);
                char* c2 = strtok(nullptr, " "); float f2 = strtof(c2, nullptr);
                coords[(uint32_t)CoordType::Texture][coords_count[(uint32_t)CoordType::Texture]++] = glm::vec4{f1, 1.f - f2, 0.f, 0.f}; // flip the Y coordinate because its upside-down for some reason?
            }
            else if (strcmp(type, "vn") == 0) {
                char* c1 = strtok(nullptr, " "); float f1 = strtof(c1, nullptr);
                char* c2 = strtok(nullptr, " "); float f2 = strtof(c2, nullptr);
                char* c3 = strtok(nullptr, " "); float f3 = strtof(c3, nullptr);
                coords[(uint32_t)CoordType::Normal][coords_count[(uint32_t)CoordType::Normal]++] = glm::vec4{fabs(f1), fabs(f2), fabs(f3), 1.f}; // these will secretly be used for COLOR so let's just hack this
            }
            else if (strcmp(type, "f") == 0) {
                int index = 0;
                uint32_t first_index[numCoordTypes];
                uint32_t last_index[numCoordTypes];
                for (int c = 0; c < numCoordTypes; ++c) {
                    first_index[c] = indices_count[c];
                }
                char *token;
                while ((token = strtok(nullptr, " "))) {
                    if (++index > 3) {
                        for (uint32_t c = 0; c < numCoordTypes; ++c) {
                            indices[c][indices_count[c]++] = indices[c][first_index[c]];
                            indices[c][indices_count[c]++] = indices[c][last_index[c]];
                        }
                    }
                    char* slice = token;
                    for (int c = 0; c < numCoordTypes; ++c) {
                        uint32_t index = (uint32_t)strtol(slice, nullptr, 10) - 1;
                        last_index[c] = indices_count[c];
                        indices[c][indices_count[c]++] = index;
                        if ((slice = strchr(slice, '/'))) {
                            ++slice;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    fclose(fp);
    
    // fill vertex data cache (note: duplicates vertices)
    const auto size = (uint16_t)indices[0].size();
    out_indices.reserve(size);
    out_vertices.reserve(size);
    for (uint16_t c = 0; c < size; ++c) {
        Vertex v{};
        v.position = coords[(uint16_t)CoordType::Position][indices[(uint16_t)CoordType::Position][c]];
        v.texture = coords[(uint16_t)CoordType::Texture][indices[(uint16_t)CoordType::Texture][c]];
        v.color = coords[(uint16_t)CoordType::Normal][indices[(uint16_t)CoordType::Normal][c]];
        out_indices.push_back(c);
        out_vertices.push_back(v);
    }
    
    // success
    return true;
}

static bool initVulkan(GLFWwindow* window) {
    // initialize vulkan instance
    vk::instance = createVulkanInstance();
    if (vk::instance == VK_NULL_HANDLE) {
        return false;
    }
    
    // get surface from window
    vk::surface = createVulkanSurface(vk::instance, window);
    if (vk::surface == nullptr) {
        return false;
    }
    
    // create physical rendering device and create logical device
    vk::physicalDevice = pickPhysicalDevice(vk::instance, vk::surface);
    if (vk::physicalDevice == VK_NULL_HANDLE) {
        return false;
    }
    vkGetPhysicalDeviceProperties(vk::physicalDevice, &vk::physicalDeviceProperties);
    vk::families = findQueueFamilies(vk::physicalDevice, vk::surface);
    printlog("%u queue families found:", (uint32_t)vk::families.size());
    for (auto& family : vk::families) {
        if (family.hasGraphics && family.hasPresent) {
            printlog(" (%u) queues = %u, %s, %s",
                family.index, family.queues,
                "has graphics", "has present");
        }
        else if (family.hasGraphics || family.hasPresent) {
            printlog(" (%u) queues = %u, %s", family.index, family.queues,
                family.hasGraphics ? "has graphics" : "has present");
        }
        else {
            printlog(" (%u) queues = %u", family.index, family.queues);
        }
    }
    vk::device = createLogicalDevice(vk::families, vk::physicalDevice, vk::surface);
    if (vk::device == VK_NULL_HANDLE) {
        return false;
    }
    
    // create uniform buffers (one needed for each frame in flight)
    vk::uniformBuffers.resize(vk::maxImagesInFlight);
    for (auto& buffer : vk::uniformBuffers) {
        buffer = createUniformBuffer(vk::device, vk::physicalDevice, vk::families);
        if (buffer.data == nullptr) {
            return false;
        }
    }
    
    // create command pools/buffers
    for (int index = 0; index < vk::families.size(); ++index) {
        auto commandPoolHandle = createCommandPool(vk::device, index);
        if (commandPoolHandle) {
            CommandPool commandPool{commandPoolHandle};
            if (createCommandBuffers(vk::device, commandPool)) {
                vk::commandPools.push_back(commandPool);
            } else {
                return false; // something went wrong
            }
        } else {
            return false; // something went wrong
        }
    }
    
    // load texture
    vk::texture = createTexture(vk::device, vk::physicalDevice, vk::families, app::imgPath);
    if (vk::texture.sampler == VK_NULL_HANDLE) {
        return false;
    }
    
    // create descriptor sets (used to bind shader uniforms to pipeline)
    vk::descriptorSetLayout = createDescriptorSetLayout(vk::device);
    if (vk::descriptorSetLayout == VK_NULL_HANDLE) {
        return false;
    }
    vk::descriptorPool = createDescriptorPool(vk::device, vk::descriptorSetLayout, vk::uniformBuffers, vk::texture);
    if (vk::descriptorPool.pool == VK_NULL_HANDLE) {
        return false;
    }
    
    // create swap chain, image views, render pass, pipeline, and framebuffer
    if (!recreateSwapChain()) {
        return false;
    }
    
    // load mesh obj
    if (!loadObj(app::objPath, app::vertices, app::indices)) {
        return false;
    }
    
    // create mesh buffers
    vk::mesh.vertexBuffer = createVertexBuffer(vk::physicalDevice, vk::device, vk::families, app::vertices);
    if (vk::mesh.vertexBuffer.buffer == VK_NULL_HANDLE) {
        return false;
    }
    vk::mesh.indexBuffer = createIndexBuffer(vk::physicalDevice, vk::device, vk::families, app::indices);
    if (vk::mesh.indexBuffer.buffer == VK_NULL_HANDLE) {
        return false;
    }
    
    // create semaphores and fences
    if (!createSyncObjects()) {
        return false;
    }
    
    // just make sure any necessary buffer copy commands have finished.
    vkDeviceWaitIdle(vk::device);
    
    return true;
}

static bool init() {
#ifdef __APPLE__
    chdir("../Resources");
#endif

    printlog("hello");
    app::ticks = 0;
    
    (void)glslang_initialize_process();
    
    if ((vk::window = initGlfw()) == nullptr) {
        return false;
    }
    
    if (!initVulkan(vk::window)) {
        return false;
    }
    
    // success
    return true;
}

static void cleanupVulkan() {
    if (vk::device) {
        vkDeviceWaitIdle(vk::device);
    }
    cleanupSwapChain();
    
    // destroy uniform buffers
    if (!vk::uniformBuffers.empty()) {
        for (auto& buffer : vk::uniformBuffers) {
            buffer.destroy(vk::device);
        }
        vk::uniformBuffers.clear();
    }
    
    // destroy descriptor sets
    if (vk::descriptorPool.pool != VK_NULL_HANDLE) {
        vk::descriptorPool.destroy(vk::device);
    }
    if (vk::descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vk::device, vk::descriptorSetLayout, nullptr);
        vk::descriptorSetLayout = VK_NULL_HANDLE;
    }
    
    // destroy texture
    vk::texture.destroy(vk::device);
    
    // destroy mesh
    vk::mesh.destroy(vk::device);
    
    // destroy semaphores and fences
    for (int c = 0; c < vk::maxImagesInFlight; ++c) {
        if (c < vk::imageAvailableSemaphores.size() && vk::imageAvailableSemaphores[c]) {
            vkDestroySemaphore(vk::device, vk::imageAvailableSemaphores[c], nullptr);
            vk::imageAvailableSemaphores[c] = VK_NULL_HANDLE;
        }
        if (c < vk::renderFinishedSemaphores.size() && vk::renderFinishedSemaphores[c]) {
            vkDestroySemaphore(vk::device, vk::renderFinishedSemaphores[c], nullptr);
            vk::renderFinishedSemaphores[c] = VK_NULL_HANDLE;
        }
        if (c < vk::inFlightFences.size() && vk::inFlightFences[c]) {
            vkDestroyFence(vk::device, vk::inFlightFences[c], nullptr);
            vk::inFlightFences[c] = VK_NULL_HANDLE;
        }
    }
    
    // destroy command pools and buffers
    if (!vk::commandPools.empty()) {
        for (auto& commandPool : vk::commandPools) {
            vkDestroyCommandPool(vk::device, commandPool.handle, nullptr);
        }
        vk::commandPools.clear();
    }
    
    // destroy logical device and surface
    if (vk::device) {
        vkDestroyDevice(vk::device, nullptr);
        vk::device = VK_NULL_HANDLE;
    }
    vk::physicalDevice = VK_NULL_HANDLE; // this was only chosen, so it should not be deleted
    if (vk::surface) {
        vkDestroySurfaceKHR(vk::instance, vk::surface, nullptr);
        vk::surface = VK_NULL_HANDLE;
    }
    
    // destroy vulkan instance
    if (vk::instance) {
        vkDestroyInstance(vk::instance, nullptr);
        vk::instance = VK_NULL_HANDLE;
    }
}

static int term() {
    app::running = false;
    glslang_finalize_process();
    cleanupVulkan();
    if (vk::window) {
        glfwDestroyWindow(vk::window);
        vk::window = nullptr;
    }
    glfwTerminate();
    printlog("goodbye (success)");
    return 0;
}

static void events() {
    glfwPollEvents();
    if (glfwWindowShouldClose(vk::window)) {
        app::running = false;
    }
}

constexpr int frames_per_second = 60;
constexpr int max_frames_before_drop = 8;
static int timer() {
    static auto last = std::chrono::high_resolution_clock::now();
    static double diff = 0.0;
    constexpr auto frame = (double)1.0 / frames_per_second;
    const auto now = std::chrono::high_resolution_clock::now();
    diff += std::chrono::duration<double, std::chrono::seconds::period>(now - last).count();
    int result = 0;
    while (diff >= frame) {
        diff -= frame;
        if (result < max_frames_before_drop) {
            ++result;
        }
    }
    last = std::chrono::high_resolution_clock::now();
    return result;
}

static void update(float seconds) {
    // advance simulation time
    app::time += seconds;

    constexpr float distance = 4.f;
    app::ubo.model = glm::rotate(glm::mat4(1.f), (float)app::time * glm::radians(90.f), glm::vec3(0.f, 1.f, 0.f));
    app::ubo.view = glm::lookAt(glm::vec3(distance, distance, distance), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
    app::ubo.proj = glm::perspective(glm::radians(45.f), vk::swapChainExtent.width / (float) vk::swapChainExtent.height, .1f, 1000.f);
    
    // in GL +Y = Up but in Vulkan +Y = Down.
    // This flips the projection matrix so the image appears rightside up.
    app::ubo.proj[0][0] *= (double)-1.0;
    app::ubo.proj[1][1] *= (double)-1.0;

    // advance simulation tick counter
    ++app::ticks;
}

static int draw() {
    vkWaitForFences(vk::device, 1, &vk::inFlightFences[vk::currentFrame], VK_TRUE, UINT64_MAX);
    
    VkCommandBuffer commandBuffer{};
    VkQueue graphicsQueue{};
    getGraphicsQueue(graphicsQueue, commandBuffer);
    
    uint32_t imageIndex;
    auto acquireResult = vkAcquireNextImageKHR(vk::device, vk::swapChain, UINT64_MAX, vk::imageAvailableSemaphores[vk::currentFrame], VK_NULL_HANDLE, &imageIndex);
    switch (acquireResult) {
    case VK_SUCCESS: break;
    case VK_SUBOPTIMAL_KHR: break;
    case VK_ERROR_OUT_OF_DATE_KHR: return recreateSwapChain() ? -2 : -1; // -2 = early abort, -1 = disaster
    default: printlog("failed to acquire swapchain image!"); return -1;
    }
    
    vkResetCommandBuffer(commandBuffer, 0);
    recordCommandBuffer(commandBuffer, vk::renderPass, vk::swapChainFramebuffers[imageIndex], vk::swapChainExtent, vk::pipeline);
    
    // update mapped uniform buffer with simulation data
    memcpy(vk::uniformBuffers[vk::currentFrame].data, &app::ubo, sizeof(app::ubo));
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // semaphore to wait on until queue can start.
    VkSemaphore waitSemaphores[] = {vk::imageAvailableSemaphores[vk::currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    // semaphore to signal when the queue has finished.
    VkSemaphore signalSemaphores[] = {vk::renderFinishedSemaphores[vk::currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    
    vkResetFences(vk::device, 1, &vk::inFlightFences[vk::currentFrame]);
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, vk::inFlightFences[vk::currentFrame]) != VK_SUCCESS) {
        printlog("failed to submit command buffer to graphics queue!");
        return -1;
    }
    
    return (int)imageIndex;
}

static bool swap(uint32_t imageIndex) {
    VkCommandBuffer commandBuffer{};
    VkQueue presentQueue{};
    getPresentQueue(presentQueue, commandBuffer);
    
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    VkSemaphore presentWaitSemaphores[] = {vk::renderFinishedSemaphores[vk::currentFrame]};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = presentWaitSemaphores;
    VkSwapchainKHR swapChains[] = {vk::swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr; // if there are multiple swap chains, this array holds the results of each one.
    
    auto presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
    switch (presentResult) {
    case VK_SUCCESS: break;
    case VK_SUBOPTIMAL_KHR: return recreateSwapChain();
    case VK_ERROR_OUT_OF_DATE_KHR: return recreateSwapChain();
    default: printlog("failed to submit command buffer to present queue!"); return false;
    }
    
    if (vk::windowResized) {
        recreateSwapChain();
    }
    
    vk::currentFrame = (vk::currentFrame + 1) % vk::maxImagesInFlight;
    
    return true;
}

// lifted from https://blat-blatnik.github.io/computerBear/making-accurate-sleep-function/
void preciseSleep(double seconds) {
    using namespace std;
    using namespace chrono;

    static double estimate = 5e-3;
    static double mean = 5e-3;
    static double m2 = 0;
    static int64_t count = 1;

    while (seconds > estimate) {
        auto start = high_resolution_clock::now();
        this_thread::sleep_for(milliseconds(1));
        auto end = high_resolution_clock::now();

        double observed = (end - start).count() / 1e9;
        seconds -= observed;

        ++count;
        double delta = observed - mean;
        mean += delta / count;
        m2   += delta * (observed - mean);
        double stddev = sqrt(m2 / (count - 1));
        estimate = mean + stddev;
    }

    // spin lock
    auto start = high_resolution_clock::now();
    while ((high_resolution_clock::now() - start).count() / 1e9 < seconds);
}

int main(int argc, const char* argv[]) {
    app::running = init();
    
    // mark start time
    const auto appStart = std::chrono::high_resolution_clock::now();
    while (app::running) {
        const auto frameStart = std::chrono::high_resolution_clock::now();
        
        // check app events (keyboard, window)
        events();
        
        // update simulation
        auto timer_update = timer();
        for (int c = 0; c < timer_update; ++c) {
            update(1.f / frames_per_second);
        }
        
        // draw
        const int drawImageIndex = draw();
        
        // swap
        if (drawImageIndex >= 0) {
            if (!swap(drawImageIndex)) {
                printlog("present failed, aborting");
                app::running = false; // failed to present
            }
        } else if (drawImageIndex == -1) {
            printlog("draw failed, aborting");
            app::running = false; // failed to draw
        }
        
        // sleep until next frame
        const auto frameEnd = std::chrono::high_resolution_clock::now();
        const auto frameTime = std::chrono::duration<double, std::chrono::seconds::period>(frameEnd - frameStart).count();
        constexpr auto frameDuration = (double)1.0 / frames_per_second;
        if (frameTime < frameDuration) {
            preciseSleep(frameDuration - frameTime);
        }
    }
    
    // count frames per second
    const auto appEnd = std::chrono::high_resolution_clock::now();
    const int frames = (int)app::ticks;
    const auto seconds = std::chrono::duration<double, std::chrono::seconds::period>(appEnd - appStart).count();
    const double fps = frames / seconds;
    printlog("%.2f frames per second (%d frames / %.2f seconds)", fps, frames, seconds);
    
    return term();
}
