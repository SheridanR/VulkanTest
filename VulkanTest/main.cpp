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
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// C headers
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

// C++ headers
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
};
using QueueFamilies = std::vector<QueueFamilyInfo>;

struct CommandPool {
    VkCommandPool handle;
    std::vector<VkCommandBuffer> buffers;
};
    
struct Vertex {
    glm::vec4 position;
    glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }
    
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, position);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

const char glsl_vert[] = 
    "layout(location = 0) in vec4 inPosition;"
    "layout(location = 1) in vec4 inColor;"
    "layout(location = 0) out vec4 outColor;"
    "void main() {"
    "    gl_Position = inPosition;"
    "    outColor = inColor;"
    "}";

const char glsl_frag[] = 
    "layout(location = 0) in vec4 inColor;"
    "layout(location = 0) out vec4 outColor;"
    "void main() {"
    "    outColor = inColor;"
    "}";

namespace app {
    static bool running = false;
    static uint64_t ticks = 0;
    static const std::vector<Vertex> vertices = {
        {
            {0.f, -0.5f, 0.f, 1.f},
            {1.f,  0.f,  0.f, 1.f},
        },
        {
            {0.5f, 0.5f, 0.f, 1.f},
            {0.f,  1.f,  0.f, 1.f},
        },
        {
            {-0.5f, 0.5f, 0.f, 1.f},
            { 0.f,  0.f,  1.f, 1.f},
        },
    };
};

namespace vk {
    static GLFWwindow* window = nullptr;                            // handle to the window in the window manager (desktop)
    static VkInstance instance = VK_NULL_HANDLE;                    // handle to the vulkan instance (mother of it all)
    static VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;        // handle to physical device we want to use
    static VkDevice device = VK_NULL_HANDLE;                        // handle to the logical device that does anything (GPU components we want to use)
    static VkSurfaceKHR surface = VK_NULL_HANDLE;                   // the visible pixel data in the window, essentially
    static VkSwapchainKHR swapChain = VK_NULL_HANDLE;               // the mechanism by which framebuffers get swapped and displayed on the surface
    static QueueFamilies families{};                                // work horses made available by our device (GPU) each queue families owns queues -> command pools -> command buffers that do work
    static std::vector<VkImage> swapChainImages{};                  // pixel data for every image in the swap chain
    static VkFormat swapChainFormat{};                              // pixel format of the images in the swap chain
    static VkExtent2D swapChainExtent{};                            // dimensions of the images in the chain (matches the surface size essentially)
    static std::vector<VkImageView> swapChainImageViews{};          // interface to the swap chain images which framebuffers need (framebuffers are output of drawing commands)
    static VkRenderPass renderPass = VK_NULL_HANDLE;                // a structure that defines the framebuffer attachment points for a shader
    static VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;        // empty in my project, but defines the uniforms that will be accessed by shaders in the pipeline
    static VkPipeline pipeline = VK_NULL_HANDLE;                    // a structure that defines the pipeline for rendering, eg: assembly, vertex shading, tesselation and geometry, fragmentation, rasterization, all the way to drawn pixels
    static std::vector<VkFramebuffer> swapChainFramebuffers{};      // framebuffers are very simple objects, all they do is define the output between a renderpass and one or more image views. (if you want a renderpass to output to more than one image, configuring a framebuffer to have more than one attachment is the way to do it)
    static std::vector<CommandPool> commandPools{};                 // command pools own common command buffers, so they don't need to be individually deleted when it's time to clean a command pool
    
    constexpr int maxImagesInFlight = 2;
    
    static std::vector<VkSemaphore> imageAvailableSemaphores{};     // image ready to present
    static std::vector<VkSemaphore> renderFinishedSemaphores{};     // image presented to swap chain
    static std::vector<VkFence> inFlightFences{};                   // waiting for an image from GPU
    
    static uint32_t currentFrame = 0;
    static bool framebufferResized = false;
    
    // according to vulkan-tutorial.com, queue families are divided by their capabilities.
    // some can't do Graphics (making pixels), some can't do Presentation (putting pixels on screen), and some are seemingly identical, but differentiated by less-obvious characteristics in the hardware.
    // each Queue Family has a number of Queues, which do Work. To instruct them to work, we create Command Pools that contain Command Buffers, that contain instructions like draw commands.
    // for graphics, we need a Framebuffer (with its Images), Render Pass, and Pipeline. These all instruct a GPU _how_ to complete work.
    // Lastly, we need a Swap Chain that is used during Graphics and Presentation. The Swap Chain just decides which Images we're generating with Graphics and which we're displaying with Presentation at any moment.
    // With all of these objects in place, we can submit Queues to Work, and get stuff on the screen.
    // Not covered:
    // * Vertex buffers, which are lists of geometry/mesh data (3D models).
    //     * Loading models from disk happens on the CPU, the vertex buffer is then created on the GPU to store the mesh data
    //     * Complete models typically contain a tree of submeshes, meaning multiple draw calls often need to be issued to render a single object. 
    // * Uniform buffers, which contain variables the CPU can push (things like transforms for actor positions and so forth), that are read by the GPU when executing shaders on a pipeline.
    // * Texture mapping, where Images are uploaded directly to the GPU, then a Sampler can be used to access an ImageView in a shader in the midst of a Pipeline
    //     * Generating mipmaps happens after the original image gets uploaded to the GPU
    // * Depth and stencil buffering require extra images and attachments to the shaders executing in the pipeline, they are setup when a pipeline is created.
    // * Multisampling requires a device that supports it and a render pass + pipeline that implements it
    // * Compute shaders, ray-tracing, video encoding/decoding, presentation, transferring, etc.
    
    // vertex buffer:
    static VkBuffer vertexBuffer = VK_NULL_HANDLE;
    static VkDeviceMemory vertexBufferMemory;
};

QueueFamilies findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
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
        result.push_back(family);
    }
    
    return result;
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
    vk::framebufferResized = true;
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
    appInfo.apiVersion = VK_API_VERSION_1_0;
    
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

static VkDevice createLogicalVulkanDevice(QueueFamilies& families, VkPhysicalDevice device, VkSurfaceKHR surface) {
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
    
    // set image sharing properties
    for (auto& family : families) {
        if (family.hasGraphics && family.hasPresent) {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        } else {
            // share the images between all queue families.
            // less performant, but also less complex than transferring ownership
            std::vector<uint32_t> indices(families.size());
            for (int c = 0; c < indices.size(); ++c) {
                indices[c] = families[c].index;
            }
            
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = (uint32_t)indices.size();
            createInfo.pQueueFamilyIndices = indices.data();
        }
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

static bool createImageViews(VkDevice device, VkSwapchainKHR swapChain) {
    uint32_t swapChainImageCount;
    vkGetSwapchainImagesKHR(device, swapChain, &swapChainImageCount, nullptr);
    vk::swapChainImages.resize(swapChainImageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &swapChainImageCount, vk::swapChainImages.data());
    vk::swapChainImageViews.resize(swapChainImageCount);
    for (size_t i = 0; i < vk::swapChainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = vk::swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = vk::swapChainFormat;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk::device, &createInfo, nullptr, &vk::swapChainImageViews[i]) != VK_SUCCESS) {
            printlog("failed to create vulkan image view");
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

static VkPipeline createGraphicsPipeline(VkDevice device, VkExtent2D extent, VkRenderPass renderPass) {
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0; // Optional
    pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
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
    
    // multisample configuration (disabled for now)
    // will revisit
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE; // requires a device feature to be enabled
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
    
    // no depth testing for now!
    
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
    pipelineInfo.pDepthStencilState = nullptr;
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
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        return VK_NULL_HANDLE;
    }
    
    // destroy compiled shader modules, they are not needed any longer
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
    
    return pipeline;
}

static VkRenderPass createRenderPass(VkDevice device, VkFormat format) {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0; // "layout(location = 0) out vec4 outColor"
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    
    // this dependency prevents this render pass from executing until the
    // color attachment in the framebuffer has been freed from its last write.
    // TODO: study what subpasses are and why they really exist.
    // They seem to have something to do with simply adjusting memory layout
    // of an image for coherency with upcoming operations.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    
    // new: include the subpass dependency we wrote above
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
    const std::vector<VkImageView>& imageViews,
    VkRenderPass renderPass,
    VkExtent2D extent
    ) {
    framebuffers.resize(imageViews.size());
    for (size_t i = 0; i < framebuffers.size(); i++) {
        VkImageView attachments[] = {
            imageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
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

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkDeviceMemory allocateVertexBufferMemory(VkPhysicalDevice physicalDevice, VkDevice device, VkBuffer vertexBuffer) {
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);
    
    // find correct memory type
    auto memTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memTypeIndex == UINT32_MAX) {
        printlog("unable to find memory heap for vertex buffer");
        return VK_NULL_HANDLE;
    }
    
    // allocation info
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    
    // allocate memory
    VkDeviceMemory vertexBufferMemory;
    if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
        printlog("failed to allocate vertex buffer memory!");
        return VK_NULL_HANDLE;
    }
    
    // bind memory to vertex buffer
    vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    
    return vertexBufferMemory;
}

VkBuffer createVertexBuffer(VkDevice device, size_t bufferSize) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer vertexBuffer;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
        printlog("failed to create vertex buffer!");
        return VK_NULL_HANDLE;
    }
    
    return vertexBuffer;
}

void copyVertexDataToVertexBufferMemory(VkDevice device, VkDeviceMemory memory, const std::vector<Vertex>& vertices) {
    void* data;
    const size_t size = sizeof(vertices[0]) * vertices.size();
    vkMapMemory(device, memory, 0, size, 0, &data);
    memcpy(data, vertices.data(), size);
    vkUnmapMemory(device, memory);
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

static bool recordCommandBuffer(VkCommandBuffer commandBuffer, VkRenderPass renderPass, VkFramebuffer framebuffer, VkExtent2D extent, VkPipeline pipeline) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // see VkCommandBufferUsageFlags
    
    // for secondary command buffers, allows us to specify what part of the command buffer to inherit:
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
    VkClearValue clearColor = {{{0.f, 0.f, 0.f, 0.f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;
    
    // begin render pass
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    
    // viewport and scissor were declared dynamic on the pipeline previously, so they must be recorded on the command line
    
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

    // bind vertex buffer
    VkBuffer vertexBuffers[] = {vk::vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    
    // draw call has 4 parameters (aside from the command buffer)
    // * vertexCount: self explanatory
    // * instanceCount: Used for instanced rendering, use 1 if you're not doing that.
    // * firstVertex: Used as an offset into the vertex buffer, defines the lowest value of gl_VertexIndex.
    // * firstInstance: Used as an offset for instanced rendering, defines the lowest value of gl_InstanceIndex.
    vkCmdDraw(commandBuffer, (uint32_t)app::vertices.size(), 1, 0, 0);
    
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
            printlog("failed to create semaphores!");
            return false;
        }
    }
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
    vk::device = createLogicalVulkanDevice(vk::families, vk::physicalDevice, vk::surface);
    if (vk::device == VK_NULL_HANDLE) {
        return false;
    }
    
    // create swap chain and image views
    vk::swapChain = createSwapChain(vk::swapChain, vk::window, vk::physicalDevice, vk::device, vk::surface, vk::families);
    if (vk::swapChain == VK_NULL_HANDLE) {
        return false;
    }
    if (!createImageViews(vk::device, vk::swapChain)) {
        return false;
    }
    
    // create render pass and pipeline
    vk::renderPass = createRenderPass(vk::device, vk::swapChainFormat);
    if (vk::renderPass == VK_NULL_HANDLE) {
        return false;
    }
    vk::pipeline = createGraphicsPipeline(vk::device, vk::swapChainExtent, vk::renderPass);
    if (vk::pipeline == VK_NULL_HANDLE) {
        return false;
    }
    
    // create framebuffer
    const bool framebufferCreationResult = createFramebuffers(
        vk::device,
        vk::swapChainFramebuffers,
        vk::swapChainImageViews,
        vk::renderPass,
        vk::swapChainExtent);
    if (!framebufferCreationResult) {
        return false;
    }
    
    // create vertex buffer
    vk::vertexBuffer = createVertexBuffer(vk::device, sizeof(app::vertices[0]) * app::vertices.size());
    if (vk::vertexBuffer == VK_NULL_HANDLE) {
        return false;
    }
    vk::vertexBufferMemory = allocateVertexBufferMemory(vk::physicalDevice, vk::device, vk::vertexBuffer);
    if (vk::vertexBufferMemory == VK_NULL_HANDLE) {
        return false;
    }
    copyVertexDataToVertexBufferMemory(vk::device, vk::vertexBufferMemory, app::vertices);
    
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
    
    // create semaphores and fences
    if (!createSyncObjects()) {
        return false;
    }
    
    return true;
}

static bool init() {
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
    auto newSwapChain = createSwapChain(vk::swapChain, vk::window, vk::physicalDevice, vk::device, vk::surface, vk::families);
    cleanupSwapChain();
    vk::swapChain = newSwapChain;
    if (vk::swapChain == VK_NULL_HANDLE) {
        return false;
    }
    if (!createImageViews(vk::device, vk::swapChain)) {
        return false;
    }
    vk::renderPass = createRenderPass(vk::device, vk::swapChainFormat);
    if (vk::renderPass == VK_NULL_HANDLE) {
        return false;
    }
    vk::pipeline = createGraphicsPipeline(vk::device, vk::swapChainExtent, vk::renderPass);
    if (vk::pipeline == VK_NULL_HANDLE) {
        return false;
    }
    const bool framebufferCreationResult = createFramebuffers(
        vk::device,
        vk::swapChainFramebuffers,
        vk::swapChainImageViews,
        vk::renderPass,
        vk::swapChainExtent);
    if (!framebufferCreationResult) {
        return false;
    }
    return true;
}

static void cleanupVulkan() {
    if (vk::device) {
        vkDeviceWaitIdle(vk::device);
    }
    cleanupSwapChain();
    
    // destroy vertex buffer
    if (vk::vertexBuffer) {
        vkDestroyBuffer(vk::device, vk::vertexBuffer, nullptr);
        vk::vertexBuffer = VK_NULL_HANDLE;
    }
    if (vk::vertexBufferMemory) {
        vkFreeMemory(vk::device, vk::vertexBufferMemory, nullptr);
        vk::vertexBufferMemory = VK_NULL_HANDLE;
    }
    
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
    int frames_to_do = 0;
    static double last = 0.0;
    static double diff = 0.0;
    const double frame = (double)CLOCKS_PER_SEC / (double)frames_per_second;
    const double now = clock();
    diff += now - last;
    while (diff >= frame) {
        diff -= frame;
        if (frames_to_do < max_frames_before_drop) {
            ++frames_to_do;
        }
    }
    last = clock();
    return frames_to_do;
}

static void update(float seconds) {
    ++app::ticks;
}

static int draw() {
    vkWaitForFences(vk::device, 1, &vk::inFlightFences[vk::currentFrame], VK_TRUE, UINT64_MAX);
    
    // just pick the first capable graphics queue to draw with.
    int familyIndex = 0;
    VkQueue graphicsQueue{};
    for (auto& family : vk::families) {
        if (family.hasGraphics && family.queues > 0) {
            vkGetDeviceQueue(vk::device, family.index, 0, &graphicsQueue);
            break;
        }
        ++familyIndex;
    }
    if (!graphicsQueue) {
        printlog("no suitable graphics queue for drawing");
        return -1;
    }
    
    // pick the command buffer associated with the command pool / queue family
    assert(familyIndex < vk::commandPools.size());
    auto& commandBuffer = vk::commandPools[familyIndex].buffers[vk::currentFrame];
    
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
    // just pick the first capable present queue
    VkQueue presentQueue{};
    for (auto& family : vk::families) {
        if (family.hasPresent && family.queues > 0) {
            vkGetDeviceQueue(vk::device, family.index, 0, &presentQueue);
            break;
        }
    }
    if (!presentQueue) {
        printlog("no suitable present queue for presentation");
        return false;
    }
    
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
    
    if (vk::framebufferResized) {
        recreateSwapChain();
    }
    
    vk::currentFrame = (vk::currentFrame + 1) % vk::maxImagesInFlight;
    
    return true;
}

int main(int argc, const char* argv[]) {
    app::running = init();
    while (app::running) {
        events();
        auto timer_update = timer();
        for (int c = 0; c < timer_update; ++c) {
            update(1.f / frames_per_second);
        }
        const int drawImageIndex = draw();
        if (drawImageIndex >= 0) {
            if (!swap(drawImageIndex)) {
                printlog("present failed, aborting");
                app::running = false; // failed to present
            }
        } else if (drawImageIndex == -1) {
            printlog("draw failed, aborting");
            app::running = false; // failed to draw
        }
    }
    return term();
}
