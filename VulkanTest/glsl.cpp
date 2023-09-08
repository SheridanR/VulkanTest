//
//  glsl.cpp
//  VulkanTest
//
//  Created by Sheridan on 2023-08-26.
//

#include "main.hpp"
#include "glsl.hpp"

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

SpirVBinary compileGLSLToSPIRV(glslang_stage_t stage, const char* shaderSource) {
    const glslang_input_t input = {
        .language = GLSLANG_SOURCE_GLSL,
        .stage = stage,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_3, // vulkan 1.3
        .target_language = GLSLANG_TARGET_SPV,
        .target_language_version = GLSLANG_TARGET_SPV_1_6, // spirv version 1.6
        .code = shaderSource,
        .default_version = 460,
        .default_profile = GLSLANG_CORE_PROFILE,
        .force_default_version_and_profile = true,
        .forward_compatible = false,
        .messages = GLSLANG_MSG_DEFAULT_BIT,
        .resource = glslang_default_resource(),
    };

    glslang_shader_t* shader = glslang_shader_create(&input);

    SpirVBinary bin;
    
    if (!glslang_shader_preprocess(shader, &input))	{
        printlog("GLSL preprocessing failed:\n%s\n%s\n%s",
            glslang_shader_get_info_log(shader),
            glslang_shader_get_info_debug_log(shader),
            input.code);
        glslang_shader_delete(shader);
        return bin;
    }

    if (!glslang_shader_parse(shader, &input)) {
        printlog("GLSL parsing failed:\n%s\n%s\n%s",
            glslang_shader_get_info_log(shader),
            glslang_shader_get_info_debug_log(shader),
            glslang_shader_get_preprocessed_code(shader));
        glslang_shader_delete(shader);
        return bin;
    }

    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader(program, shader);

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT)) {
        printlog("GLSL linking failed:\n%s\n%s",
            glslang_program_get_info_log(program),
            glslang_program_get_info_debug_log(program));
        glslang_program_delete(program);
        glslang_shader_delete(shader);
        return bin;
    }

    glslang_program_SPIRV_generate(program, stage);

    bin.code.resize(glslang_program_SPIRV_get_size(program));
    glslang_program_SPIRV_get(program, bin.code.data());

    const char* spirv_messages = glslang_program_SPIRV_get_messages(program);
    if (spirv_messages) {
        printlog("%s", spirv_messages);
    }

    glslang_program_delete(program);
    glslang_shader_delete(shader);

    return bin;
}
