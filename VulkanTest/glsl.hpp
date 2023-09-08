//
//  glsl.hpp
//  VulkanTest
//
//  Created by Sheridan on 2023-08-26.
//

#pragma once

#include <glslang/Include/glslang_c_shader_types.h>
#include <vector>
#include <cstdint>

struct SpirVBinary {
    std::vector<uint32_t> code;
};

SpirVBinary compileGLSLToSPIRV(glslang_stage_t stage, const char* shaderSource);
