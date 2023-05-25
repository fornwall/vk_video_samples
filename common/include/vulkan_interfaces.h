/*
* Copyright 2020 NVIDIA Corporation.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifdef VK_API_USE_DRIVER_REPO
// If using the local driver repo with Vulkan APIs
#include "vulkan/vulkannv.h"
#else
// Using the Vulkan APIs from Vulkan SDK
#ifndef VK_ENABLE_BETA_EXTENSIONS
#define VK_ENABLE_BETA_EXTENSIONS 1
#endif
#include "vulkan/vulkan.h"
#endif

#ifndef VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR
// Very succinct definition of AV1 types. This needs to be converted to using the STD/KHR headers.
#define VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_API_VERSION_0_9_0      VK_MAKE_VIDEO_STD_VERSION(0, 9, 0)
#define VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_SPEC_VERSION           VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_API_VERSION_0_9_0
#define VK_STD_VULKAN_VIDEO_CODEC_AV1_DECODE_EXTENSION_NAME         "VK_STD_vulkan_video_codec_av1_decode"
// Please update these to the correct version/value.
#define VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR         ((VkStructureType)1000509000)
#define VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR         ((VkStructureType)1000509001)
#define VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_KHR ((VkStructureType)1000509002)
#define VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_ADD_INFO_KHR ((VkStructureType)1000509003)
#define VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PICTURE_INFO_KHR         ((VkStructureType)1000509004)
#define VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_DPB_SLOT_INFO_KHR        ((VkStructureType)1000509005)
#define VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR                 ((VkVideoCodecOperationFlagBitsKHR)0x00000004)

#define VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR                 ((VkVideoCodecOperationFlagBitsKHR)0x00000005)
#endif
