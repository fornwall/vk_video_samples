/*
* Copyright 2023 NVIDIA Corporation.
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
#ifndef _VULKANSEMAPHORESET_H_
#define _VULKANSEMAPHORESET_H_

#include <vector>
#include <atomic>
#include <iostream>
#include "VkCodecUtils/VulkanDeviceContext.h"

class VulkanSemaphoreSet
{
public:
    VulkanSemaphoreSet(): m_vkDevCtx() {}

    VkResult CreateSet(const VulkanDeviceContext* vkDevCtx, uint32_t numSemaphores,
                       VkSemaphoreCreateFlags flags = VkSemaphoreCreateFlags(), const void* pNext = nullptr) {

        DestroySet();

        m_vkDevCtx = vkDevCtx;
        m_semaphores.resize(numSemaphores);
        const VkSemaphoreCreateInfo semInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, pNext, flags };
        for (uint32_t semIdx = 0; semIdx < numSemaphores; semIdx++ ) {
            VkResult result = m_vkDevCtx->CreateSemaphore(*m_vkDevCtx, &semInfo, nullptr, &m_semaphores[semIdx]);
            if (result != VK_SUCCESS) {
                return result;
            }
        }
        return VK_SUCCESS;
    }

    void DestroySet() {
        if (m_vkDevCtx && !m_semaphores.empty()) {
            for (size_t semIdx = 0; semIdx < m_semaphores.size(); semIdx++) {
                if (m_semaphores[semIdx] != VK_NULL_HANDLE) {
                    m_vkDevCtx->DestroySemaphore(*m_vkDevCtx, m_semaphores[semIdx], nullptr);
                    m_semaphores[semIdx] = VK_NULL_HANDLE;
                }
            }
        }
    }

    VkSemaphore getSemaphore(size_t semIdx = 0) const {
        return m_semaphores[semIdx];
    }

    virtual ~VulkanSemaphoreSet() {
        DestroySet();
    }

private:
    const VulkanDeviceContext* m_vkDevCtx;
    std::vector<VkSemaphore>   m_semaphores;
};

#endif /* _VULKANSEMAPHORESET_H_ */
