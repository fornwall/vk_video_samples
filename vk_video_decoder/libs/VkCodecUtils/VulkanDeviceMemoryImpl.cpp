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

#include <string.h>
#include "VkCodecUtils/VulkanDeviceMemoryImpl.h"
#include "VkCodecUtils/Helpers.h"

VkResult
VulkanDeviceMemoryImpl::Create(const VulkanDeviceContext* vkDevCtx,
                               const VkMemoryRequirements& memoryRequirements,
                               VkMemoryPropertyFlags& memoryPropertyFlags,
                               const void* pInitializeMemory, size_t initializeMemorySize, bool clearMemory,
                               VkSharedBaseObj<VulkanDeviceMemoryImpl>& vulkanDeviceMemory)
{
    VkSharedBaseObj<VulkanDeviceMemoryImpl> vkDeviceMemory(new VulkanDeviceMemoryImpl(vkDevCtx));
    if (!vkDeviceMemory) {
        assert(!"Couldn't allocate host memory!");
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VkResult result = vkDeviceMemory->Initialize(memoryRequirements, memoryPropertyFlags,
                                                 pInitializeMemory,
                                                 initializeMemorySize,
                                                 clearMemory);
    if (result == VK_SUCCESS) {
        vulkanDeviceMemory = vkDeviceMemory;
    }

    return result;
}

VkResult VulkanDeviceMemoryImpl::CreateDeviceMemory(const VulkanDeviceContext* vkDevCtx,
                                                    const VkMemoryRequirements& memoryRequirements,
                                                    VkMemoryPropertyFlags& memoryPropertyFlags,
                                                    VkDeviceMemory& deviceMemory,
                                                    VkDeviceSize&   deviceMemoryOffset)
{
    assert(memoryRequirements.size ==
            ((memoryRequirements.size + (memoryRequirements.alignment - 1)) & ~(memoryRequirements.alignment - 1)));
    deviceMemoryOffset = 0;

    VkMemoryAllocateInfo allocInfo = VkMemoryAllocateInfo();
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.memoryTypeIndex = 0;  // Memory type assigned in the next step

    // Assign the proper memory type for that buffer
    allocInfo.allocationSize = memoryRequirements.size;
    MapMemoryTypeToIndex(vkDevCtx, vkDevCtx->getPhysicalDevice(),
                         memoryRequirements.memoryTypeBits,
                         memoryPropertyFlags,
                         &allocInfo.memoryTypeIndex);

    // Allocate memory for the buffer
    VkResult result = vkDevCtx->AllocateMemory(*vkDevCtx, &allocInfo, nullptr, &deviceMemory);
    if (result != VK_SUCCESS) {
        assert(!"Couldn't allocate device memory!");
        return result;
    }

    return result;
}

VkResult VulkanDeviceMemoryImpl::Initialize(const VkMemoryRequirements& memoryRequirements,
                                            VkMemoryPropertyFlags& memoryPropertyFlags,
                                            const void* pInitializeMemory,
                                            size_t initializeMemorySize,
                                            bool clearMemory)
{
    if (m_memoryRequirements.size >= memoryRequirements.size) {
        size_t ret = MemsetData(0x00, 0, m_memoryRequirements.size);
        if (ret != m_memoryRequirements.size) {
            assert(!"Couldn't allocate device memory!");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return VK_SUCCESS;
    }

    Deinitialize();

    VkResult result = CreateDeviceMemory(m_vkDevCtx,
                                         memoryRequirements,
                                         memoryPropertyFlags,
                                         m_deviceMemory,
                                         m_deviceMemoryOffset);

    if (result != VK_SUCCESS) {
        assert(!"Couldn't CreateDeviceMemory()!");
        return result;
    }

    m_memoryPropertyFlags = memoryPropertyFlags;
    m_memoryRequirements = memoryRequirements;

    if (m_memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {

        size_t copySize = std::min(initializeMemorySize, m_memoryRequirements.size);

        CopyDataFromBuffer((const uint8_t*)pInitializeMemory,
                           0, // srcOffset
                           0, // dstOffset
                           copySize);

        if (clearMemory) {
            MemsetData(0x0, copySize, m_memoryRequirements.size - copySize);
        }
    }
    return result;
}

void VulkanDeviceMemoryImpl::Deinitialize()
{
    if (m_deviceMemoryDataPtr != nullptr) {
        m_vkDevCtx->UnmapMemory(*m_vkDevCtx, m_deviceMemory);
        m_deviceMemoryDataPtr = nullptr;
    }

    if (m_deviceMemory) {
        m_vkDevCtx->FreeMemory(*m_vkDevCtx, m_deviceMemory, nullptr);
        m_deviceMemory = VK_NULL_HANDLE;
    }

    m_deviceMemoryOffset = 0;
}

VkResult VulkanDeviceMemoryImpl::FlushInvalidateMappedMemoryRange(VkDeviceSize offset, VkDeviceSize size,
                                                                  bool flush) const
{
    VkResult result = VK_SUCCESS;

    if (((m_memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) &&
        ((m_memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0)) {

        const VkMappedMemoryRange range = {
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,  // sType
            NULL,                                   // pNext
            m_deviceMemory,                         // memory
            offset,                                 // offset
            size,                                   // size
        };

        if (flush) {
            result = m_vkDevCtx->FlushMappedMemoryRanges(*m_vkDevCtx, 1u, &range);
        } else {
            result = m_vkDevCtx->InvalidateMappedMemoryRanges(*m_vkDevCtx, 1u, &range);
        }
    }

    return result;
}

void VulkanDeviceMemoryImpl::FlushRange(size_t offset, size_t size) const
{
    FlushInvalidateMappedMemoryRange(offset, size);
}

void VulkanDeviceMemoryImpl::InvalidateRange(size_t offset, size_t size) const
{
    FlushInvalidateMappedMemoryRange(offset, size, false);
}

VkResult VulkanDeviceMemoryImpl::CopyDataToMemory(const uint8_t* pData,
                                                  VkDeviceSize size,
                                                  VkDeviceSize memoryOffset) const
{
    if ((pData == nullptr) || (size == 0)) {
        assert(!"Couldn't CopyDataToMemory()!");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    uint8_t* pDst = NULL;
    assert((memoryOffset + size) <= m_memoryRequirements.size);
    VkResult result = m_vkDevCtx->MapMemory(*m_vkDevCtx, m_deviceMemory, memoryOffset,
                                            size, 0, (void**)&pDst);

    if (result != VK_SUCCESS) {
        return result;
    }

    memcpy(pDst, pData, (size_t)size);

    result = FlushInvalidateMappedMemoryRange(memoryOffset, size);
    if (result != VK_SUCCESS) {
        assert(!"Couldn't FlushMappedMemoryRange()!");
        return result;
    }

    m_vkDevCtx->UnmapMemory(*m_vkDevCtx, m_deviceMemory);

    return VK_SUCCESS;
}

size_t VulkanDeviceMemoryImpl::GetMaxSize() const
{
    return m_memoryRequirements.size;
}

size_t VulkanDeviceMemoryImpl::GetSizeAlignment() const
{
    return m_memoryRequirements.alignment;
}

size_t VulkanDeviceMemoryImpl::Resize(size_t newSize, size_t copySize, size_t copyOffset)
{
    if (m_memoryRequirements.size >= newSize) {
        return VK_SUCCESS;
    }

    VkMemoryRequirements memoryRequirements(m_memoryRequirements);
    memoryRequirements.size = ((newSize + (memoryRequirements.alignment - 1)) & ~(memoryRequirements.alignment - 1));
    VkDeviceMemory  newDeviceMemory = VK_NULL_HANDLE;
    VkDeviceSize    newBufferOffset = 0;
    VkMemoryPropertyFlags newMemoryPropertyFlags = m_memoryPropertyFlags;
    VkResult result = CreateDeviceMemory(m_vkDevCtx,
                                         memoryRequirements,
                                         newMemoryPropertyFlags,
                                         newDeviceMemory,
                                         newBufferOffset);

    if (result != VK_SUCCESS) {
        assert(!"Couldn't CreateDeviceMemory()!");
        return 0;
    }

    uint8_t* newBufferDataPtr = nullptr;
    if (copySize != 0) {
        result = m_vkDevCtx->MapMemory(*m_vkDevCtx, newDeviceMemory, newBufferOffset,
                                        newSize, 0, (void**)&newBufferDataPtr);

        if ((result != VK_SUCCESS) || (newBufferDataPtr == nullptr)) {
            m_vkDevCtx->UnmapMemory(*m_vkDevCtx, newDeviceMemory);
            m_vkDevCtx->FreeMemory(*m_vkDevCtx, newDeviceMemory, nullptr);
            assert(!"Couldn't MapMemory()!");
            return 0;
        }

        copySize = std::min(copyOffset + copySize, m_memoryRequirements.size);
        memset(newBufferDataPtr + copySize, 0x00, newSize - copySize);

        // Copy the old data.
        uint8_t* readData = CheckAccess(copyOffset, copySize);
        memcpy(newBufferDataPtr, readData, copySize);
    }

    Deinitialize();

    m_memoryRequirements = memoryRequirements;
    m_memoryPropertyFlags = newMemoryPropertyFlags;
    m_deviceMemory = newDeviceMemory;
    m_deviceMemoryOffset = newBufferOffset;
    m_deviceMemoryDataPtr = newBufferDataPtr;

    if (copySize == 0) {
        MemsetData(0x0, 0, newSize);
    }

    return newSize;
}

uint8_t* VulkanDeviceMemoryImpl::CheckAccess(size_t offset, size_t size) const
{
    if (offset + size <= m_memoryRequirements.size) {
        if (m_deviceMemoryDataPtr == nullptr) {
            VkResult result = m_vkDevCtx->MapMemory(*m_vkDevCtx, m_deviceMemory, m_deviceMemoryOffset,
                                                    m_memoryRequirements.size, 0, (void**)&m_deviceMemoryDataPtr);
            if ((result != VK_SUCCESS) || (m_deviceMemoryDataPtr == nullptr)) {
                assert(!"Couldn't MapMemory()!");
                return nullptr;
            }
        }
        return m_deviceMemoryDataPtr + offset;
    }

    assert(!"CheckAccess() failed - buffer out of range!");
    return nullptr;
}

int64_t VulkanDeviceMemoryImpl::MemsetData(uint32_t value, size_t offset, size_t size)
{
    if (size == 0) {
        return 0;
    }
    uint8_t* setData = CheckAccess(offset, size);
    if (setData == nullptr) {
        assert(!"MemsetData() failed - buffer out of range!");
        return -1;
    }
    memset(setData, value, size);
    return size;
}

int64_t VulkanDeviceMemoryImpl::CopyDataToBuffer(uint8_t *dstBuffer, size_t dstOffset,
                                                 size_t srcOffset, size_t size) const
{
    if (size == 0) {
        return 0;
    }
    const uint8_t* readData = CheckAccess(srcOffset, size);
    if (readData == nullptr) {
        assert(!"CopyDataToBuffer() failed - buffer out of range!");
        return -1;
    }
    memcpy(dstBuffer + dstOffset, readData, size);
    return size;
}

int64_t VulkanDeviceMemoryImpl::CopyDataToBuffer(VkSharedBaseObj<VulkanDeviceMemoryImpl>& dstBuffer, size_t dstOffset,
                                                 size_t srcOffset, size_t size) const
{
    if (size == 0) {
        return 0;
    }
    const uint8_t* readData = CheckAccess(srcOffset, size);
    if (readData == nullptr) {
        assert(!"CopyDataToBuffer() failed - buffer out of range!");
        return -1;
    }

    dstBuffer->CopyDataFromBuffer(readData, 0, dstOffset, size);
    return size;
}

int64_t  VulkanDeviceMemoryImpl::CopyDataFromBuffer(const uint8_t* sourceBuffer, size_t srcOffset,
                                                    size_t dstOffset, size_t size)
{
    uint8_t* writeData = CheckAccess(dstOffset, size);
    if (writeData == nullptr) {
        assert(!"CopyDataFromBuffer() failed - buffer out of range!");
        return -1;
    }
    if ((size != 0) && (sourceBuffer != nullptr)) {
        memcpy(writeData, sourceBuffer + srcOffset, size);
    }
    return size;
}

int64_t VulkanDeviceMemoryImpl::CopyDataFromBuffer(const VkSharedBaseObj<VulkanDeviceMemoryImpl>& sourceMemory,
                                                   size_t srcOffset, size_t dstOffset, size_t size)
{
    if (size == 0) {
        return 0;
    }
    uint8_t* writeData = CheckAccess(dstOffset, size);
    if (writeData == nullptr) {
        assert(!"CopyDataFromBuffer() failed - buffer out of range!");
        return -1;
    }

    size_t maxSize = 0;
    const uint8_t* srcPtr = sourceMemory->GetReadOnlyDataPtr(srcOffset, maxSize);
    if ((srcPtr == nullptr) || (maxSize < size)) {
        assert(!"GetReadOnlyDataPtr() failed - buffer out of range!");
        return -1;
    }

    memcpy(writeData, srcPtr, size);
    return size;
}

uint8_t* VulkanDeviceMemoryImpl::GetDataPtr(size_t offset, size_t &maxSize)
{
    uint8_t* readData = CheckAccess(offset, 1);
    if (readData == nullptr) {
        assert(!"GetDataPtr() failed - buffer out of range!");
        return nullptr;
    }
    maxSize = m_memoryRequirements.size - offset;
    return (uint8_t*)readData;
}

const uint8_t* VulkanDeviceMemoryImpl::GetReadOnlyDataPtr(size_t offset, size_t &maxSize) const
{
    const uint8_t* readData = CheckAccess(offset, 1);
    if (readData == nullptr) {
        assert(!"GetReadOnlyDataPtr() failed - buffer out of range!");
        return nullptr;
    }
    maxSize = m_memoryRequirements.size - offset;
    return readData;
}
