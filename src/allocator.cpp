// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "allocator.h"

#include "gpu.h"
#include "pipeline.h"

#if __ANDROID_API__ >= 26
#include <android/hardware_buffer.h>
#endif // __ANDROID_API__ >= 26

namespace ncnn {

Allocator::~Allocator()
{
}

class PoolAllocatorPrivate
{
public:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

PoolAllocator::PoolAllocator()
    : Allocator(), d(new PoolAllocatorPrivate)
{
    d->size_compare_ratio = 0;
    d->size_drop_threshold = 10;
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if (!d->payouts.empty())
    {
        NCNN_LOGE("FATAL ERROR! pool allocator destroyed too early");
#if NCNN_STDIO
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); ++it)
        {
            void* ptr = it->second;
            NCNN_LOGE("%p still in use", ptr);
        }
#endif
    }

    delete d;
}

PoolAllocator::PoolAllocator(const PoolAllocator&)
    : d(0)
{
}

PoolAllocator& PoolAllocator::operator=(const PoolAllocator&)
{
    return *this;
}

void PoolAllocator::clear()
{
    d->budgets_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();

    d->budgets_lock.unlock();
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void PoolAllocator::set_size_drop_threshold(size_t threshold)
{
    d->size_drop_threshold = threshold;
}

void* PoolAllocator::fastMalloc(size_t size)
{
    d->budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->budgets_lock.unlock();

            d->payouts_lock.lock();

            d->payouts.push_back(std::make_pair(bs, ptr));

            d->payouts_lock.unlock();

            return ptr;
        }

        if (bs < it_min->first)
        {
            it_min = it;
        }
        if (bs > it_max->first)
        {
            it_max = it;
        }
    }

    if (d->budgets.size() >= d->size_drop_threshold)
    {
        // All chunks in pool are not chosen. Then try to drop some outdated
        // chunks and return them to OS.
        if (it_max->first < size)
        {
            // Current query is asking for a chunk larger than any cached chunks.
            // Then remove the smallest one.
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            // Current query is asking for a chunk smaller than any cached chunks.
            // Then remove the largest one.
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    d->budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts_lock.lock();

    d->payouts.push_back(std::make_pair(size, ptr));

    d->payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    d->payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->payouts_lock.unlock();

            d->budgets_lock.lock();

            d->budgets.push_back(std::make_pair(size, ptr));

            d->budgets_lock.unlock();

            return;
        }
    }

    d->payouts_lock.unlock();

    NCNN_LOGE("FATAL ERROR! pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

class UnlockedPoolAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

UnlockedPoolAllocator::UnlockedPoolAllocator()
    : Allocator(), d(new UnlockedPoolAllocatorPrivate)
{
    d->size_compare_ratio = 0;
    d->size_drop_threshold = 10;
}

UnlockedPoolAllocator::~UnlockedPoolAllocator()
{
    clear();

    if (!d->payouts.empty())
    {
        NCNN_LOGE("FATAL ERROR! unlocked pool allocator destroyed too early");
#if NCNN_STDIO
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); ++it)
        {
            void* ptr = it->second;
            NCNN_LOGE("%p still in use", ptr);
        }
#endif
    }

    delete d;
}

UnlockedPoolAllocator::UnlockedPoolAllocator(const UnlockedPoolAllocator&)
    : d(0)
{
}

UnlockedPoolAllocator& UnlockedPoolAllocator::operator=(const UnlockedPoolAllocator&)
{
    return *this;
}

void UnlockedPoolAllocator::clear()
{
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();
}

void UnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void UnlockedPoolAllocator::set_size_drop_threshold(size_t threshold)
{
    d->size_drop_threshold = threshold;
}

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }

        if (bs > it_max->first)
        {
            it_max = it;
        }
        if (bs < it_min->first)
        {
            it_min = it;
        }
    }

    if (d->budgets.size() >= d->size_drop_threshold)
    {
        if (it_max->first < size)
        {
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

#if NCNN_VULKAN
VkAllocator::VkAllocator(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    buffer_memory_type_index = (uint32_t)-1;
    image_memory_type_index = (uint32_t)-1;
    reserved_type_index = (uint32_t)-1;
    mappable = false;
    coherent = false;
}

VkAllocator::~VkAllocator()
{
    clear();
}

void VkAllocator::clear()
{
}

static inline size_t round_up(size_t n, size_t multiple)
{
    return (n + multiple - 1) / multiple * multiple;
}

static inline size_t round_down(size_t n, size_t multiple)
{
    return n / multiple * multiple;
}

int VkAllocator::flush(VkBufferMemory* ptr)
{
    if (coherent)
        return 0;

    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = ptr->memory;
    mappedMemoryRange.offset = round_down(ptr->offset, vkdev->info.non_coherent_atom_size());
    mappedMemoryRange.size = round_up(ptr->offset + ptr->capacity, vkdev->info.non_coherent_atom_size()) - mappedMemoryRange.offset;

    VkResult ret = vkFlushMappedMemoryRanges(vkdev->vkdevice(), 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkFlushMappedMemoryRanges failed %d", ret);
        return -1;
    }

    return 0;
}

int VkAllocator::invalidate(VkBufferMemory* ptr)
{
    if (coherent)
        return 0;

    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = ptr->memory;
    mappedMemoryRange.offset = round_down(ptr->offset, vkdev->info.non_coherent_atom_size());
    mappedMemoryRange.size = round_up(ptr->offset + ptr->capacity, vkdev->info.non_coherent_atom_size()) - mappedMemoryRange.offset;

    VkResult ret = vkInvalidateMappedMemoryRanges(vkdev->vkdevice(), 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkInvalidateMappedMemoryRanges failed %d", ret);
        return -1;
    }

    return 0;
}

VkBuffer VkAllocator::create_buffer(size_t size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = 0;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = 0;

    VkBuffer buffer = 0;
    VkResult ret = vkCreateBuffer(vkdev->vkdevice(), &bufferCreateInfo, 0, &buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateBuffer failed %d", ret);
        return 0;
    }

    return buffer;
}

VkDeviceMemory VkAllocator::allocate_memory(size_t size, uint32_t memory_type_index)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    return memory;
}

VkDeviceMemory VkAllocator::allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
    memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
    memoryDedicatedAllocateInfo.pNext = 0;
    memoryDedicatedAllocateInfo.image = image;
    memoryDedicatedAllocateInfo.buffer = buffer;
    memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    return memory;
}

VkImage VkAllocator::create_image(int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage)
{
    VkImageCreateInfo imageCreateInfo;
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    imageCreateInfo.pNext = 0;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
    imageCreateInfo.format = format;
    imageCreateInfo.extent.width = width;
    imageCreateInfo.extent.height = height;
    imageCreateInfo.extent.depth = depth;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = 0;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    VkResult ret = vkCreateImage(vkdev->vkdevice(), &imageCreateInfo, 0, &image);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImage failed %d %d %d %d %d %d %d", ret, width, height, depth, format, tiling, usage);
        return 0;
    }

    return image;
}

VkImageView VkAllocator::create_imageview(VkImage image, VkFormat format)
{
    VkImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = 0;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageview;
    VkResult ret = vkCreateImageView(vkdev->vkdevice(), &imageViewCreateInfo, 0, &imageview);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImageView failed %d", ret);
        return 0;
    }

    return imageview;
}

static inline size_t least_common_multiple(size_t a, size_t b)
{
    if (a == b)
        return a;

    if (a > b)
        return least_common_multiple(b, a);

    size_t lcm = b;
    while (lcm % a != 0)
    {
        lcm += b;
    }

    return lcm;
}

class VkBlobAllocatorPrivate
{
public:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<std::list<std::pair<size_t, size_t> > > buffer_budgets;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<std::list<std::pair<size_t, size_t> > > image_memory_budgets;
    std::vector<VkDeviceMemory> image_memory_blocks;
};

VkBlobAllocator::VkBlobAllocator(const VulkanDevice* _vkdev, size_t preferred_block_size)
    : VkAllocator(_vkdev), d(new VkBlobAllocatorPrivate)
{
    d->buffer_offset_alignment = vkdev->info.buffer_offset_alignment();
    d->bind_memory_offset_alignment = vkdev->info.buffer_image_granularity();

    if (vkdev->info.type() == 1)
    {
        // on integrated gpu, there may be device local only memory too, eg. AMD APU
        // assuming larger alignment always keeps us safe :)

        // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.memory_map_alignment());
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.non_coherent_atom_size());
    }

    d->block_size = alignSize(preferred_block_size, d->buffer_offset_alignment);
}

VkBlobAllocator::~VkBlobAllocator()
{
    clear();

    delete d;
}

VkBlobAllocator::VkBlobAllocator(const VkBlobAllocator&)
    : VkAllocator(0), d(0)
{
}

VkBlobAllocator& VkBlobAllocator::operator=(const VkBlobAllocator&)
{
    return *this;
}

void VkBlobAllocator::clear()
{
    //     NCNN_LOGE("VkBlobAllocator %lu", buffer_blocks.size());

    for (size_t i = 0; i < d->buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = d->buffer_blocks[i];

        //         std::list< std::pair<size_t, size_t> >::iterator it = buffer_budgets[i].begin();
        //         while (it != buffer_budgets[i].end())
        //         {
        //             NCNN_LOGE("VkBlobAllocator budget %p %lu %lu", ptr->buffer, it->first, it->second);
        //             it++;
        //         }

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->buffer_blocks.clear();

    d->buffer_budgets.clear();

    for (size_t i = 0; i < d->image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = d->image_memory_blocks[i];

        //         std::list< std::pair<size_t, size_t> >::iterator it = d->image_memory_budgets[i].begin();
        //         while (it != d->image_memory_budgets[i].end())
        //         {
        //             NCNN_LOGE("VkBlobAllocator budget %p %lu %lu", memory, it->first, it->second);
        //             it++;
        //         }

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    d->image_memory_blocks.clear();

    d->image_memory_budgets.clear();
}

VkBufferMemory* VkBlobAllocator::fastMalloc(size_t size)
{
    size_t aligned_size = alignSize(size, d->buffer_offset_alignment);

    const int buffer_block_count = d->buffer_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < buffer_block_count; i++)
    {
        std::list<std::pair<size_t, size_t> >::iterator it = d->buffer_budgets[i].begin();
        while (it != d->buffer_budgets[i].end())
        {
            size_t budget_size = it->second;
            if (budget_size < aligned_size)
            {
                it++;
                continue;
            }

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = d->buffer_blocks[i]->buffer;
            ptr->offset = it->first;
            ptr->memory = d->buffer_blocks[i]->memory;
            ptr->capacity = aligned_size;
            ptr->mapped_ptr = d->buffer_blocks[i]->mapped_ptr;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            // adjust buffer_budgets
            if (budget_size == aligned_size)
            {
                d->buffer_budgets[i].erase(it);
            }
            else
            {
                it->first += aligned_size;
                it->second -= aligned_size;
            }

            //             NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

            return ptr;
        }
    }

    size_t new_block_size = std::max(d->block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    block->offset = 0;

    // TODO respect VK_KHR_dedicated_allocation ?

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    // setup memory type and alignment
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type() == 1)
        {
            // integrated gpu, prefer unified memory
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

            // on amd integrated gpu, there is a faster and larger device-only heap
            uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
            uint32_t buffer_heap_index = memory_properties.memoryTypes[buffer_memory_type_index].heapIndex;
            uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
            if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
            {
                buffer_memory_type_index = device_local_memory_type_index;
            }
        }
        else
        {
            // discrete gpu, device local
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(buffer_memory_type_index);
        coherent = vkdev->is_coherent(buffer_memory_type_index);
    }

    block->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

    block->mapped_ptr = 0;
    if (mappable)
    {
        vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
    }

    d->buffer_blocks.push_back(block);

    // return sub buffer
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;
    ptr->capacity = aligned_size;
    ptr->mapped_ptr = block->mapped_ptr;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // adjust buffer_budgets
    std::list<std::pair<size_t, size_t> > budget;
    if (new_block_size > aligned_size)
    {
        budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
    }
    d->buffer_budgets.push_back(budget);

    //     NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

    return ptr;
}

void VkBlobAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkBlobAllocator F %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

    const int buffer_block_count = d->buffer_blocks.size();

    int block_index = -1;
    for (int i = 0; i < buffer_block_count; i++)
    {
        if (d->buffer_blocks[i]->buffer == ptr->buffer && d->buffer_blocks[i]->memory == ptr->memory)
        {
            block_index = i;
            break;
        }
    }

    if (block_index == -1)
    {
        NCNN_LOGE("FATAL ERROR! unlocked VkBlobAllocator get wild %p", ptr->buffer);

        delete ptr;

        return;
    }

    // merge
    std::list<std::pair<size_t, size_t> >::iterator it_merge_left = d->buffer_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it_merge_right = d->buffer_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it = d->buffer_budgets[block_index].begin();
    for (; it != d->buffer_budgets[block_index].end(); it++)
    {
        if (it->first + it->second == ptr->offset)
        {
            it_merge_left = it;
        }
        else if (ptr->offset + ptr->capacity == it->first)
        {
            it_merge_right = it;
        }
    }

    if (it_merge_left != d->buffer_budgets[block_index].end() && it_merge_right != d->buffer_budgets[block_index].end())
    {
        it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
        d->buffer_budgets[block_index].erase(it_merge_right);
    }
    else if (it_merge_left != d->buffer_budgets[block_index].end())
    {
        it_merge_left->second = ptr->offset + ptr->capacity - it_merge_left->first;
    }
    else if (it_merge_right != d->buffer_budgets[block_index].end())
    {
        it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->offset;
        it_merge_right->first = ptr->offset;
    }
    else
    {
        if (ptr->offset == 0)
        {
            // chain leading block
            d->buffer_budgets[block_index].push_front(std::make_pair(ptr->offset, ptr->capacity));
        }
        else
        {
            d->buffer_budgets[block_index].push_back(std::make_pair(ptr->offset, ptr->capacity));
        }
    }

    delete ptr;
}

VkImageMemory* VkBlobAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
{
    if (elempack != 1 && elempack != 4 && elempack != 8)
    {
        NCNN_LOGE("elempack must be 1 4 8");
        return 0;
    }

    // resolve format
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (elemsize / elempack == 4)
    {
        // fp32
        if (elempack == 1) format = VK_FORMAT_R32_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    if (elemsize / elempack == 2)
    {
        // fp16
        if (elempack == 1) format = VK_FORMAT_R16_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (elemsize / elempack == 1)
    {
        // int8
        if (elempack == 1) format = VK_FORMAT_R8_SINT;
        if (elempack == 4) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 8) format = VK_FORMAT_R8G8B8A8_SINT;
    }

    // resolve image width height depth
    int width = w;
    int height = h;
    int depth = c;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;

    if (width > (int)vkdev->info.max_image_dimension_3d() || height > (int)vkdev->info.max_image_dimension_3d() || depth > (int)vkdev->info.max_image_dimension_3d())
    {
        NCNN_LOGE("image dimension too large %d %d %d > %d", width, height, depth, (int)vkdev->info.max_image_dimension_3d());
        return 0;
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = create_image(width, height, depth, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
    ptr->format = format;

    // TODO respect VK_KHR_dedicated_allocation ?
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(vkdev->vkdevice(), ptr->image, &memoryRequirements);

    const size_t size = memoryRequirements.size;
    const size_t alignment = std::max((size_t)memoryRequirements.alignment, d->bind_memory_offset_alignment);

    size_t aligned_size = alignSize(size, alignment);

    const int image_memory_block_count = d->image_memory_blocks.size();

    // find first spare space in image_memory_blocks
    for (int i = 0; i < image_memory_block_count; i++)
    {
#if __APPLE__
        // HACK moltenvk v1.2.3 is unhappy for image binding with offset  :(
        break;
#endif

        std::list<std::pair<size_t, size_t> >::iterator it = d->image_memory_budgets[i].begin();
        while (it != d->image_memory_budgets[i].end())
        {
            // we cannot use it->first directly for base offset alignment
            size_t bind_base_offset = it->first;
            size_t bind_offset = alignSize(bind_base_offset, alignment);
            size_t budget_size = it->second;
            if (budget_size < aligned_size + (bind_offset - bind_base_offset))
            {
                it++;
                continue;
            }

            // bind at memory offset
            ptr->memory = d->image_memory_blocks[i];
            ptr->bind_offset = bind_offset;
            ptr->bind_capacity = aligned_size;

            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;

            ptr->imageview = create_imageview(ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            if (bind_base_offset != bind_offset)
            {
                // NOTE there is small offset inside bind_base_offset and bind_offset
                // adjust ptr->bind_offset and ptr->bind_capacity after vkBindImageMemory
                // so that memory management could be easier
                aligned_size += (bind_offset - bind_base_offset);

                ptr->bind_offset = bind_base_offset;
                ptr->bind_capacity = aligned_size;
            }

            // adjust image_memory_budgets
            if (budget_size == aligned_size)
            {
                d->image_memory_budgets[i].erase(it);
            }
            else
            {
                it->first += aligned_size;
                it->second -= aligned_size;
            }

            //             NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

            return ptr;
        }
    }

    // setup memory type and alignment
    if (image_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type() == 1)
        {
            // integrated gpu, prefer unified memory
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

            // on amd integrated gpu, there is a faster and larger device-only heap
            uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
            uint32_t buffer_heap_index = memory_properties.memoryTypes[image_memory_type_index].heapIndex;
            uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
            if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
            {
                image_memory_type_index = device_local_memory_type_index;
            }
        }
        else
        {
            // discrete gpu, device local
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(image_memory_type_index);
        coherent = vkdev->is_coherent(image_memory_type_index);
    }

    // create new block
    size_t new_block_size = std::max(d->block_size, aligned_size);

#if __APPLE__
    // HACK moltenvk v1.2.3 is unhappy for image binding with offset
    // always ignore block size for smaller memory footprint :(
    new_block_size = aligned_size;
#endif

    // bind at memory offset
    ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
    ptr->bind_offset = 0;
    ptr->bind_capacity = aligned_size;

    // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
    vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

    // do not allow host access to optimal tiling image
    ptr->mapped_ptr = 0;

    ptr->imageview = create_imageview(ptr->image, format);

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->command_refcount = 0;

    // adjust image_memory_budgets
    d->image_memory_blocks.push_back(ptr->memory);

    std::list<std::pair<size_t, size_t> > budget;
    if (new_block_size > aligned_size)
    {
        budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
    }
    d->image_memory_budgets.push_back(budget);

    //     NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

    return ptr;
}

void VkBlobAllocator::fastFree(VkImageMemory* ptr)
{
    //     NCNN_LOGE("VkBlobAllocator F %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

    const int image_memory_block_count = d->image_memory_blocks.size();

    int block_index = -1;
    for (int i = 0; i < image_memory_block_count; i++)
    {
        if (d->image_memory_blocks[i] == ptr->memory)
        {
            block_index = i;
            break;
        }
    }

    if (block_index == -1)
    {
        NCNN_LOGE("FATAL ERROR! unlocked VkBlobAllocator get wild %p", ptr->memory);

        if (!ptr->command_refcount)
        {
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }

        return;
    }

    // merge
    std::list<std::pair<size_t, size_t> >::iterator it_merge_left = d->image_memory_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it_merge_right = d->image_memory_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it = d->image_memory_budgets[block_index].begin();
    for (; it != d->image_memory_budgets[block_index].end(); it++)
    {
        if (it->first + it->second == ptr->bind_offset)
        {
            it_merge_left = it;
        }
        else if (ptr->bind_offset + ptr->bind_capacity == it->first)
        {
            it_merge_right = it;
        }
    }

    if (it_merge_left != d->image_memory_budgets[block_index].end() && it_merge_right != d->image_memory_budgets[block_index].end())
    {
        it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
        d->image_memory_budgets[block_index].erase(it_merge_right);
    }
    else if (it_merge_left != d->image_memory_budgets[block_index].end())
    {
        it_merge_left->second = ptr->bind_offset + ptr->bind_capacity - it_merge_left->first;
    }
    else if (it_merge_right != d->image_memory_budgets[block_index].end())
    {
        it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->bind_offset;
        it_merge_right->first = ptr->bind_offset;
    }
    else
    {
        if (ptr->bind_offset == 0)
        {
            // chain leading block
            d->image_memory_budgets[block_index].push_front(std::make_pair(ptr->bind_offset, ptr->bind_capacity));
        }
        else
        {
            d->image_memory_budgets[block_index].push_back(std::make_pair(ptr->bind_offset, ptr->bind_capacity));
        }
    }

    if (!ptr->command_refcount)
    {
        vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

        delete ptr;
    }
}

class VkWeightAllocatorPrivate
{
public:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<size_t> buffer_block_free_spaces;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<VkBufferMemory*> dedicated_buffer_blocks;
    std::vector<size_t> image_memory_block_free_spaces;
    std::vector<VkDeviceMemory> image_memory_blocks;
    std::vector<VkDeviceMemory> dedicated_image_memory_blocks;
};

VkWeightAllocator::VkWeightAllocator(const VulkanDevice* _vkdev, size_t preferred_block_size)
    : VkAllocator(_vkdev), d(new VkWeightAllocatorPrivate)
{
    d->buffer_offset_alignment = vkdev->info.buffer_offset_alignment();
    d->bind_memory_offset_alignment = vkdev->info.buffer_image_granularity();

    if (vkdev->info.type() == 1)
    {
        // on integrated gpu, there may be device local only memory too, eg. AMD APU
        // assuming larger alignment always keeps us safe :)

        // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.memory_map_alignment());
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.non_coherent_atom_size());
    }

    d->block_size = alignSize(preferred_block_size, d->buffer_offset_alignment);
}

VkWeightAllocator::~VkWeightAllocator()
{
    clear();

    delete d;
}

VkWeightAllocator::VkWeightAllocator(const VkWeightAllocator&)
    : VkAllocator(0), d(0)
{
}

VkWeightAllocator& VkWeightAllocator::operator=(const VkWeightAllocator&)
{
    return *this;
}

void VkWeightAllocator::clear()
{
    //     NCNN_LOGE("VkWeightAllocator %lu %lu", d->buffer_blocks.size(), d->dedicated_buffer_blocks.size());

    d->buffer_block_free_spaces.clear();

    for (size_t i = 0; i < d->buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = d->buffer_blocks[i];

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->buffer_blocks.clear();

    for (size_t i = 0; i < d->dedicated_buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = d->dedicated_buffer_blocks[i];

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->dedicated_buffer_blocks.clear();

    d->image_memory_block_free_spaces.clear();

    for (size_t i = 0; i < d->image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = d->image_memory_blocks[i];

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    d->image_memory_blocks.clear();

    for (size_t i = 0; i < d->dedicated_image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = d->dedicated_image_memory_blocks[i];

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    d->dedicated_image_memory_blocks.clear();
}

VkBufferMemory* VkWeightAllocator::fastMalloc(size_t size)
{
    //     NCNN_LOGE("VkWeightAllocator fastMalloc %lu", size);

    size_t aligned_size = alignSize(size, d->buffer_offset_alignment);

    const int buffer_block_count = d->buffer_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < buffer_block_count; i++)
    {
        size_t free_size = d->buffer_block_free_spaces[i];
        if (free_size >= aligned_size)
        {
            size_t block_offset = d->block_size - free_size;

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = d->buffer_blocks[i]->buffer;
            ptr->offset = block_offset;
            ptr->memory = d->buffer_blocks[i]->memory;
            ptr->capacity = aligned_size;
            ptr->mapped_ptr = d->buffer_blocks[i]->mapped_ptr;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            d->buffer_block_free_spaces[i] -= aligned_size;

            return ptr;
        }
    }

    size_t new_block_size = std::max(d->block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    block->offset = 0;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2() && vkdev->info.support_VK_KHR_dedicated_allocation())
    {
        VkBufferMemoryRequirementsInfo2KHR bufferMemoryRequirementsInfo2;
        bufferMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR;
        bufferMemoryRequirementsInfo2.pNext = 0;
        bufferMemoryRequirementsInfo2.buffer = block->buffer;

        VkMemoryRequirements2KHR memoryRequirements2;
        memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
        memoryRequirements2.pNext = 0;

        VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
        memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
        memoryDedicatedRequirements.pNext = 0;
        memoryRequirements2.pNext = &memoryDedicatedRequirements;

        vkdev->vkGetBufferMemoryRequirements2KHR(vkdev->vkdevice(), &bufferMemoryRequirementsInfo2, &memoryRequirements2);

        bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;

        if (dedicatedAllocation)
        {
            // setup memory type and alignment
            if (buffer_memory_type_index == (uint32_t)-1)
            {
                if (vkdev->info.type() == 1)
                {
                    // integrated gpu, prefer unified memory
                    buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

                    // on amd integrated gpu, there is a faster and larger device-only heap
                    uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                    const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
                    uint32_t buffer_heap_index = memory_properties.memoryTypes[buffer_memory_type_index].heapIndex;
                    uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
                    if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
                    {
                        buffer_memory_type_index = device_local_memory_type_index;
                    }
                }
                else
                {
                    // discrete gpu, device local
                    buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                }

                mappable = vkdev->is_mappable(buffer_memory_type_index);
                coherent = vkdev->is_coherent(buffer_memory_type_index);
            }

            block->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, buffer_memory_type_index, 0, block->buffer);

            // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
            vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

            block->mapped_ptr = 0;
            if (mappable)
            {
                vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
            }

            d->dedicated_buffer_blocks.push_back(block);

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = block->buffer;
            ptr->offset = 0;
            ptr->memory = block->memory;
            ptr->capacity = new_block_size;
            ptr->mapped_ptr = block->mapped_ptr;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    // setup memory type and alignment
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type() == 1)
        {
            // integrated gpu, prefer unified memory
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

            // on amd integrated gpu, there is a faster and larger device-only heap
            uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
            uint32_t buffer_heap_index = memory_properties.memoryTypes[buffer_memory_type_index].heapIndex;
            uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
            if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
            {
                buffer_memory_type_index = device_local_memory_type_index;
            }
        }
        else
        {
            // discrete gpu, device local
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(buffer_memory_type_index);
        coherent = vkdev->is_coherent(buffer_memory_type_index);
    }

    block->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

    //     NCNN_LOGE("VkWeightAllocator M %p", block->buffer);

    block->mapped_ptr = 0;
    if (mappable)
    {
        vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
    }

    d->buffer_blocks.push_back(block);

    d->buffer_block_free_spaces.push_back(new_block_size - aligned_size);

    // return sub buffer
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;
    ptr->capacity = aligned_size;
    ptr->mapped_ptr = block->mapped_ptr;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    return ptr;
}

void VkWeightAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkWeightAllocator F %p", ptr->buffer);

    delete ptr;
}

VkImageMemory* VkWeightAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
{
    if (elempack != 1 && elempack != 4 && elempack != 8 && elempack != 16 && elempack != 32 && elempack != 64)
    {
        NCNN_LOGE("elempack must be 1 4 8 16 32 64");
        return 0;
    }

    // resolve format
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (elemsize / elempack == 4)
    {
        // fp32
        if (elempack == 1) format = VK_FORMAT_R32_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 16) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 32) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 64) format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    if (elemsize / elempack == 2)
    {
        // fp16
        if (elempack == 1) format = VK_FORMAT_R16_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 16) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 32) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 64) format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (elemsize / elempack == 1)
    {
        // int8
        if (elempack == 1) format = VK_FORMAT_R8_SINT;
        if (elempack == 4) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 8) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 16) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 32) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 64) format = VK_FORMAT_R8G8B8A8_SINT;
    }

    // resolve image width height depth
    int width = w;
    int height = h;
    int depth = c;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;
    if (elempack == 16) width *= 4;
    if (elempack == 32) width *= 8;
    if (elempack == 64) width *= 16;

    if (width > (int)vkdev->info.max_image_dimension_3d() || height > (int)vkdev->info.max_image_dimension_3d() || depth > (int)vkdev->info.max_image_dimension_3d())
    {
        NCNN_LOGE("image dimension too large %d %d %d > %d", width, height, depth, (int)vkdev->info.max_image_dimension_3d());
        return 0;
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = create_image(width, height, depth, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
    ptr->format = format;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2() && vkdev->info.support_VK_KHR_dedicated_allocation())
    {
        VkImageMemoryRequirementsInfo2KHR imageMemoryRequirementsInfo2;
        imageMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR;
        imageMemoryRequirementsInfo2.pNext = 0;
        imageMemoryRequirementsInfo2.image = ptr->image;

        VkMemoryRequirements2KHR memoryRequirements2;
        memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
        memoryRequirements2.pNext = 0;

        VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
        memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
        memoryDedicatedRequirements.pNext = 0;
        memoryRequirements2.pNext = &memoryDedicatedRequirements;

        vkdev->vkGetImageMemoryRequirements2KHR(vkdev->vkdevice(), &imageMemoryRequirementsInfo2, &memoryRequirements2);

        bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;

        if (dedicatedAllocation)
        {
            // setup memory type and alignment
            if (image_memory_type_index == (uint32_t)-1)
            {
                if (vkdev->info.type() == 1)
                {
                    // integrated gpu, prefer unified memory
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

                    // on amd integrated gpu, there is a faster and larger device-only heap
                    uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                    const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
                    uint32_t buffer_heap_index = memory_properties.memoryTypes[image_memory_type_index].heapIndex;
                    uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
                    if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
                    {
                        image_memory_type_index = device_local_memory_type_index;
                    }
                }
                else
                {
                    // discrete gpu, device local
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                }

                mappable = vkdev->is_mappable(image_memory_type_index);
                coherent = vkdev->is_coherent(image_memory_type_index);
            }

            // bind memory
            ptr->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, image_memory_type_index, ptr->image, 0);
            ptr->bind_offset = 0;
            ptr->bind_capacity = memoryRequirements2.memoryRequirements.size;

            // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;

            ptr->imageview = create_imageview(ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            d->dedicated_image_memory_blocks.push_back(ptr->memory);

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(vkdev->vkdevice(), ptr->image, &memoryRequirements);

    const size_t size = memoryRequirements.size;
    const size_t alignment = std::max((size_t)memoryRequirements.alignment, d->bind_memory_offset_alignment);

    size_t aligned_size = alignSize(size, alignment);

    const int image_memory_block_count = d->image_memory_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < image_memory_block_count; i++)
    {
        // we cannot use image_memory_block_free_spaces[i] directly for base offset alignment
        size_t bind_base_offset = d->block_size - d->image_memory_block_free_spaces[i];
        size_t bind_offset = alignSize(bind_base_offset, alignment);
        if (d->image_memory_block_free_spaces[i] >= aligned_size + (bind_offset - bind_base_offset))
        {
            // bind at memory offset
            ptr->memory = d->image_memory_blocks[i];
            ptr->bind_offset = bind_offset;
            ptr->bind_capacity = aligned_size;

            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;

            ptr->imageview = create_imageview(ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            if (bind_base_offset != bind_offset)
            {
                // NOTE there is small offset inside bind_base_offset and bind_offset
                // adjust ptr->bind_offset and ptr->bind_capacity after vkBindImageMemory
                // so that memory management could be easier
                aligned_size += (bind_offset - bind_base_offset);

                ptr->bind_offset = bind_base_offset;
                ptr->bind_capacity = aligned_size;
            }

            d->image_memory_block_free_spaces[i] -= aligned_size;

            return ptr;
        }
    }

    // setup memory type and alignment
    if (image_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type() == 1)
        {
            // integrated gpu, prefer unified memory
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

            // on amd integrated gpu, there is a faster and larger device-only heap
            uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
            uint32_t buffer_heap_index = memory_properties.memoryTypes[image_memory_type_index].heapIndex;
            uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
            if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
            {
                image_memory_type_index = device_local_memory_type_index;
            }
        }
        else
        {
            // discrete gpu, device local
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(image_memory_type_index);
        coherent = vkdev->is_coherent(image_memory_type_index);
    }

    // create new block
    size_t new_block_size = std::max(d->block_size, aligned_size);

    // bind at memory offset
    ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
    ptr->bind_offset = 0;
    ptr->bind_capacity = aligned_size;

    // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
    vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

    // do not allow host access to optimal tiling image
    ptr->mapped_ptr = 0;

    ptr->imageview = create_imageview(ptr->image, format);

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->command_refcount = 0;

    d->image_memory_blocks.push_back(ptr->memory);
    d->image_memory_block_free_spaces.push_back(new_block_size - aligned_size);

    return ptr;
}

void VkWeightAllocator::fastFree(VkImageMemory* ptr)
{
    //     NCNN_LOGE("VkWeightAllocator F %p", ptr->memory);

    if (!ptr->command_refcount)
    {
        vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

        delete ptr;
    }
}

class VkStagingAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    std::list<VkBufferMemory*> buffer_budgets;
};

VkStagingAllocator::VkStagingAllocator(const VulkanDevice* _vkdev)
    : VkAllocator(_vkdev), d(new VkStagingAllocatorPrivate)
{
    mappable = true;
    coherent = true;

    d->size_compare_ratio = 192; // 0.75f * 256
}

VkStagingAllocator::~VkStagingAllocator()
{
    clear();

    delete d;
}

VkStagingAllocator::VkStagingAllocator(const VkStagingAllocator&)
    : VkAllocator(0), d(0)
{
}

VkStagingAllocator& VkStagingAllocator::operator=(const VkStagingAllocator&)
{
    return *this;
}

void VkStagingAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void VkStagingAllocator::clear()
{
    //     NCNN_LOGE("VkStagingAllocator %lu", buffer_budgets.size());

    for (std::list<VkBufferMemory*>::iterator it = d->buffer_budgets.begin(); it != d->buffer_budgets.end(); it++)
    {
        VkBufferMemory* ptr = *it;

        //         NCNN_LOGE("VkStagingAllocator F %p", ptr->buffer);

        vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->buffer_budgets.clear();
}

VkBufferMemory* VkStagingAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<VkBufferMemory*>::iterator it = d->buffer_budgets.begin();
    for (; it != d->buffer_budgets.end(); it++)
    {
        VkBufferMemory* ptr = *it;

        size_t capacity = ptr->capacity;

        // size_compare_ratio ~ 100%
        if (capacity >= size && ((capacity * d->size_compare_ratio) >> 8) <= size)
        {
            d->buffer_budgets.erase(it);

            //             NCNN_LOGE("VkStagingAllocator M %p %lu reused %lu", ptr->buffer, size, capacity);

            return ptr;
        }
    }

    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    // setup memory type
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    ptr->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

    ptr->capacity = size;

    vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);

    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    //     NCNN_LOGE("VkStagingAllocator M %p %lu", ptr->buffer, size);

    return ptr;
}

void VkStagingAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkStagingAllocator F %p", ptr->buffer);

    // return to buffer_budgets
    d->buffer_budgets.push_back(ptr);
}

VkImageMemory* VkStagingAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int /* elempack */)
{
    // staging image is mainly used for storing small piece of dynamic parameters
    // we allocate host memory as a fake image, it's simple and good

    const size_t size = w * h * c * elemsize;

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = 0;
    ptr->width = w;
    ptr->height = h;
    ptr->depth = c;
    ptr->format = VK_FORMAT_UNDEFINED;
    ptr->memory = 0;
    ptr->bind_offset = 0;
    ptr->bind_capacity = size;

    ptr->mapped_ptr = malloc(size);

    ptr->imageview = 0;

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ptr->command_refcount = 0;

    //     NCNN_LOGE("VkStagingAllocator M %p %d %d %d %d %d", ptr->image, dims, width, height, depth, format);

    return ptr;
}

void VkStagingAllocator::fastFree(VkImageMemory* ptr)
{
    //     NCNN_LOGE("VkStagingAllocator F %p", ptr->image);

    free(ptr->mapped_ptr);

    delete ptr;
}

class VkWeightStagingAllocatorPrivate
{
public:
};

VkWeightStagingAllocator::VkWeightStagingAllocator(const VulkanDevice* _vkdev)
    : VkAllocator(_vkdev), d(new VkWeightStagingAllocatorPrivate)
{
    mappable = true;
    coherent = true;
}

VkWeightStagingAllocator::~VkWeightStagingAllocator()
{
    delete d;
}

VkWeightStagingAllocator::VkWeightStagingAllocator(const VkWeightStagingAllocator&)
    : VkAllocator(0), d(0)
{
}

VkWeightStagingAllocator& VkWeightStagingAllocator::operator=(const VkWeightStagingAllocator&)
{
    return *this;
}

VkBufferMemory* VkWeightStagingAllocator::fastMalloc(size_t size)
{
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    // setup memory type
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    ptr->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

    ptr->capacity = size;

    vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);

    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    //     NCNN_LOGE("VkWeightStagingAllocator M %p %lu", ptr->buffer, size);

    return ptr;
}

void VkWeightStagingAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkWeightStagingAllocator F %p", ptr->buffer);

    vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
    vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

VkImageMemory* VkWeightStagingAllocator::fastMalloc(int /*w*/, int /*h*/, int /*c*/, size_t /*elemsize*/, int /*elempack*/)
{
    return 0;
}

void VkWeightStagingAllocator::fastFree(VkImageMemory* /*ptr*/)
{
}

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
VkAndroidHardwareBufferImageAllocator::VkAndroidHardwareBufferImageAllocator(const VulkanDevice* _vkdev, AHardwareBuffer* _hb)
    : VkAllocator(_vkdev), hb(_hb)
{
    samplerYcbcrConversion = 0;

    init();
}

VkAndroidHardwareBufferImageAllocator::~VkAndroidHardwareBufferImageAllocator()
{
    if (samplerYcbcrConversion)
    {
        vkdev->vkDestroySamplerYcbcrConversionKHR(vkdev->vkdevice(), samplerYcbcrConversion, 0);
        samplerYcbcrConversion = 0;
    }
}

VkAndroidHardwareBufferImageAllocator::VkAndroidHardwareBufferImageAllocator(const VkAndroidHardwareBufferImageAllocator&)
    : VkAllocator(0)
{
}

VkAndroidHardwareBufferImageAllocator& VkAndroidHardwareBufferImageAllocator::operator=(const VkAndroidHardwareBufferImageAllocator&)
{
    return *this;
}

VkBufferMemory* VkAndroidHardwareBufferImageAllocator::fastMalloc(size_t /*size*/)
{
    return 0;
}

void VkAndroidHardwareBufferImageAllocator::fastFree(VkBufferMemory* /*ptr*/)
{
}

VkImageMemory* VkAndroidHardwareBufferImageAllocator::fastMalloc(int /*w*/, int /*h*/, int /*c*/, size_t /*elemsize*/, int /*elempack*/)
{
    VkResult ret;

    VkExternalFormatANDROID externalFormat;
    externalFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    externalFormat.pNext = 0;
    externalFormat.externalFormat = bufferFormatProperties.externalFormat;

    VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo;
    externalMemoryImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    externalMemoryImageCreateInfo.pNext = &externalFormat,
    externalMemoryImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageCreateInfo;
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    imageCreateInfo.pNext = &externalMemoryImageCreateInfo;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_UNDEFINED;
    imageCreateInfo.extent.width = bufferDesc.width;
    imageCreateInfo.extent.height = bufferDesc.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = 0;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = 0;
    ret = vkCreateImage(vkdev->vkdevice(), &imageCreateInfo, 0, &image);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImage failed %d", ret);
        return 0;
    }

    // setup memory type
    if (image_memory_type_index == (uint32_t)-1)
    {
        image_memory_type_index = vkdev->find_memory_index(bufferProperties.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    VkImportAndroidHardwareBufferInfoANDROID importAndroidHardwareBufferInfo;
    importAndroidHardwareBufferInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importAndroidHardwareBufferInfo.pNext = 0;
    importAndroidHardwareBufferInfo.buffer = hb;

    VkMemoryDedicatedAllocateInfo memoryDedicatedAllocateInfo;
    memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    memoryDedicatedAllocateInfo.pNext = &importAndroidHardwareBufferInfo;
    memoryDedicatedAllocateInfo.image = image;
    memoryDedicatedAllocateInfo.buffer = VK_NULL_HANDLE;

    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;
    memoryAllocateInfo.allocationSize = bufferProperties.allocationSize;
    memoryAllocateInfo.memoryTypeIndex = image_memory_type_index;

    VkDeviceMemory memory = 0;
    ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    VkBindImageMemoryInfo bindImageMemoryInfo;
    bindImageMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    bindImageMemoryInfo.pNext = 0;
    bindImageMemoryInfo.image = image;
    bindImageMemoryInfo.memory = memory;
    bindImageMemoryInfo.memoryOffset = 0;
    ret = vkdev->vkBindImageMemory2KHR(vkdev->vkdevice(), 1, &bindImageMemoryInfo);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBindImageMemory2KHR failed %d", ret);
        vkDestroyImage(vkdev->vkdevice(), image, 0);
        return 0;
    }

    VkSamplerYcbcrConversionInfoKHR samplerYcbcrConversionInfo;
    samplerYcbcrConversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR;
    samplerYcbcrConversionInfo.pNext = &externalFormat;
    samplerYcbcrConversionInfo.conversion = samplerYcbcrConversion;

    VkImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = &samplerYcbcrConversionInfo;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = VK_FORMAT_UNDEFINED;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageview = 0;
    ret = vkCreateImageView(vkdev->vkdevice(), &imageViewCreateInfo, 0, &imageview);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImageView failed %d", ret);
        vkDestroyImage(vkdev->vkdevice(), image, 0);
        vkFreeMemory(vkdev->vkdevice(), memory, 0);
        return 0;
    }

    VkImageMemory* ptr = new VkImageMemory;
    ptr->image = image;
    ptr->memory = memory;
    ptr->imageview = imageview;
    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    return ptr;
}

void VkAndroidHardwareBufferImageAllocator::fastFree(VkImageMemory* ptr)
{
    vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
    vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

int VkAndroidHardwareBufferImageAllocator::init()
{
    AHardwareBuffer_describe(hb, &bufferDesc);

    VkResult ret;

    // resolve externalFormat
    bufferFormatProperties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
    bufferFormatProperties.pNext = 0;

    bufferProperties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    bufferProperties.pNext = &bufferFormatProperties;

    ret = vkdev->vkGetAndroidHardwareBufferPropertiesANDROID(vkdev->vkdevice(), hb, &bufferProperties);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetAndroidHardwareBufferPropertiesANDROID failed %d", ret);
        return -1;
    }

    // setup samplerYcbcrConversion
    VkExternalFormatANDROID externalFormat;
    externalFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    externalFormat.pNext = 0;
    externalFormat.externalFormat = bufferFormatProperties.externalFormat;

    VkSamplerYcbcrConversionCreateInfoKHR samplerYcbcrConversionCreateInfo;
    samplerYcbcrConversionCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR;
    samplerYcbcrConversionCreateInfo.pNext = &externalFormat;
    samplerYcbcrConversionCreateInfo.format = VK_FORMAT_UNDEFINED;
    samplerYcbcrConversionCreateInfo.ycbcrModel = bufferFormatProperties.suggestedYcbcrModel;
    samplerYcbcrConversionCreateInfo.ycbcrRange = bufferFormatProperties.suggestedYcbcrRange;
    samplerYcbcrConversionCreateInfo.components = bufferFormatProperties.samplerYcbcrConversionComponents;
    samplerYcbcrConversionCreateInfo.xChromaOffset = bufferFormatProperties.suggestedXChromaOffset;
    samplerYcbcrConversionCreateInfo.yChromaOffset = bufferFormatProperties.suggestedYChromaOffset;
    samplerYcbcrConversionCreateInfo.chromaFilter = VK_FILTER_NEAREST;
    samplerYcbcrConversionCreateInfo.forceExplicitReconstruction = VK_FALSE;

    ret = vkdev->vkCreateSamplerYcbcrConversionKHR(vkdev->vkdevice(), &samplerYcbcrConversionCreateInfo, 0, &samplerYcbcrConversion);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateSamplerYcbcrConversionKHR failed %d", ret);
        return -1;
    }

    return 0;
}

int VkAndroidHardwareBufferImageAllocator::width() const
{
    return bufferDesc.width;
}

int VkAndroidHardwareBufferImageAllocator::height() const
{
    return bufferDesc.height;
}

uint64_t VkAndroidHardwareBufferImageAllocator::external_format() const
{
    return bufferFormatProperties.externalFormat;
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

#endif // NCNN_VULKAN

} // namespace ncnn
