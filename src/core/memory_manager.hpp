#pragma once

#include <cstdint>
#include <memory>
#include "device.hpp"

namespace uta {
namespace core {

class MemoryManager {
public:
    // Memory allocation strategies
    enum class AllocStrategy {
        ZERO_COPY,    // Direct mapping between host and device memory
        POOLED,       // Memory pooling for better reuse
        UNIFIED       // Unified memory access
    };

    static MemoryManager& getInstance();

    // Allocate device memory
    void* allocateDevice(size_t size, const Device& device, AllocStrategy strategy = AllocStrategy::POOLED);

    // Free device memory
    void freeDevice(void* ptr, const Device& device);

    // Memory transfer operations
    void copyHostToDevice(void* dst, const void* src, size_t size, const Device& device);
    void copyDeviceToHost(void* dst, const void* src, size_t size, const Device& device);
    void copyDeviceToDevice(void* dst, const void* src, size_t size, 
                          const Device& srcDevice, const Device& dstDevice);

    // Memory pool management
    void createMemoryPool(size_t initialSize, const Device& device);
    void releaseMemoryPool(const Device& device);

private:
    MemoryManager() = default;
    
    // Implementation details for different memory management strategies
    void* allocateZeroCopy(size_t size, const Device& device);
    void* allocatePooled(size_t size, const Device& device);
    void* allocateUnified(size_t size, const Device& device);

    // Memory pool implementation
    struct MemoryPool;
    std::unique_ptr<MemoryPool> memoryPool;
};

} // namespace core
} // namespace uta
