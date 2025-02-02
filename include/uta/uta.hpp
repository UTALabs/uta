#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace uta {

// forward statement
class Context;
class Tensor;
class Device;
class Stream;
class Event;

// API version
constexpr int UTA_VERSION_MAJOR = 1;
constexpr int UTA_VERSION_MINOR = 0;
constexpr int UTA_VERSION_PATCH = 0;

// error handling
enum class Status {
    OK,
    ERROR_INVALID_ARGUMENT,
    ERROR_OUT_OF_MEMORY,
    ERROR_DEVICE_LOST,
    ERROR_INVALID_OPERATION,
    ERROR_UNKNOWN
};

// Equipment type
enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    VULKAN
};

// data type
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64,
    UINT32,
    UINT64,
    BOOL
};

// Memory type
enum class MemoryType {
    HOST,
    DEVICE,
    MANAGED
};

// Context configuration
struct ContextConfig {
    std::vector<DeviceType> enabled_devices;
    bool enable_profiling;
    bool enable_debug;
    size_t memory_pool_size;
    std::string cache_dir;
};

// Device configuration
struct DeviceConfig {
    DeviceType type;
    int device_id;
    size_t memory_limit;
    bool enable_tensor_cores;
    bool enable_peer_access;
};

// Context class
class Context {
public:
    static std::shared_ptr<Context> create(const ContextConfig& config);
    
    // Device management
    std::shared_ptr<Device> getDevice(DeviceType type, int device_id);
    std::vector<std::shared_ptr<Device>> getDevices();
    
    // Memory management
    void* allocate(size_t size, MemoryType type);
    void deallocate(void* ptr, MemoryType type);
    
    // Synchronization
    void synchronize();
    
    // Error handling
    Status getLastError();
    std::string getErrorString(Status status);
};

// Equipment class
class Device {
public:
    // Equipment information
    DeviceType getType() const;
    int getId() const;
    std::string getName() const;
    size_t getMemoryCapacity() const;
    size_t getAvailableMemory() const;
    
    // Stream management
    std::shared_ptr<Stream> createStream();
    std::shared_ptr<Stream> getDefaultStream();
    
    // Event management
    std::shared_ptr<Event> createEvent();
    
    // Device control
    void synchronize();
    bool supportsPeerAccess(const Device& peer);
    void enablePeerAccess(const Device& peer);
};

// Tensor class
class Tensor {
public:
    // Tensor creation
    static std::shared_ptr<Tensor> create(
        const std::vector<size_t>& shape,
        DataType dtype,
        Device& device
    );
    
    // data access
    template<typename T>
    T* data();
    
    template<typename T>
    const T* data() const;
    
    // Tensor information
    std::vector<size_t> getShape() const;
    size_t getDim() const;
    size_t getSize() const;
    DataType getDataType() const;
    Device& getDevice() const;
    
    // data transmission
    void copyTo(Tensor& dst);
    void copyFrom(const Tensor& src);
    
    // Memory management
    void zero();
    void fill(const void* value);
};

// Stream class
class Stream {
public:
    // Stream control
    void synchronize();
    bool query();
    void wait(Event& event);
    
    // Memory operations
    void memcpy(void* dst, const void* src, size_t size);
    void memset(void* ptr, int value, size_t size);
    
    // Computation operations
    void launch(const std::function<void()>& kernel);
};

// Event class
class Event {
public:
    // Event control
    void record(Stream& stream);
    void synchronize();
    bool query();
    float elapsed(const Event& start);
};

// Global functions
Status initialize();
void finalize();
std::string getVersion();
std::vector<DeviceType> getAvailableDevices();

} // namespace uta
