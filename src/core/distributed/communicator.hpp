#pragma once

#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace uta {
namespace distributed {

enum class DeviceType {
    GPU,
    CPU,
    TPU,
    OTHER
};

struct DeviceInfo {
    DeviceType type;
    int device_id;
    size_t total_memory;
    size_t available_memory;
    std::string name;
    std::string capabilities;
};

enum class CommunicationPattern {
    ALL_REDUCE,
    ALL_GATHER,
    REDUCE_SCATTER,
    BROADCAST,
    POINT_TO_POINT
};

class Communicator {
public:
    static Communicator& getInstance();

    // Initialize the communication environment
    void initialize(const std::vector<DeviceInfo>& devices);
    
    // communication primitive
    void allReduce(void* buffer, size_t count, const std::string& dtype,
                  const std::string& reduction_op = "sum");
    
    void broadcast(void* buffer, size_t count, const std::string& dtype,
                  int root_rank);
    
    void send(const void* buffer, size_t count, const std::string& dtype,
              int destination);
    
    void receive(void* buffer, size_t count, const std::string& dtype,
                int source);

    // Advanced Communication Interface
    struct CollectiveOptions {
        bool async;                     // asynchronous operation
        bool enable_compression;        // enable compression
        float compression_ratio;        // compression ratio
        std::string compression_type;   // compression type
        bool use_nccl;                 // use NCCL
        bool use_mpi;                  // use MPI
    };

    void setCollectiveOptions(const CollectiveOptions& options);

    // Communication optimization
    struct CommunicationOptimizer {
        bool enable_fusion;            // enable operation fusion
        bool enable_overlapping;       // enable computation-communication overlapping
        size_t fusion_threshold;       // fusion threshold
        size_t buffer_size;           // communication buffer size
    };

    void setOptimizationOptions(const CommunicationOptimizer& options);

private:
    Communicator() = default;

    // internal optimization function
    void optimizeCommunication(CommunicationPattern pattern);
    void compressData(void* data, size_t size, const std::string& dtype);
    void decompressData(void* data, size_t size, const std::string& dtype);
    
    // communication backend
    class CommunicationBackend;
    std::unique_ptr<CommunicationBackend> backend_;
    
    // optimizer state
    CommunicationOptimizer optimizer_;
    CollectiveOptions collective_options_;
};

} // namespace distributed
} // namespace uta
