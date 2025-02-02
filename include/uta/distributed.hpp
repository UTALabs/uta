#pragma once

#include "uta.hpp"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace uta {
namespace distributed {

// distributed configuration
struct DistributedConfig {
    int world_size;
    int rank;
    std::string backend;
    std::string init_method;
    std::vector<std::string> host_list;
};

// communication group
class ProcessGroup {
public:
    static std::shared_ptr<ProcessGroup> create(
        const DistributedConfig& config
    );
    
    // basic information
    int getRank() const;
    int getWorldSize() const;
    
    // communication primitives
    void broadcast(Tensor& tensor, int root_rank);
    void allReduce(Tensor& tensor, const std::string& op = "sum");
    void reduce(Tensor& tensor, int root_rank, const std::string& op = "sum");
    void allGather(const Tensor& tensor, std::vector<Tensor>& output_tensors);
    void gather(const Tensor& tensor, std::vector<Tensor>& output_tensors, int root_rank);
    void scatter(const std::vector<Tensor>& input_tensors, Tensor& output_tensor, int root_rank);
    
    // peer-to-peer communication
    void send(const Tensor& tensor, int dst_rank);
    void receive(Tensor& tensor, int src_rank);
    
    // synchronization
    void barrier();
};

// distributed tensor
class DistributedTensor {
public:
    static std::shared_ptr<DistributedTensor> create(
        const std::vector<size_t>& shape,
        DataType dtype,
        const std::vector<int>& device_ids
    );
    
    // Sharding management
    void partition(const std::vector<size_t>& dims);
    void replicate();
    
    // data access
    std::shared_ptr<Tensor> getLocalTensor();
    std::vector<std::shared_ptr<Tensor>> getAllTensors();
    
    // synchronization
    void synchronize();
};

// distributed operation
namespace ops {

// distributed matrix multiplication
std::shared_ptr<DistributedTensor> distributedMatmul(
    const DistributedTensor& a,
    const DistributedTensor& b
);

// distributed batch normalization
std::shared_ptr<DistributedTensor> distributedBatchNorm(
    const DistributedTensor& input,
    const Tensor& scale,
    const Tensor& bias,
    float epsilon = 1e-5
);

// distributed optimizer
void distributedSGD(
    DistributedTensor& param,
    const DistributedTensor& grad,
    float learning_rate,
    float momentum = 0.0,
    float weight_decay = 0.0
);

void distributedAdam(
    DistributedTensor& param,
    DistributedTensor& m,
    DistributedTensor& v,
    const DistributedTensor& grad,
    float learning_rate,
    float beta1 = 0.9,
    float beta2 = 0.999,
    float epsilon = 1e-8
);

} // namespace ops

// distributed model parallel
class DistributedModelParallel {
public:
    // parallel configuration
    struct ParallelConfig {
        int pipeline_stages;
        int tensor_parallel_size;
        bool enable_activation_checkpointing;
        size_t micro_batch_size;
    };
    
    static std::shared_ptr<ModelParallel> create(
        const ParallelConfig& config
    );
    
    // model partitioning
    void partition(const std::function<void(int stage)>& stage_fn);
    
    // executive control
    void forward(const std::vector<Tensor>& inputs);
    void backward(const std::vector<Tensor>& grad_outputs);
    
    // synchronization
    void synchronize();
};

// data parallelism
class DataParallel {
public:
    static std::shared_ptr<DataParallel> create(
        const std::vector<Device>& devices
    );
    
    // data distribution
    void scatter(const Tensor& input);
    void gather(Tensor& output);
    
    // gradient synchronization
    void synchronizeGradients();
    
    // executive control
    void forward(const std::function<void(Device&)>& forward_fn);
    void backward(const std::function<void(Device&)>& backward_fn);
};

} // namespace distributed
} // namespace uta
