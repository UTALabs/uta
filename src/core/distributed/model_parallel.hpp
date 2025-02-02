#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "communicator.hpp"

namespace uta {
namespace distributed {

class ModelParallelManager {
public:
    static ModelParallelManager& getInstance();

    // model parallelism strategy
    enum class ParallelStrategy {
        PIPELINE,            // pipeline parallelism
        TENSOR,             // tensor parallelism
        HYBRID              // hybrid parallelism
    };

    struct ParallelConfig {
        ParallelStrategy strategy;
        size_t num_pipeline_stages;    // number of pipeline stages
        size_t micro_batch_size;       // micro batch size
        bool enable_activation_recomputation;  // enable activation recomputation
        bool enable_selective_recomputation;   // enable selective recomputation
    };

    // initialize parallel manager
    void initialize(const ParallelConfig& config);

    // model partition interface
    struct ModelPartition {
        std::vector<size_t> layer_indices;    // layer indices
        std::vector<size_t> tensor_splits;    // tensor splits
        int device_id;                        // device ID
    };

    std::vector<ModelPartition> partitionModel(
        const std::vector<size_t>& layer_sizes,
        const std::vector<size_t>& tensor_sizes
    );

    // pipeline execution
    struct PipelineSchedule {
        size_t num_micro_batches;
        std::vector<int> forward_schedule;
        std::vector<int> backward_schedule;
        bool enable_interleaved;
    };

    void executePipeline(const PipelineSchedule& schedule);

    // Tensor Parallel Operation
    void splitTensor(void* tensor, size_t size, const std::string& dtype);
    void mergeTensor(void* tensor, size_t size, const std::string& dtype);

    // memory optimization
    struct MemoryOptimization {
        bool enable_activation_checkpointing;
        bool enable_memory_efficient_attention;
        size_t max_memory_per_device;
    };

    void setMemoryOptimization(const MemoryOptimization& config);

private:
    ModelParallelManager() = default;

    // internal implementation
    void pipelineParallel(
        const std::vector<size_t>& layer_sizes,
        size_t micro_batch_size
    );

    void tensorParallel(
        const std::vector<size_t>& tensor_sizes,
        size_t num_devices
    );

    // Recalculation management
    struct RecomputationManager {
        std::unordered_map<int, std::vector<void*>> checkpoints;
        std::vector<bool> recompute_mask;
    };

    void manageCheckpoints(
        int layer_id,
        void* activation,
        size_t size
    );

    void recomputeActivations(
        int layer_id,
        const std::vector<void*>& inputs
    );

    // internal state
    ParallelConfig parallel_config_;
    MemoryOptimization memory_config_;
    RecomputationManager recomputation_manager_;
    std::shared_ptr<Communicator> communicator_;

    // scheduler
    class PipelineScheduler {
    public:
        void schedule(const PipelineSchedule& config);
        void optimize(size_t num_stages, size_t batch_size);
    private:
        std::vector<std::vector<int>> schedule_table_;
    };

    std::unique_ptr<PipelineScheduler> scheduler_;
};

} // namespace distributed
} // namespace uta
