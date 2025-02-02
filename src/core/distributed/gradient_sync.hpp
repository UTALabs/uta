#pragma once

#include <vector>
#include <memory>
#include <string>
#include "communicator.hpp"

namespace uta {
namespace distributed {

class GradientSynchronizer {
public:
    static GradientSynchronizer& getInstance();

    // Layer synchronization strategy
    enum class SyncStrategy {
        SYNCHRONOUS,          // synchronous SGD
        ASYNCHRONOUS,         // asynchronous SGD
        HYBRID               // hybrid strategy
    };

    struct SyncConfig {
        SyncStrategy strategy;
        size_t local_steps;           // local update steps
        float staleness_threshold;    // asynchronous tolerance
        bool enable_compression;      // enable gradient compression
        float compression_ratio;      // compression ratio
    };

    // initialize synchronizer
    void initialize(const SyncConfig& config);

    // gradient synchronization interface
    void synchronizeGradients(
        std::vector<void*>& gradients,
        const std::vector<size_t>& sizes,
        const std::string& dtype
    );

    // compression strategy
    enum class CompressionType {
        NONE,
        QUANTIZATION,
        SPARSIFICATION,
        ADAPTIVE
    };

    struct CompressionConfig {
        CompressionType type;
        int bits_per_value;          // Quantification bit
        float sparsity_ratio;        // sparsity
        bool use_error_feedback;     // Whether to use error feedback
    };

    void setCompressionConfig(const CompressionConfig& config);

private:
    GradientSynchronizer() = default;

    // internal synchronization implementation
    void synchronousSGD(
        std::vector<void*>& gradients,
        const std::vector<size_t>& sizes
    );

    void asynchronousSGD(
        std::vector<void*>& gradients,
        const std::vector<size_t>& sizes
    );

    void hybridSGD(
        std::vector<void*>& gradients,
        const std::vector<size_t>& sizes
    );

    // compression correlation
    void compressGradients(
        void* data,
        size_t size,
        const CompressionConfig& config
    );

    void decompressGradients(
        void* data,
        size_t size,
        const CompressionConfig& config
    );

    // error feedback
    struct ErrorFeedback {
        std::vector<float> local_error;
        std::vector<float> global_error;
    };

    void updateErrorFeedback(
        const std::vector<float>& current_gradients,
        const std::vector<float>& compressed_gradients
    );

    // internal state
    SyncConfig sync_config_;
    CompressionConfig compression_config_;
    ErrorFeedback error_feedback_;
    std::shared_ptr<Communicator> communicator_;
};

} // namespace distributed
} // namespace uta
