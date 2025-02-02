#include <uta/uta.hpp>
#include <uta/profiler.hpp>
#include <iostream>
#include <vector>

// custom fusion operation
class FusedMLPOp {
public:
    static std::shared_ptr<uta::Tensor> forward(
        const std::shared_ptr<uta::Tensor>& input,
        const std::vector<std::shared_ptr<uta::Tensor>>& weights,
        const std::vector<std::shared_ptr<uta::Tensor>>& biases
    ) {
        auto x = input;
        
        for (size_t i = 0; i < weights.size(); ++i) {
            // Forward propagation operation of fusion:
            // 1. Matrix multiplication
            // 2. Bias addition
            // 3. ReLU activation
            // 4. Dropout
            x = uta::ops::fused_linear_relu_dropout(
                x, weights[i], biases[i], 0.1f
            );
        }
        
        return x;
    }
};

// Core Fusion Example
int main() {
    try {
        // Initialize UTA
        uta::initialize();

        // Create context
        auto context = uta::Context::create({
            .enabled_devices = {uta::DeviceType::CUDA},
            .enable_profiling = true
        });

        // Get device
        auto device = context->getDevice(uta::DeviceType::CUDA, 0);
        std::cout << "Using device: " << device->getName() << std::endl;

        // Configure performance profiler
        uta::profiler::Profiler::getInstance().configure({
            .enabled = true,
            .record_shapes = true,
            .record_memory = true,
            .record_bandwidth = true
        });

        // Create MLP parameters
        const size_t BATCH_SIZE = 128;
        const size_t INPUT_SIZE = 1024;
        const std::vector<size_t> HIDDEN_SIZES = {2048, 1024, 512};
        
        // Create input
        auto input = uta::Tensor::create(
            {BATCH_SIZE, INPUT_SIZE},
            uta::DataType::FLOAT32,
            *device
        );

        // Create weights and biases
        std::vector<std::shared_ptr<uta::Tensor>> weights;
        std::vector<std::shared_ptr<uta::Tensor>> biases;
        
        size_t in_features = INPUT_SIZE;
        for (size_t out_features : HIDDEN_SIZES) {
            weights.push_back(uta::Tensor::create(
                {in_features, out_features},
                uta::DataType::FLOAT32,
                *device
            ));
            
            biases.push_back(uta::Tensor::create(
                {out_features},
                uta::DataType::FLOAT32,
                *device
            ));
            
            in_features = out_features;
        }

        // Initialize parameters
        for (auto& w : weights) w->fill(0.1f);
        for (auto& b : biases) b->fill(0.0f);

        // Start performance profiling
        uta::profiler::Profiler::getInstance().start();

        // 1. Without Kernel Fusion
        std::shared_ptr<uta::Tensor> output_unfused;
        {
            UTA_PROFILE_SCOPE("Unfused_Forward");
            
            auto x = input;
            for (size_t i = 0; i < weights.size(); ++i) {
                // 1. Matrix multiplication
                auto linear = uta::Tensor::create(
                    {BATCH_SIZE, weights[i]->getShape()[1]},
                    uta::DataType::FLOAT32,
                    *device
                );
                uta::ops::matmul(*x, *weights[i], *linear);
                
                // 2. Bias addition
                uta::ops::add(*linear, *biases[i], *linear);
                
                // 3. ReLU activation
                uta::ops::relu(*linear, *linear);
                
                // 4. Dropout
                uta::ops::dropout(*linear, *linear, 0.1f);
                
                x = linear;
            }
            output_unfused = x;
        }

        // 2. Forward propagation using fusion
        std::shared_ptr<uta::Tensor> output_fused;
        {
            UTA_PROFILE_SCOPE("Fused_Forward");
            output_fused = FusedMLPOp::forward(input, weights, biases);
        }

        // sync device
        device->synchronize();

        // stop performance analysis
        uta::profiler::Profiler::getInstance().stop();

        // get performance statistics
        auto stats = uta::profiler::Profiler::getInstance().getStats();
        
        // print performance comparison
        std::cout << "\nPerformance Comparison:" << std::endl;
        
        double unfused_time = 0;
        double fused_time = 0;
        
        for (const auto& op : stats) {
            if (op.name == "Unfused_Forward") {
                unfused_time = op.metrics.execution_time;
                std::cout << "\nUnfused Forward:" << std::endl;
                std::cout << "  Time: " << op.metrics.execution_time << " ms" << std::endl;
                std::cout << "  Memory: " << op.metrics.memory_used << " bytes" << std::endl;
                std::cout << "  Bandwidth: " << op.metrics.bandwidth << " GB/s" << std::endl;
            }
            else if (op.name == "Fused_Forward") {
                fused_time = op.metrics.execution_time;
                std::cout << "\nFused Forward:" << std::endl;
                std::cout << "  Time: " << op.metrics.execution_time << " ms" << std::endl;
                std::cout << "  Memory: " << op.metrics.memory_used << " bytes" << std::endl;
                std::cout << "  Bandwidth: " << op.metrics.bandwidth << " GB/s" << std::endl;
            }
        }

        // computing speedup ratio
        double speedup = unfused_time / fused_time;
        std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

        // verifying results
        std::vector<float> h_unfused(output_unfused->getSize());
        std::vector<float> h_fused(output_fused->getSize());
        
        output_unfused->copyTo(h_unfused.data());
        output_fused->copyTo(h_fused.data());

        // Calculate the maximum error
        float max_error = 0.0f;
        for (size_t i = 0; i < h_unfused.size(); ++i) {
            max_error = std::max(max_error, std::abs(h_unfused[i] - h_fused[i]));
        }
        
        std::cout << "Maximum Error: " << max_error << std::endl;

        // Generate performance reports
        uta::profiler::Profiler::getInstance().generateReport("fusion_profile.json");

        // Clean up
        uta::finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
