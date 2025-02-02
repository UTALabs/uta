#include <uta/uta.hpp>
#include <iostream>
#include <random>

// Matrix multiplication example
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

        // Create tensors
        const size_t M = 1024;
        const size_t N = 1024;
        const size_t K = 1024;

        auto a = uta::Tensor::create({M, K}, uta::DataType::FLOAT32, *device);
        auto b = uta::Tensor::create({K, N}, uta::DataType::FLOAT32, *device);
        auto c = uta::Tensor::create({M, N}, uta::DataType::FLOAT32, *device);

        // Initialize data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        std::vector<float> h_a(M * K);
        std::vector<float> h_b(K * N);
        
        for (size_t i = 0; i < M * K; ++i) {
            h_a[i] = dis(gen);
        }
        
        for (size_t i = 0; i < K * N; ++i) {
            h_b[i] = dis(gen);
        }

        // Copy data to device
        a->copyFrom(h_a.data());
        b->copyFrom(h_b.data());

        // Start performance analysis
        uta::profiler::Profiler::getInstance().start();

        // Execute matrix multiplication
        {
            UTA_PROFILE_SCOPE("MatMul");
            uta::ops::matmul(*a, *b, *c);
        }

        // Wait for completion
        device->synchronize();

        // Stop performance analysis
        uta::profiler::Profiler::getInstance().stop();

        // Get performance statistics
        auto stats = uta::profiler::Profiler::getInstance().getStats();
        for (const auto& op : stats) {
            std::cout << "Operation: " << op.name << std::endl;
            std::cout << "  Time: " << op.metrics.execution_time << " ms" << std::endl;
            std::cout << "  FLOPS: " << op.metrics.flops_per_second << " FLOPS" << std::endl;
            std::cout << "  Memory: " << op.metrics.memory_used << " bytes" << std::endl;
            std::cout << "  Bandwidth: " << op.metrics.bandwidth << " GB/s" << std::endl;
        }

        
        uta::finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
