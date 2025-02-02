#include <uta/uta.hpp>
#include <uta/profiler.hpp>
#include <iostream>
#include <vector>

// Memory analysis example
int main() {
    try {
        // Initialize UTA
        uta::initialize();

        // Create context
        auto context = uta::Context::create({
            .enabled_devices = {uta::DeviceType::CUDA},
            .enable_profiling = true,
            .memory_pool_size = 1024 * 1024 * 1024  // 1GB
        });

        // Acquire equipment
        auto device = context->getDevice(uta::DeviceType::CUDA, 0);
        std::cout << "Using device: " << device->getName() << std::endl;

        // Configure Performance Analyzer
        uta::profiler::Profiler::getInstance().configure({
            .enabled = true,
            .record_shapes = true,
            .record_memory = true,
            .record_bandwidth = true,
            .output_dir = "profile_output"
        });

        // startup performance analysis
        uta::profiler::Profiler::getInstance().start();

        // Create a Memory Profiler
        auto memory_analyzer = uta::profiler::MemoryAnalyzer::create();
        memory_analyzer->start();

        // Allocate and use memory
        std::vector<std::shared_ptr<uta::Tensor>> tensors;
        
        for (int i = 0; i < 10; ++i) {
            // Allocate tensor
            auto tensor = uta::Tensor::create(
                {1024, 1024}, 
                uta::DataType::FLOAT32, 
                *device
            );
            tensors.push_back(tensor);

            // Execute some operations
            {
                UTA_PROFILE_SCOPE("Computation_" + std::to_string(i));
                tensor->zero();
                auto temp = uta::Tensor::create(
                    {1024, 1024}, 
                    uta::DataType::FLOAT32, 
                    *device
                );
                uta::ops::matmul(*tensor, *tensor, *temp);
            }
        }

        // sync device
        device->synchronize();

        // Get memory statistics
        auto memory_stats = memory_analyzer->getStats();
        
        std::cout << "Memory Statistics:" << std::endl;
        std::cout << "Peak Memory Usage: " << memory_stats.peak_memory << " bytes" << std::endl;
        std::cout << "Current Memory Usage: " << memory_stats.current_memory << " bytes" << std::endl;
        std::cout << "Total Allocations: " << memory_stats.total_allocations << std::endl;
        std::cout << "Total Deallocations: " << memory_stats.total_deallocations << std::endl;

        // Analyze memory leaks
        auto leaks = memory_analyzer->detectLeaks();
        if (!leaks.empty()) {
            std::cout << "\nPotential Memory Leaks:" << std::endl;
            for (const auto& leak : leaks) {
                std::cout << "Address: " << leak.address 
                          << ", Size: " << leak.size 
                          << " bytes, Allocation Site: " << leak.stack_trace 
                          << std::endl;
            }
        }

        // Generate memory timeline
        memory_analyzer->generateTimeline("memory_timeline.json");

        // Analyze bandwidth utilization
        auto bandwidth_stats = memory_analyzer->analyzeBandwidth();
        std::cout << "\nBandwidth Statistics:" << std::endl;
        std::cout << "Peak Bandwidth: " << bandwidth_stats.peak_bandwidth << " GB/s" << std::endl;
        std::cout << "Average Bandwidth: " << bandwidth_stats.average_bandwidth << " GB/s" << std::endl;

        // Analyze cache performance
        auto cache_stats = memory_analyzer->analyzeCachePerformance();
        std::cout << "\nCache Statistics:" << std::endl;
        std::cout << "L1 Cache Hit Rate: " << cache_stats.l1_hit_rate * 100 << "%" << std::endl;
        std::cout << "L2 Cache Hit Rate: " << cache_stats.l2_hit_rate * 100 << "%" << std::endl;

        // Generate memory access pattern report
        memory_analyzer->generateAccessPatternReport("memory_patterns.json");

        // Stop analysis
        memory_analyzer->stop();
        uta::profiler::Profiler::getInstance().stop();

        // Generate full performance report
        uta::profiler::Profiler::getInstance().generateReport("profile.json");

        // Cleanup
        tensors.clear();
        uta::finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
