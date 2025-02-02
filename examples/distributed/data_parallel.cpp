#include <uta/uta.hpp>
#include <uta/distributed.hpp>
#include <iostream>
#include <vector>

// Data parallelism training example
int main(int argc, char** argv) {
    try {
        // Initialize UTA
        uta::initialize();

        // Create context
        auto context = uta::Context::create({
            .enabled_devices = {uta::DeviceType::CUDA},
            .enable_profiling = true
        });

        // Create process group
        auto group = uta::distributed::ProcessGroup::create({
            .world_size = 4,
            .rank = std::stoi(argv[1]),
            .backend = "nccl",
            .init_method = "tcp://localhost:23456"
        });

        std::cout << "Process " << group->getRank() 
                  << " of " << group->getWorldSize() << std::endl;

        // Get local device
        auto device = context->getDevice(uta::DeviceType::CUDA, group->getRank());
        std::cout << "Using device: " << device->getName() << std::endl;

        // Create data parallelism trainer
        auto dp = uta::distributed::DataParallel::create({device});

        // Create model tensor
        const size_t BATCH_SIZE = 32;
        const size_t HIDDEN_SIZE = 1024;

        auto weights = uta::Tensor::create(
            {HIDDEN_SIZE, HIDDEN_SIZE}, 
            uta::DataType::FLOAT32, 
            *device
        );

        auto gradients = uta::Tensor::create(
            {HIDDEN_SIZE, HIDDEN_SIZE}, 
            uta::DataType::FLOAT32, 
            *device
        );

        // Initialize weights
        weights->fill(1.0f);

        // Start performance analysis
        uta::profiler::Profiler::getInstance().start();

        // Training loop
        for (int epoch = 0; epoch < 10; ++epoch) {
            std::cout << "Epoch " << epoch << std::endl;

            // Forward pass
            {
                UTA_PROFILE_SCOPE("Forward");
                dp->forward([&](uta::Device& device) {
                    // Simulate computation
                    auto local_input = uta::Tensor::create(
                        {BATCH_SIZE, HIDDEN_SIZE}, 
                        uta::DataType::FLOAT32, 
                        device
                    );
                    auto local_output = uta::Tensor::create(
                        {BATCH_SIZE, HIDDEN_SIZE}, 
                        uta::DataType::FLOAT32, 
                        device
                    );
                    uta::ops::matmul(*local_input, *weights, *local_output);
                });
            }

            // backpropagation
            {
                UTA_PROFILE_SCOPE("Backward");
                dp->backward([&](uta::Device& device) {
                    // Simulate gradient computation
                    auto local_grad = uta::Tensor::create(
                        {BATCH_SIZE, HIDDEN_SIZE}, 
                        uta::DataType::FLOAT32, 
                        device
                    );
                    uta::ops::matmul(*local_grad, *weights, *gradients);
                });
            }

            // Synchronization layer
            {
                UTA_PROFILE_SCOPE("GradientSync");
                dp->synchronizeGradients();
            }

            // update weight
            {
                UTA_PROFILE_SCOPE("Update");
                uta::ops::sgd(*weights, *gradients, 0.01f);
            }

            // sync device
            device->synchronize();
        }

        // stop performance analysis
        uta::profiler::Profiler::getInstance().stop();

        // generate performance report
        if (group->getRank() == 0) {
            uta::profiler::Profiler::getInstance().generateReport(
                "data_parallel_profile.json"
            );
        }

        // clean up
        uta::finalize();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
