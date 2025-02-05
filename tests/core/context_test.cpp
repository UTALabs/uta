#include <gtest/gtest.h>
#include <uta/uta.hpp>

class ContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        uta::initialize();
    }

    void TearDown() override {
        uta::finalize();
    }
};

TEST_F(ContextTest, CreateContext) {
    auto context = uta::Context::create({
        .enabled_devices = {uta::DeviceType::GPU},
        .enable_profiling = true
    });
    ASSERT_NE(context, nullptr);
}

TEST_F(ContextTest, GetDevices) {
    auto context = uta::Context::create({
        .enabled_devices = {uta::DeviceType::GPU},
        .enable_profiling = true
    });
    auto devices = context->getDevices();
    ASSERT_GT(devices.size(), 0);
}

TEST_F(ContextTest, MemoryAllocation) {
    auto context = uta::Context::create({
        .enabled_devices = {uta::DeviceType::GPU},
        .enable_profiling = true
    });
    
    const size_t size = 1024;
    void* ptr = context->allocate(size, uta::MemoryType::DEVICE);
    ASSERT_NE(ptr, nullptr);
    
    context->deallocate(ptr, uta::MemoryType::DEVICE);
}

TEST_F(ContextTest, ErrorHandling) {
    auto context = uta::Context::create({
        .enabled_devices = {uta::DeviceType::GPU},
        .enable_profiling = true
    });
    
    // Test invalid allocation
    EXPECT_THROW(
        context->allocate(SIZE_MAX, uta::MemoryType::DEVICE),
        std::runtime_error
    );
    
    //  Test invalid device
    EXPECT_THROW(
        context->getDevice(uta::DeviceType::GPU, 9999),
        std::runtime_error
    );
}

TEST_F(ContextTest, Synchronization) {
    auto context = uta::Context::create({
        .enabled_devices = {uta::DeviceType::GPU},
        .enable_profiling = true
    });
    
    auto device = context->getDevice(uta::DeviceType::GPU, 0);
    auto stream = device->createStream();
    
    // Create and execute some operations
    auto a = uta::Tensor::create({1024}, uta::DataType::FLOAT32, *device);
    auto b = uta::Tensor::create({1024}, uta::DataType::FLOAT32, *device);
    
    uta::ops::add(*a, *b);
    
    // Test synchronization
    EXPECT_NO_THROW(context->synchronize());
}
