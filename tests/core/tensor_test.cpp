#include <gtest/gtest.h>
#include <uta/uta.hpp>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        uta::initialize();
        context_ = uta::Context::create({
            .enabled_devices = {uta::DeviceType::GPU},
            .enable_profiling = true
        });
        device_ = context_->getDevice(uta::DeviceType::GPU, 0);
    }

    void TearDown() override {
        uta::finalize();
    }

    std::shared_ptr<uta::Context> context_;
    std::shared_ptr<uta::Device> device_;
};

TEST_F(TensorTest, CreateTensor) {
    auto tensor = uta::Tensor::create({2, 3, 4}, uta::DataType::FLOAT32, *device_);
    ASSERT_NE(tensor, nullptr);
    
    auto shape = tensor->getShape();
    EXPECT_EQ(shape.size(), 3);
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 4);
    
    EXPECT_EQ(tensor->getDataType(), uta::DataType::FLOAT32);
}

TEST_F(TensorTest, DataAccess) {
    auto tensor = uta::Tensor::create({2, 2}, uta::DataType::FLOAT32, *device_);
    
    std::vector<float> host_data = {1.0f, 2.0f, 3.0f, 4.0f};
    tensor->copyFromHost(host_data.data());
    
    std::vector<float> result(4);
    tensor->copyToHost(result.data());
    
    for (size_t i = 0; i < host_data.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], host_data[i]);
    }
}

TEST_F(TensorTest, Reshape) {
    auto tensor = uta::Tensor::create({2, 3}, uta::DataType::FLOAT32, *device_);
    
    tensor->reshape({3, 2});
    auto shape = tensor->getShape();
    
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 2);
}

TEST_F(TensorTest, ElementwiseOperations) {
    auto a = uta::Tensor::create({2, 2}, uta::DataType::FLOAT32, *device_);
    auto b = uta::Tensor::create({2, 2}, uta::DataType::FLOAT32, *device_);
    
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b_data = {5.0f, 6.0f, 7.0f, 8.0f};
    
    a->copyFromHost(a_data.data());
    b->copyFromHost(b_data.data());
    
    // Test addition
    auto c = uta::ops::add(*a, *b);
    std::vector<float> result(4);
    c->copyToHost(result.data());
    
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], a_data[i] + b_data[i]);
    }
}

TEST_F(TensorTest, MatrixMultiplication) {
    auto a = uta::Tensor::create({2, 3}, uta::DataType::FLOAT32, *device_);
    auto b = uta::Tensor::create({3, 2}, uta::DataType::FLOAT32, *device_);
    
    std::vector<float> a_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> b_data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    
    a->copyFromHost(a_data.data());
    b->copyFromHost(b_data.data());
    
    auto c = uta::ops::matmul(*a, *b);
    std::vector<float> result(4);
    c->copyToHost(result.data());
    
    // Expected result: [[58, 64], [139, 154]]
    EXPECT_FLOAT_EQ(result[0], 58.0f);
    EXPECT_FLOAT_EQ(result[1], 64.0f);
    EXPECT_FLOAT_EQ(result[2], 139.0f);
    EXPECT_FLOAT_EQ(result[3], 154.0f);
}

TEST_F(TensorTest, ErrorHandling) {
    // Test invalid shape
    EXPECT_THROW(
        uta::Tensor::create({0, -1}, uta::DataType::FLOAT32, *device_),
        std::invalid_argument
    );
    
    // Test mismatched shapes in operations
    auto a = uta::Tensor::create({2, 2}, uta::DataType::FLOAT32, *device_);
    auto b = uta::Tensor::create({3, 3}, uta::DataType::FLOAT32, *device_);
    
    EXPECT_THROW(
        uta::ops::add(*a, *b),
        std::runtime_error
    );
    
    EXPECT_THROW(
        uta::ops::matmul(*a, *b),
        std::runtime_error
    );
}
