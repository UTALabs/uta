#pragma once

#include "uta.hpp"
#include <functional>

namespace uta {
namespace ops {

//  Basic mathematical operations
std::shared_ptr<Tensor> add(const Tensor& a, const Tensor& b);
std::shared_ptr<Tensor> subtract(const Tensor& a, const Tensor& b);
std::shared_ptr<Tensor> multiply(const Tensor& a, const Tensor& b);
std::shared_ptr<Tensor> divide(const Tensor& a, const Tensor& b);

// Matrix operation
std::shared_ptr<Tensor> matmul(const Tensor& a, const Tensor& b);
std::shared_ptr<Tensor> transpose(const Tensor& input);
std::shared_ptr<Tensor> inverse(const Tensor& input);
std::shared_ptr<Tensor> solve(const Tensor& a, const Tensor& b);

// normalized operation
std::shared_ptr<Tensor> batchNorm(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& bias,
    float epsilon = 1e-5
);

std::shared_ptr<Tensor> layerNorm(
    const Tensor& input,
    const std::vector<int>& normalized_shape,
    const Tensor& scale,
    const Tensor& bias,
    float epsilon = 1e-5
);

// activation function
std::shared_ptr<Tensor> relu(const Tensor& input);
std::shared_ptr<Tensor> sigmoid(const Tensor& input);
std::shared_ptr<Tensor> tanh(const Tensor& input);
std::shared_ptr<Tensor> gelu(const Tensor& input);

// attention mechanism
struct AttentionConfig {
    size_t num_heads;
    float dropout_prob;
    bool use_bias;
    bool causal;
};

std::shared_ptr<Tensor> multiHeadAttention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const AttentionConfig& config
);

// convolutional operation
struct ConvConfig {
    std::vector<size_t> kernel_size;
    std::vector<size_t> stride;
    std::vector<size_t> padding;
    std::vector<size_t> dilation;
    size_t groups;
};

std::shared_ptr<Tensor> convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const ConvConfig& config
);

// pooling operation
struct PoolConfig {
    std::vector<size_t> kernel_size;
    std::vector<size_t> stride;
    std::vector<size_t> padding;
};

std::shared_ptr<Tensor> maxPool(
    const Tensor& input,
    const PoolConfig& config
);

std::shared_ptr<Tensor> avgPool(
    const Tensor& input,
    const PoolConfig& config
);

// downsampling and upsampling
std::shared_ptr<Tensor> interpolate(
    const Tensor& input,
    const std::vector<size_t>& size,
    const std::string& mode = "linear"
);

// loss function
std::shared_ptr<Tensor> crossEntropy(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight = nullptr
);

std::shared_ptr<Tensor> mseLoss(
    const Tensor& input,
    const Tensor& target
);

// optimizer operation
void sgd(
    Tensor& param,
    const Tensor& grad,
    float learning_rate,
    float momentum = 0.0,
    float weight_decay = 0.0
);

void adam(
    Tensor& param,
    Tensor& m,
    Tensor& v,
    const Tensor& grad,
    float learning_rate,
    float beta1 = 0.9,
    float beta2 = 0.999,
    float epsilon = 1e-8
);

// custom action
using CustomOp = std::function<std::shared_ptr<Tensor>(
    const std::vector<std::shared_ptr<Tensor>>& inputs
)>;

std::shared_ptr<Tensor> customOp(
    const std::vector<std::shared_ptr<Tensor>>& inputs,
    const CustomOp& op
);

} // namespace ops
} // namespace uta
