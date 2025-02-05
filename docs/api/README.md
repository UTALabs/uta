# UTA API Reference

This document provides detailed information about the UTA API.

## Table of Contents

1. [Core API](#core-api)
2. [Operations API](#operations-api)
3. [Profiler API](#profiler-api)
4. [Distributed API](#distributed-api)

## Core API

### Context

The Context class manages the global state and resources of UTA.

```cpp
// Create a new context 
auto context = uta::Context::create({
    .enabled_devices = {
        uta::DeviceType::GPU,  // Any PTX-compatible GPU
        uta::DeviceType::CPU   // Fallback option
    },
    .enable_profiling = true,
    .enable_debug = false,
    .memory_pool_size = 1024 * 1024 * 1024  // 1GB
});

// Get available devices
auto devices = context->getDevices();

// Allocate memory
void* ptr = context->allocate(1024, uta::MemoryType::DEVICE);

// Deallocate memory
context->deallocate(ptr, uta::MemoryType::DEVICE);

// Synchronize all operations
context->synchronize();
```

### Device

The Device class represents a physical computing device (CPU/GPU).

```cpp
// Get a specific device (automatically selects best available)
auto device = context->getDevice(uta::DeviceType::GPU, 0);

// Create a new stream
auto stream = device->createStream();

// Create an event
auto event = device->createEvent();

// Enable peer access
device->enablePeerAccess(peer_device);

// Synchronize device
device->synchronize();
```

### Tensor

The Tensor class is the basic data structure for storing and manipulating data.

```cpp
// Create a new tensor
auto tensor = uta::Tensor::create({1024, 1024}, uta::DataType::FLOAT32, device);

// Access data
float* data = tensor->data<float>();

// Get tensor information
auto shape = tensor->getShape();
auto dtype = tensor->getDataType();
auto size = tensor->getSize();

// Copy data
tensor->copyTo(dst_tensor);
tensor->copyFrom(src_tensor);

// Initialize data
tensor->zero();
tensor->fill(&value);
```

## Operations API

### Basic Math Operations

```cpp
// Element-wise operations
auto c = uta::ops::add(a, b);
auto c = uta::ops::subtract(a, b);
auto c = uta::ops::multiply(a, b);
auto c = uta::ops::divide(a, b);

// Matrix operations
auto c = uta::ops::matmul(a, b);
auto b = uta::ops::transpose(a);
auto b = uta::ops::inverse(a);
```

### Neural Network Operations

```cpp
// Normalization
auto output = uta::ops::batchNorm(input, scale, bias, 1e-5);
auto output = uta::ops::layerNorm(input, {256}, scale, bias);

// Activation functions
auto output = uta::ops::relu(input);
auto output = uta::ops::sigmoid(input);
auto output = uta::ops::gelu(input);

// Attention mechanism
auto output = uta::ops::multiHeadAttention(query, key, value, {
    .num_heads = 8,
    .dropout_prob = 0.1,
    .causal = true
});

// Convolution
auto output = uta::ops::convolution(input, weight, bias, {
    .kernel_size = {3, 3},
    .stride = {1, 1},
    .padding = {1, 1}
});
```

## Profiler API

### Performance Profiling

```cpp
// Configure profiler
uta::profiler::Profiler::getInstance().configure({
    .enabled = true,
    .record_shapes = true,
    .record_memory = true,
    .record_bandwidth = true
});

// Start profiling
uta::profiler::Profiler::getInstance().start();

// Record operation
{
    UTA_PROFILE_SCOPE("MatMul");
    uta::ops::matmul(a, b, c);
}

// Stop profiling
uta::profiler::Profiler::getInstance().stop();

// Generate report
uta::profiler::Profiler::getInstance().generateReport("profile.json");
```

## Distributed API

### Process Group

```cpp
// Create process group
auto group = uta::distributed::ProcessGroup::create({
    .world_size = 4,
    .rank = 0,
    .backend = "ptx"  // Uses PTX-based communication
});

// Collective operations
group->broadcast(tensor, 0);
group->allReduce(tensor, "sum");
group->allGather(tensor, output_tensors);

// Point-to-point communication
group->send(tensor, dst_rank);
group->receive(tensor, src_rank);
```

### Distributed Training

```cpp
// Data parallel training
auto dp = uta::distributed::DataParallel::create(devices);
dp->forward([&](auto& device) {
    // Forward pass on each device
});
dp->backward([&](auto& device) {
    // Backward pass on each device
});
dp->synchronizeGradients();

// Model parallel training
auto mp = uta::distributed::ModelParallel::create({
    .pipeline_stages = 4,
    .tensor_parallel_size = 2
});
mp->partition([&](int stage) {
    // Define computation for each stage
});
mp->forward(inputs);
mp->backward(grad_outputs);
```

For more detailed information about each API component, please refer to:
- [Context and Device Management](context.md)
- [Tensor Operations](tensor.md)
- [Neural Network Operations](nn.md)
- [Profiling and Analysis](profiler.md)
- [Distributed Training](distributed.md)
