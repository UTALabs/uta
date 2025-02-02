#include <benchmark/benchmark.h>
#include <uta/uta.hpp>

class TensorBenchmark : public benchmark::Fixture {
protected:
    void SetUp(const benchmark::State& state) override {
        uta::initialize();
        context_ = uta::Context::create({
            .enabled_devices = {uta::DeviceType::GPU},
            .enable_profiling = true
        });
        device_ = context_->getDevice(uta::DeviceType::GPU, 0);
    }

    void TearDown(const benchmark::State& state) override {
        uta::finalize();
    }

    std::shared_ptr<uta::Context> context_;
    std::shared_ptr<uta::Device> device_;
};

BENCHMARK_DEFINE_F(TensorBenchmark, MatMul)(benchmark::State& state) {
    const int M = state.range(0);
    const int N = state.range(0);
    const int K = state.range(0);

    auto a = uta::Tensor::create({M, K}, uta::DataType::FLOAT32, *device_);
    auto b = uta::Tensor::create({K, N}, uta::DataType::FLOAT32, *device_);
    auto c = uta::Tensor::create({M, N}, uta::DataType::FLOAT32, *device_);

    // Initialize with random data
    std::vector<float> a_data(M * K);
    std::vector<float> b_data(K * N);
    for (int i = 0; i < M * K; ++i) a_data[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) b_data[i] = static_cast<float>(rand()) / RAND_MAX;

    a->copyFromHost(a_data.data());
    b->copyFromHost(b_data.data());

    for (auto _ : state) {
        uta::ops::matmul(*a, *b, *c);
        context_->synchronize();
    }

    // Report throughput
    state.SetBytesProcessed(int64_t(state.iterations()) * M * N * K * sizeof(float));
    state.SetItemsProcessed(int64_t(state.iterations()) * M * N * K);
}

BENCHMARK_REGISTER_F(TensorBenchmark, MatMul)
    ->RangeMultiplier(2)
    ->Range(128, 2048)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_DEFINE_F(TensorBenchmark, ElementwiseAdd)(benchmark::State& state) {
    const int size = state.range(0);

    auto a = uta::Tensor::create({size}, uta::DataType::FLOAT32, *device_);
    auto b = uta::Tensor::create({size}, uta::DataType::FLOAT32, *device_);
    auto c = uta::Tensor::create({size}, uta::DataType::FLOAT32, *device_);

    std::vector<float> a_data(size);
    std::vector<float> b_data(size);
    for (int i = 0; i < size; ++i) {
        a_data[i] = static_cast<float>(rand()) / RAND_MAX;
        b_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    a->copyFromHost(a_data.data());
    b->copyFromHost(b_data.data());

    for (auto _ : state) {
        uta::ops::add(*a, *b, *c);
        context_->synchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * size * sizeof(float) * 3);
    state.SetItemsProcessed(int64_t(state.iterations()) * size);
}

BENCHMARK_REGISTER_F(TensorBenchmark, ElementwiseAdd)
    ->RangeMultiplier(2)
    ->Range(1<<20, 1<<24)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_DEFINE_F(TensorBenchmark, MemoryBandwidth)(benchmark::State& state) {
    const int size = state.range(0);

    auto src = uta::Tensor::create({size}, uta::DataType::FLOAT32, *device_);
    auto dst = uta::Tensor::create({size}, uta::DataType::FLOAT32, *device_);

    std::vector<float> host_data(size);
    for (int i = 0; i < size; ++i) {
        host_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    src->copyFromHost(host_data.data());

    for (auto _ : state) {
        dst->copyFrom(*src);
        context_->synchronize();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * size * sizeof(float) * 2);
    state.SetItemsProcessed(int64_t(state.iterations()) * size);
}

BENCHMARK_REGISTER_F(TensorBenchmark, MemoryBandwidth)
    ->RangeMultiplier(2)
    ->Range(1<<20, 1<<24)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_MAIN();
