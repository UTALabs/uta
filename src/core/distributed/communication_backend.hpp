#pragma once

#include <memory>
#include <string>
#include <vector>
#include "communicator.hpp"

namespace uta {
namespace distributed {

class CommunicationBackend {
public:
    // Communication backend type
    enum class BackendType {
        NCCL,
        MPI,
        GLOO,
        CUSTOM
    };

    struct BackendConfig {
        BackendType type;
        bool enable_rdma;           // Enable RDMA
        bool enable_gpu_direct;     // Enable GPU Direct
        size_t buffer_size;         // Communication buffer size
        std::string network_interface;  // Network interface
    };

    // Initialize backend
    static std::unique_ptr<CommunicationBackend> create(
        const BackendConfig& config
    );

    // Communication primitives
    virtual void allReduce(
        void* buffer,
        size_t count,
        const std::string& dtype,
        const std::string& reduction_op
    ) = 0;

    virtual void broadcast(
        void* buffer,
        size_t count,
        const std::string& dtype,
        int root_rank
    ) = 0;

    virtual void send(
        const void* buffer,
        size_t count,
        const std::string& dtype,
        int destination
    ) = 0;

    virtual void receive(
        void* buffer,
        size_t count,
        const std::string& dtype,
        int source
    ) = 0;

    // advanced features
    virtual bool supportsFusion() const = 0;
    virtual bool supportsGPUDirect() const = 0;
    virtual bool supportsRDMA() const = 0;

protected:
    virtual ~CommunicationBackend() = default;
};

// NCCL backend implementation
class NCCLBackend : public CommunicationBackend {
public:
    explicit NCCLBackend(const BackendConfig& config);

    void allReduce(
        void* buffer,
        size_t count,
        const std::string& dtype,
        const std::string& reduction_op
    ) override;

    void broadcast(
        void* buffer,
        size_t count,
        const std::string& dtype,
        int root_rank
    ) override;

    void send(
        const void* buffer,
        size_t count,
        const std::string& dtype,
        int destination
    ) override;

    void receive(
        void* buffer,
        size_t count,
        const std::string& dtype,
        int source
    ) override;

    bool supportsFusion() const override { return true; }
    bool supportsGPUDirect() const override { return true; }
    bool supportsRDMA() const override { return true; }

private:
    struct NCCLState;
    std::unique_ptr<NCCLState> state_;
};

// MPI backend implementation
class MPIBackend : public CommunicationBackend {
public:
    explicit MPIBackend(const BackendConfig& config);

    void allReduce(
        void* buffer,
        size_t count,
        const std::string& dtype,
        const std::string& reduction_op
    ) override;

    void broadcast(
        void* buffer,
        size_t count,
        const std::string& dtype,
        int root_rank
    ) override;

    void send(
        const void* buffer,
        size_t count,
        const std::string& dtype,
        int destination
    ) override;

    void receive(
        void* buffer,
        size_t count,
        const std::string& dtype,
        int source
    ) override;

    bool supportsFusion() const override { return false; }
    bool supportsGPUDirect() const override { return false; }
    bool supportsRDMA() const override { return true; }

private:
    struct MPIState;
    std::unique_ptr<MPIState> state_;
};

} // namespace distributed
} // namespace uta
