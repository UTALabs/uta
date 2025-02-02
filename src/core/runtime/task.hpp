#pragma once

#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <any>
#include <vector>
#include "scheduler.hpp"

namespace uta {
namespace runtime {

// task base class
class Task {
public:
    using TaskFunction = std::function<void(ExecutionContext&)>;
    
    Task(std::string name, TaskFunction func, TaskPriority priority = TaskPriority::NORMAL)
        : name_(std::move(name))
        , function_(std::move(func))
        , priority_(priority)
        , status_(TaskStatus::PENDING)
    {}

    virtual ~Task() = default;

    // task execution
    virtual void execute(ExecutionContext& context) {
        start_time_ = std::chrono::steady_clock::now();
        status_ = TaskStatus::RUNNING;
        
        try {
            function_(context);
            status_ = TaskStatus::COMPLETED;
        } catch (...) {
            status_ = TaskStatus::FAILED;
            throw;
        }
        
        end_time_ = std::chrono::steady_clock::now();
    }

    // task control
    virtual void cancel() {
        status_ = TaskStatus::CANCELLED;
    }

    virtual bool isCancellable() const {
        return true;
    }

    // attribute access
    const std::string& getName() const { return name_; }
    TaskPriority getPriority() const { return priority_; }
    TaskStatus getStatus() const { return status_; }
    
    // time statistics
    std::chrono::duration<double> getExecutionTime() const {
        if (start_time_ == std::chrono::steady_clock::time_point() ||
            end_time_ == std::chrono::steady_clock::time_point()) {
            return std::chrono::duration<double>::zero();
        }
        return end_time_ - start_time_;
    }

protected:
    std::string name_;
    TaskFunction function_;
    TaskPriority priority_;
    TaskStatus status_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point end_time_;
};

// computing task
class ComputeTask : public Task {
public:
    ComputeTask(std::string name, TaskFunction func, 
                int device_id, size_t memory_requirement,
                TaskPriority priority = TaskPriority::NORMAL)
        : Task(std::move(name), std::move(func), priority)
        , device_id_(device_id)
        , memory_requirement_(memory_requirement)
    {}

    void execute(ExecutionContext& context) override {
        context.setDevice(device_id_);
        Task::execute(context);
    }

    int getDeviceId() const { return device_id_; }
    size_t getMemoryRequirement() const { return memory_requirement_; }

private:
    int device_id_;
    size_t memory_requirement_;
};

// communication task
class CommunicationTask : public Task {
public:
    CommunicationTask(std::string name, TaskFunction func,
                     std::vector<int> involved_devices,
                     size_t data_size,
                     TaskPriority priority = TaskPriority::NORMAL)
        : Task(std::move(name), std::move(func), priority)
        , involved_devices_(std::move(involved_devices))
        , data_size_(data_size)
    {}

    const std::vector<int>& getInvolvedDevices() const { return involved_devices_; }
    size_t getDataSize() const { return data_size_; }

private:
    std::vector<int> involved_devices_;
    size_t data_size_;
};

// memory transfer task
class MemoryTransferTask : public Task {
public:
    MemoryTransferTask(std::string name, TaskFunction func,
                      int source_device, int target_device,
                      size_t data_size,
                      TaskPriority priority = TaskPriority::NORMAL)
        : Task(std::move(name), std::move(func), priority)
        , source_device_(source_device)
        , target_device_(target_device)
        , data_size_(data_size)
    {}

    int getSourceDevice() const { return source_device_; }
    int getTargetDevice() const { return target_device_; }
    size_t getDataSize() const { return data_size_; }

private:
    int source_device_;
    int target_device_;
    size_t data_size_;
};

// synchronization task
class SynchronizationTask : public Task {
public:
    SynchronizationTask(std::string name, TaskFunction func,
                       std::vector<std::shared_ptr<Task>> dependencies,
                       TaskPriority priority = TaskPriority::HIGH)
        : Task(std::move(name), std::move(func), priority)
        , dependencies_(std::move(dependencies))
    {}

    bool isCancellable() const override {
        return false;
    }

    const std::vector<std::shared_ptr<Task>>& getDependencies() const {
        return dependencies_;
    }

private:
    std::vector<std::shared_ptr<Task>> dependencies_;
};

} // namespace runtime
} // namespace uta
