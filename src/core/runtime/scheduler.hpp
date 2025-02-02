#pragma once

#include <memory>
#include <queue>
#include <vector>
#include <functional>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace uta {
namespace runtime {

// forward statement
class Task;
class TaskQueue;
class ExecutionContext;

// task priority
enum class TaskPriority {
    HIGH,
    NORMAL,
    LOW,
    BACKGROUND
};

// task status
enum class TaskStatus {
    PENDING,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED
};

// Task dependence graph
class TaskGraph {
public:
    void addTask(std::shared_ptr<Task> task);
    void addDependency(std::shared_ptr<Task> dependent, std::shared_ptr<Task> dependency);
    std::vector<std::shared_ptr<Task>> getReadyTasks();
    bool hasUnfinishedTasks() const;

private:
    struct Node {
        std::shared_ptr<Task> task;
        std::vector<std::shared_ptr<Task>> dependencies;
        std::vector<std::shared_ptr<Task>> dependents;
    };
    std::unordered_map<std::shared_ptr<Task>, Node> graph_;
    std::mutex graph_mutex_;
};

// scheduling policy
class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::shared_ptr<Task> selectNext(const TaskQueue& queue) = 0;
    virtual void onTaskComplete(std::shared_ptr<Task> task) = 0;
};

// priority scheduling policy
class PriorityScheduler : public SchedulingPolicy {
public:
    std::shared_ptr<Task> selectNext(const TaskQueue& queue) override;
    void onTaskComplete(std::shared_ptr<Task> task) override;
};

// fair scheduling policy
class FairScheduler : public SchedulingPolicy {
public:
    std::shared_ptr<Task> selectNext(const TaskQueue& queue) override;
    void onTaskComplete(std::shared_ptr<Task> task) override;
private:
    std::unordered_map<std::string, size_t> task_counts_;
    std::mutex counts_mutex_;
};

// task scheduler
class Scheduler {
public:
    static Scheduler& getInstance();

    // initialization
    void initialize(size_t num_threads);
    void shutdown();

    // task submission
    template<typename F, typename... Args>
    auto submitTask(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;

    // batch task submission
    void submitTaskGraph(const TaskGraph& graph);

    // scheduling policy configuration
    void setSchedulingPolicy(std::unique_ptr<SchedulingPolicy> policy);

    // performance monitoring
    struct PerformanceMetrics {
        size_t tasks_completed;
        size_t tasks_failed;
        double average_wait_time;
        double average_execution_time;
    };

    PerformanceMetrics getMetrics() const;

private:
    Scheduler() = default;
    
    // Worker thread function
    void workerThread();
    
    // Task execution
    void executeTask(std::shared_ptr<Task> task);
    
    // Internal state
    std::unique_ptr<TaskQueue> task_queue_;
    std::unique_ptr<SchedulingPolicy> scheduling_policy_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    // performance monitoring
    struct Metrics {
        std::atomic<size_t> completed_tasks{0};
        std::atomic<size_t> failed_tasks{0};
        std::vector<double> wait_times;
        std::vector<double> execution_times;
        mutable std::mutex metrics_mutex;
    };
    
    Metrics metrics_;
};

// task queue
class TaskQueue {
public:
    void push(std::shared_ptr<Task> task);
    std::shared_ptr<Task> pop();
    bool empty() const;
    size_t size() const;

private:
    std::priority_queue<
        std::shared_ptr<Task>,
        std::vector<std::shared_ptr<Task>>,
        std::function<bool(const std::shared_ptr<Task>&, const std::shared_ptr<Task>&)>
    > queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
};

// execution context
class ExecutionContext {
public:
    void* allocateMemory(size_t size);
    void freeMemory(void* ptr);
    void setDevice(int device_id);
    void synchronize();

private:
    int current_device_{0};
    std::unordered_map<void*, size_t> allocations_;
    std::mutex context_mutex_;
};

} // namespace runtime
} // namespace uta
