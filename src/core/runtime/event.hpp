#pragma once

#include <memory>
#include <string>
#include <functional>
#include <chrono>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace uta {
namespace runtime {

// event type
enum class EventType {
    TASK_START,
    TASK_COMPLETE,
    TASK_FAILED,
    MEMORY_ALLOCATED,
    MEMORY_FREED,
    DEVICE_SYNC,
    COMMUNICATION_START,
    COMMUNICATION_COMPLETE,
    ERROR,
    CUSTOM
};

// event priority
enum class EventPriority {
    CRITICAL,
    HIGH,
    NORMAL,
    LOW
};

// event base class
class Event {
public:
    Event(EventType type, EventPriority priority = EventPriority::NORMAL)
        : type_(type)
        , priority_(priority)
        , timestamp_(std::chrono::steady_clock::now())
    {}

    virtual ~Event() = default;

    EventType getType() const { return type_; }
    EventPriority getPriority() const { return priority_; }
    std::chrono::steady_clock::time_point getTimestamp() const { return timestamp_; }

    virtual std::string toString() const = 0;

protected:
    EventType type_;
    EventPriority priority_;
    std::chrono::steady_clock::time_point timestamp_;
};

// task event
class TaskEvent : public Event {
public:
    TaskEvent(EventType type, std::string task_name, 
             std::string task_id, EventPriority priority = EventPriority::NORMAL)
        : Event(type, priority)
        , task_name_(std::move(task_name))
        , task_id_(std::move(task_id))
    {}

    const std::string& getTaskName() const { return task_name_; }
    const std::string& getTaskId() const { return task_id_; }

    std::string toString() const override {
        return "TaskEvent: " + task_name_ + " (" + task_id_ + ")";
    }

private:
    std::string task_name_;
    std::string task_id_;
};

// memory event
class MemoryEvent : public Event {
public:
    MemoryEvent(EventType type, size_t size, int device_id,
                EventPriority priority = EventPriority::NORMAL)
        : Event(type, priority)
        , size_(size)
        , device_id_(device_id)
    {}

    size_t getSize() const { return size_; }
    int getDeviceId() const { return device_id_; }

    std::string toString() const override {
        return "MemoryEvent: " + std::to_string(size_) + " bytes on device " + 
               std::to_string(device_id_);
    }

private:
    size_t size_;
    int device_id_;
};

// error event
class ErrorEvent : public Event {
public:
    ErrorEvent(std::string error_message, std::string stack_trace = "",
               EventPriority priority = EventPriority::CRITICAL)
        : Event(EventType::ERROR, priority)
        , error_message_(std::move(error_message))
        , stack_trace_(std::move(stack_trace))
    {}

    const std::string& getErrorMessage() const { return error_message_; }
    const std::string& getStackTrace() const { return stack_trace_; }

    std::string toString() const override {
        return "ErrorEvent: " + error_message_;
    }

private:
    std::string error_message_;
    std::string stack_trace_;
};

// Event Listener Interface
class EventListener {
public:
    virtual ~EventListener() = default;
    virtual void onEvent(const Event& event) = 0;
};

// Event Manager
class EventManager {
public:
    static EventManager& getInstance();

    // Event Registration
    void registerListener(EventType type, std::shared_ptr<EventListener> listener);
    void unregisterListener(EventType type, std::shared_ptr<EventListener> listener);

    // Event Dispatch
    void dispatchEvent(const Event& event);

    // Event Filtering
    void setPriorityFilter(EventPriority min_priority);
    void addTypeFilter(EventType type);
    void removeTypeFilter(EventType type);

    // Event Query
    std::vector<std::shared_ptr<Event>> getEventHistory(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end
    );

private:
    EventManager() = default;

    struct ListenerEntry {
        std::weak_ptr<EventListener> listener;
        EventPriority priority;
    };

    std::unordered_map<EventType, std::vector<ListenerEntry>> listeners_;
    std::vector<std::shared_ptr<Event>> event_history_;
    EventPriority min_priority_{EventPriority::LOW};
    std::vector<EventType> filtered_types_;
    
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

} // namespace runtime
} // namespace uta
