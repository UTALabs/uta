#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace uta {
namespace autotuner {

struct Instruction {
    enum class Type {
        ARITHMETIC,    // Calculation operation
        MEMORY,       // Memory operation
        CONTROL,      // Control flow
        SYNC,         // Synchronization
        SPECIAL       // Special instruction
    };

    std::string opcode;
    Type type;
    std::vector<std::string> operands;
    size_t latency;
    size_t throughput;
};

struct DependencyGraph {
    struct Node {
        Instruction instruction;
        std::vector<size_t> predecessors;
        std::vector<size_t> successors;
        int earliest_start;
        int latest_start;
    };

    std::vector<Node> nodes;
    std::vector<std::vector<bool>> adjacency_matrix;
};

class InstructionScheduler {
public:
    static InstructionScheduler& getInstance();

    // Instruction scheduling optimization
    struct SchedulingResult {
        std::string optimized_ptx;
        float estimated_speedup;
        std::vector<std::string> optimization_steps;
    };

    SchedulingResult optimizeScheduling(const std::string& ptx_code);

    // Instruction Level Parallel Optimization
    struct ILPConfig {
        size_t max_instruction_distance;  // Maximum Command Rearrangement Distance
        bool allow_speculative_execution; // Is speculative execution allowed?
        size_t unroll_factor;            // loop unfolding factor
        bool enable_dual_issue;          // Whether to enable dual launch
    };

    void setILPConfig(const ILPConfig& config);

    // Register allocation optimization
    struct RegisterAllocation {
        std::unordered_map<std::string, int> register_mapping;
        size_t total_registers_used;
        bool spill_needed;
    };

    RegisterAllocation optimizeRegisterAllocation(
        const std::string& ptx_code,
        size_t max_registers
    );

private:
    InstructionScheduler() = default;

    // internal optimization algorithm
    DependencyGraph buildDependencyGraph(const std::string& ptx_code);
    std::vector<Instruction> listScheduling(const DependencyGraph& graph);
    std::vector<Instruction> softwarePipelining(const std::vector<Instruction>& instructions);
    
    // Instruction reordering optimization
    struct ReorderingStrategy {
        bool canReorder(const Instruction& i1, const Instruction& i2);
        float estimateBenefit(const Instruction& i1, const Instruction& i2);
    };

    std::string applyReordering(
        const std::string& ptx_code,
        const ReorderingStrategy& strategy
    );

    // Register pressure analysis
    struct RegisterPressure {
        size_t max_live_registers;
        std::vector<size_t> pressure_points;
        std::vector<std::string> spill_candidates;
    };

    RegisterPressure analyzeRegisterPressure(
        const std::vector<Instruction>& instructions
    );

    // optimization auxiliary function
    bool validateDataDependencies(const std::vector<Instruction>& schedule);
    float estimatePerformance(const std::vector<Instruction>& schedule);
    std::string generateOptimizedPTX(const std::vector<Instruction>& schedule);

    // architecture-specific optimization
    struct ArchitectureConstraints {
        size_t max_active_warps;
        size_t instruction_buffer_size;
        bool supports_predication;
        bool supports_dual_issue;
    };

    std::string applyArchitectureSpecificOptimizations(
        const std::string& ptx_code,
        const ArchitectureConstraints& constraints
    );

    // internal state
    ILPConfig ilp_config_;
    ArchitectureConstraints arch_constraints_;
};

} // namespace autotuner
} // namespace uta
