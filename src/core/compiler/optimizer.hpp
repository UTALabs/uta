#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/Analysis/LoopInfo.h>

namespace uta {
namespace compiler {

// optimization level
enum class OptimizationLevel {
    O0,  // no optimization
    O1,  // basic optimization
    O2,  // medium optimization
    O3   // aggressive optimization
};

// Optimized channel interface
class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;
    virtual void runOnModule(llvm::Module* module) = 0;
    virtual void runOnFunction(llvm::Function* function) = 0;
    virtual std::string getName() const = 0;
};

// dead code elimination
class DeadCodeElimination : public OptimizationPass {
public:
    void runOnModule(llvm::Module* module) override;
    void runOnFunction(llvm::Function* function) override;
    std::string getName() const override { return "DeadCodeElimination"; }

private:
    void eliminateUnreachableCode(llvm::Function* function);
    void eliminateUnusedVariables(llvm::Function* function);
};

// Constant folding
class ConstantFolding : public OptimizationPass {
public:
    void runOnModule(llvm::Module* module) override;
    void runOnFunction(llvm::Function* function) override;
    std::string getName() const override { return "ConstantFolding"; }

private:
    llvm::Value* foldConstants(llvm::Instruction* inst);
    bool isConstantExpression(llvm::Value* value);
};

// loop optimization
class LoopOptimizations : public OptimizationPass {
public:
    void runOnModule(llvm::Module* module) override;
    void runOnFunction(llvm::Function* function) override;
    std::string getName() const override { return "LoopOptimizations"; }

    struct LoopOptConfig {
        bool enable_unrolling;
        bool enable_vectorization;
        bool enable_fusion;
        int unroll_factor;
        int vectorization_width;
    };

    void setConfig(const LoopOptConfig& config) { config_ = config; }

private:
    void optimizeLoop(llvm::Loop* loop);
    void unrollLoop(llvm::Loop* loop);
    void vectorizeLoop(llvm::Loop* loop);
    void fuseLoops(llvm::Loop* loop1, llvm::Loop* loop2);

    LoopOptConfig config_;
};

// memory optimization
class MemoryOptimizations : public OptimizationPass {
public:
    void runOnModule(llvm::Module* module) override;
    void runOnFunction(llvm::Function* function) override;
    std::string getName() const override { return "MemoryOptimizations"; }

    struct MemoryOptConfig {
        bool enable_load_store_opt;
        bool enable_memory_coalescing;
        bool enable_register_promotion;
    };

    void setConfig(const MemoryOptConfig& config) { config_ = config; }

private:
    void optimizeLoadStore(llvm::Function* function);
    void coalesceMemoryAccess(llvm::Function* function);
    void promoteToRegisters(llvm::Function* function);

    MemoryOptConfig config_;
};

// vectorized optimization
class VectorizationPass : public OptimizationPass {
public:
    void runOnModule(llvm::Module* module) override;
    void runOnFunction(llvm::Function* function) override;
    std::string getName() const override { return "Vectorization"; }

    struct VectorizationConfig {
        int min_vector_size;
        int max_vector_size;
        bool enable_slp;
        bool enable_loop_vectorization;
    };

    void setConfig(const VectorizationConfig& config) { config_ = config; }

private:
    void vectorizeBasicBlock(llvm::BasicBlock* bb);
    void vectorizeLoop(llvm::Loop* loop);
    bool isVectorizable(llvm::Value* value);

    VectorizationConfig config_;
};

// optimization manager
class OptimizationManager {
public:
    static OptimizationManager& getInstance();

    // initialization
    void initialize(OptimizationLevel level);

    // add optimization pass
    void addPass(std::unique_ptr<OptimizationPass> pass);

    // run optimization
    void runPasses(llvm::Module* module);
    void runPasses(llvm::Function* function);

    // optimization statistics
    struct OptimizationStats {
        double total_time;
        std::unordered_map<std::string, double> pass_times;
        std::unordered_map<std::string, size_t> improvements;
    };

    OptimizationStats getStats() const;

private:
    OptimizationManager() = default;

    // Optimize channel management
    std::vector<std::unique_ptr<OptimizationPass>> passes_;
    OptimizationLevel level_;
    OptimizationStats stats_;

    // inner function
    void runPassesInternal(llvm::Module* module);
    void runPassesInternal(llvm::Function* function);
    void collectStats(const std::string& pass_name, double time, size_t improvement);
};

} // namespace compiler
} // namespace uta
