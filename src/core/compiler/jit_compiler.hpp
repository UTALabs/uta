#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>

namespace uta {
namespace compiler {

// compilation options
struct CompileOptions {
    bool optimize_level;           // Optimization level (O0-O3)
    bool enable_fast_math;        // Enable Fast Math
    bool enable_loop_unroll;      // Enable loop unrolling
    bool enable_vectorization;    // Enable vectorization
    bool enable_inline;          // Enable inlining
};

// compilation cache item
struct CacheEntry {
    std::vector<uint8_t> code;
    size_t size;
    std::string hash;
    std::chrono::steady_clock::time_point timestamp;
};

// JIT compiler
class JITCompiler {
public:
    static JITCompiler& getInstance();

    // initialization
    void initialize(const CompileOptions& options);

    // compilation interface
    void* compileFunction(
        const std::string& name,
        const std::string& source,
        const std::string& target_arch
    );

    // optimization interface
    void optimizeModule(llvm::Module* module);
    void optimizeFunction(llvm::Function* function);

    // cache management
    void enableCache(bool enable);
    void clearCache();
    void setCacheSize(size_t max_size);

    // performance analysis
    struct CompileStats {
        double compile_time;
        double optimization_time;
        size_t code_size;
        size_t cache_hits;
        size_t cache_misses;
    };

    CompileStats getStats() const;

private:
    JITCompiler() = default;

    // LLVM context
    std::unique_ptr<llvm::LLVMContext> context_;
    std::unique_ptr<llvm::IRBuilder<>> builder_;

    // compilation cache
    std::unordered_map<std::string, CacheEntry> cache_;
    size_t max_cache_size_;
    bool cache_enabled_;

    // compilation options
    CompileOptions options_;

    // performance statistics
    CompileStats stats_;

    // internal functions
    std::string generateHash(const std::string& source);
    void updateCache(const std::string& key, const CacheEntry& entry);
    void evictCache();
};

// code generator base class
class CodeGenerator {
public:
    virtual ~CodeGenerator() = default;
    virtual llvm::Value* generateCode(llvm::IRBuilder<>& builder) = 0;
    virtual std::string getSourceCode() const = 0;
};

// PTX Code Generator
class PTXGenerator : public CodeGenerator {
public:
    PTXGenerator(const std::string& kernel_name);

    llvm::Value* generateCode(llvm::IRBuilder<>& builder) override;
    std::string getSourceCode() const override;

    // PTX specific interface
    void addParameter(const std::string& type, const std::string& name);
    void addSharedMemory(const std::string& type, const std::string& name, size_t size);
    void addRegister(const std::string& type, const std::string& name);
    void addInstruction(const std::string& instruction);

private:
    std::string kernel_name_;
    std::vector<std::pair<std::string, std::string>> parameters_;
    std::vector<std::tuple<std::string, std::string, size_t>> shared_memory_;
    std::vector<std::pair<std::string, std::string>> registers_;
    std::vector<std::string> instructions_;
};

// CUDA Code Generator
class CUDAGenerator : public CodeGenerator {
public:
    CUDAGenerator(const std::string& kernel_name);

    llvm::Value* generateCode(llvm::IRBuilder<>& builder) override;
    std::string getSourceCode() const override;

    // CUDA specific interface
    void setGridDim(int x, int y = 1, int z = 1);
    void setBlockDim(int x, int y = 1, int z = 1);
    void addDeviceFunction(const std::string& function);
    void addKernelCode(const std::string& code);

private:
    std::string kernel_name_;
    struct {
        int x, y, z;
    } grid_dim_, block_dim_;
    std::vector<std::string> device_functions_;
    std::string kernel_code_;
};

// Loop Optimizer Interface
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void optimize(llvm::Module* module) = 0;
    virtual void optimize(llvm::Function* function) = 0;
};

// loop optimizer
class LoopOptimizer : public Optimizer {
public:
    void optimize(llvm::Module* module) override;
    void optimize(llvm::Function* function) override;

private:
    void unrollLoops(llvm::Loop* loop);
    void vectorizeLoop(llvm::Loop* loop);
    void parallelizeLoop(llvm::Loop* loop);
};

// Internal connection optimizer
class InlineOptimizer : public Optimizer {
public:
    void optimize(llvm::Module* module) override;
    void optimize(llvm::Function* function) override;

private:
    bool shouldInline(llvm::Function* function);
    void inlineFunction(llvm::Function* function);
};

} // namespace compiler
} // namespace uta
