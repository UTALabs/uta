#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <llvm/IR/Module.h>

namespace uta {
namespace compiler {

// link option
struct LinkOptions {
    bool strip_debug_info;        // Whether to remove debugging information
    bool enable_lto;             // Whether to enable link-time optimization
    bool create_shared_lib;      // Whether to create a shared library
    std::string output_name;     // Output file name
};

// Symbol visibility
enum class SymbolVisibility {
    DEFAULT,
    HIDDEN,
    PROTECTED,
    INTERNAL
};

// Symbolic Information
struct SymbolInfo {
    std::string name;
    SymbolVisibility visibility;
    bool is_exported;
    bool is_weak;
    std::string version;
};

// linker
class Linker {
public:
    static Linker& getInstance();

    // initialization
    void initialize(const LinkOptions& options);

    // linking interface
    void addModule(std::unique_ptr<llvm::Module> module);
    void addObject(const std::string& object_file);
    void addLibrary(const std::string& library_path);

    // linking optimization
    void optimizeLink();
    
    // symbol management
    void addSymbol(const SymbolInfo& symbol);
    void setSymbolVisibility(const std::string& name, SymbolVisibility visibility);
    
    // execute linking
    bool link(const std::string& output_file);

    // linking statistics
    struct LinkStats {
        size_t num_modules;
        size_t num_symbols;
        double link_time;
        double optimization_time;
        size_t output_size;
    };

    LinkStats getStats() const;

private:
    Linker() = default;

    // internal state
    LinkOptions options_;
    std::vector<std::unique_ptr<llvm::Module>> modules_;
    std::unordered_map<std::string, SymbolInfo> symbols_;
    LinkStats stats_;

    // internal functions
    void performLTO();
    void resolveSymbols();
    void optimizeGlobally();
    void generateOutput(const std::string& output_file);
};

// Link-time Optimization
class LTOptimizer {
public:
    static LTOptimizer& getInstance();

    // Link-time Optimization Configuration
    struct LTOConfig {
        bool enable_internalize;      // Internalize unexported symbols
        bool enable_global_dce;       // Global Dead Code Elimination
        bool enable_global_opt;       // Global Optimization
        int optimization_level;       // Optimization level (O0-O3)
    };

    void initialize(const LTOConfig& config);

    // Optimization Interface
    void optimize(std::vector<llvm::Module*>& modules);

    // Optimization Statistics
    struct OptimizationStats {
        double optimization_time;
        size_t removed_functions;
        size_t removed_globals;
        size_t size_reduction;
    };

    OptimizationStats getStats() const;

private:
    LTOptimizer() = default;

    // internal optimization
    void internalizeSymbols(llvm::Module* module);
    void globalDCE(std::vector<llvm::Module*>& modules);
    void globalOptimizations(std::vector<llvm::Module*>& modules);

    // configuration and statistics
    LTOConfig config_;
    OptimizationStats stats_;
};

} // namespace compiler
} // namespace uta
