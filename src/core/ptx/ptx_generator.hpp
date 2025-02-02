#pragma once

#include <string>
#include <vector>
#include <memory>

namespace uta {
namespace ptx {

enum class DataType {
    FP16,
    FP32,
    FP64,
    INT8,
    INT32,
    INT64
};

enum class Operation {
    MATMUL,
    CONV2D,
    RELU,
    SOFTMAX,
    LAYERNORM,
    ATTENTION
};

class PTXGenerator {
public:
    static PTXGenerator& getInstance();

    // Generate PTX code for specific operation
    std::string generatePTX(Operation op, 
                           const std::vector<DataType>& inputTypes,
                           const std::vector<size_t>& shapes,
                           const std::string& targetArch);

    // Compile PTX to CUBIN
    std::vector<uint8_t> compileToCUBIN(const std::string& ptx,
                                       const std::string& targetArch);

private:
    PTXGenerator() = default;

    // Template generators for different operations
    std::string generateMatMul(const std::vector<DataType>& types,
                              const std::vector<size_t>& shapes);
    std::string generateConv2D(const std::vector<DataType>& types,
                              const std::vector<size_t>& shapes);
    std::string generateActivation(Operation op,
                                 const DataType& type);
    
    // Helper functions
    std::string getDataTypeString(DataType type);
    std::string generateHeader(const std::string& targetArch);
    std::string generateRegisters(const std::vector<DataType>& types);
};

} // namespace ptx
} // namespace uta
