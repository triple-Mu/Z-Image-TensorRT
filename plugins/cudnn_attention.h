#pragma once

#include <NvInfer.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <mutex>
#include <string>
#include <vector>

using NANO_SECOND = std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds>;

std::string costStr(NANO_SECOND a, NANO_SECOND b)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(5)
       << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count()) / 1000.f;
    return ss.str();
}

std::string dataTypeToString(nvinfer1::DataType dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32   ";
        case nvinfer1::DataType::kHALF:
            return "FP16   ";
        case nvinfer1::DataType::kINT8:
            return "INT8   ";
        case nvinfer1::DataType::kINT32:
            return "INT32  ";
        case nvinfer1::DataType::kBOOL:
            return "BOOL   ";
        case nvinfer1::DataType::kUINT8:
            return "UINT8  ";
        case nvinfer1::DataType::kFP8:
            return "FP8    ";
        case nvinfer1::DataType::kBF16:
            return "BF16   ";
        case nvinfer1::DataType::kINT64:
            return "INT64  ";
        case nvinfer1::DataType::kINT4:
            return "INT4   ";
        case nvinfer1::DataType::kFP4:
            return "kFP4   ";
        default:
            return "Unknown";
    }
}

cudnn_frontend::DataType_t dataTypeToFeType(nvinfer1::DataType dataType)
{
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return cudnn_frontend::DataType_t::FLOAT;
        case nvinfer1::DataType::kHALF:
            return cudnn_frontend::DataType_t::HALF;
        case nvinfer1::DataType::kINT8:
            return cudnn_frontend::DataType_t::INT8;
        case nvinfer1::DataType::kINT32:
            return cudnn_frontend::DataType_t::INT32;
        case nvinfer1::DataType::kBOOL:
            return cudnn_frontend::DataType_t::BOOLEAN;
        case nvinfer1::DataType::kUINT8:
            return cudnn_frontend::DataType_t::UINT8;
        case nvinfer1::DataType::kFP8:
            return cudnn_frontend::DataType_t::FP8_E4M3;
        case nvinfer1::DataType::kBF16:
            return cudnn_frontend::DataType_t::BFLOAT16;
        case nvinfer1::DataType::kINT64:
            return cudnn_frontend::DataType_t::INT64;
        case nvinfer1::DataType::kINT4:
            return cudnn_frontend::DataType_t::INT4;
        case nvinfer1::DataType::kFP4:
            return cudnn_frontend::DataType_t::FP4_E2M1;
        default:
            return cudnn_frontend::DataType_t::NOT_SET;
    }
}

std::string formatToString(nvinfer1::TensorFormat format)
{
    switch (format) {
        case nvinfer1::TensorFormat::kLINEAR:
            return "LINEAR    ";
        case nvinfer1::TensorFormat::kCHW2:
            return "CHW2      ";
        case nvinfer1::TensorFormat::kHWC8:
            return "HWC8      ";
        case nvinfer1::TensorFormat::kCHW4:
            return "CHW4      ";
        case nvinfer1::TensorFormat::kCHW16:
            return "CHW16     ";
        case nvinfer1::TensorFormat::kCHW32:
            return "CHW32     ";
        case nvinfer1::TensorFormat::kDHWC8:
            return "kDHWC8    ";
        case nvinfer1::TensorFormat::kCDHW32:
            return "CDHW32    ";
        case nvinfer1::TensorFormat::kHWC:
            return "HWC       ";
        case nvinfer1::TensorFormat::kDLA_LINEAR:
            return "DLA_LINEAR";
        case nvinfer1::TensorFormat::kDLA_HWC4:
            return "DHWC4     ";
        case nvinfer1::TensorFormat::kHWC16:
            return "HWC16     ";
        case nvinfer1::TensorFormat::kDHWC:
            return "DHWC      ";
        default:
            return "None      ";
    }
}

#ifndef NDEBUG
#define WHERE_AM_I()                                                                                                   \
    printf("%14p[%s]\n", this, __func__);                                                                              \
    fflush(stdout);
#define PRINT_FORMAT_COMBINATION()                                                                                     \
    do {                                                                                                               \
        std::cout << "    pos=" << pos << ":[";                                                                        \
        for (int32_t i = 0; i < nbInputs + nbOutputs; ++i) {                                                           \
            std::cout << dataTypeToString(inOut[i].desc.type) << ",";                                                  \
        }                                                                                                              \
        std::cout << "],[";                                                                                            \
        for (int32_t i = 0; i < nbInputs + nbOutputs; ++i) {                                                           \
            std::cout << formatToString(inOut[i].desc.format) << ",";                                                  \
        }                                                                                                              \
        std::cout << "]->";                                                                                            \
        std::cout << "res=" << res << std::endl;                                                                       \
    } while (0);

#else
#define WHERE_AM_I()                                                                                                   \
    do {                                                                                                               \
    } while (0)
#define PRINT_FORMAT_COMBINATION()                                                                                     \
    do {                                                                                                               \
    } while (0)
#endif  // ifdef NDEBUG

#ifndef NDEBUG
#define CUDA_CHECK(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = (cmd);                                                                                         \
        if (e != cudaSuccess) {                                                                                        \
            printf("Failed: CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUDNN_CHECK(cmd)                                                                                               \
    do {                                                                                                               \
        cudnnStatus_t e = (cmd);                                                                                       \
        if (e != CUDNN_STATUS_SUCCESS) {                                                                               \
            printf("Failed: CUDNN error %s:%d '%s'\n", __FILE__, __LINE__, cudnnGetErrorString(e));                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define GRAPH_CHECK(cmd)                                                                                               \
    do {                                                                                                               \
        fe::error_object e = (cmd);                                                                                    \
        if (!e.is_good()) {                                                                                            \
            printf("Failed: GRAPH error %s:%d '%s'\n", __FILE__, __LINE__, e.err_msg.c_str());                         \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#else
#define CUDA_CHECK(cmd)                                                                                                \
    do {                                                                                                               \
        cudaError_t e = (cmd);                                                                                         \
    } while (0)
#define CUDNN_CHECK(cmd)                                                                                               \
    do {                                                                                                               \
        cudnnStatus_t e = (cmd);                                                                                       \
    } while (0)
#define GRAPH_CHECK(cmd)                                                                                               \
    do {                                                                                                               \
        cudnn_frontend::error_object e = (cmd);                                                                        \
    } while (0)
#endif

namespace nvinfer1 {

class CUDNNAttentionPlugin:
    public IPluginV3,
    public IPluginV3OneCore,
    public IPluginV3OneBuild,
    public IPluginV3OneRuntime {
private:
    const std::string     mName;
    PluginFieldCollection mFCToSerialize;
    cudnnHandle_t         mHandle;

public:
    CUDNNAttentionPlugin() = delete;

    explicit CUDNNAttentionPlugin(const std::string& name);

    CUDNNAttentionPlugin(CUDNNAttentionPlugin const& p) = default;

    ~CUDNNAttentionPlugin() noexcept override;

    void initializeContext();

    void initFieldsToSerialize();

    // IPluginV3 methods
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    CUDNNAttentionPlugin* clone() noexcept override;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild methods
    int32_t configurePlugin(DynamicPluginTensorDesc const* in,
                            int32_t                        nbInputs,
                            DynamicPluginTensorDesc const* out,
                            int32_t                        nbOutputs) noexcept override;

    int32_t getOutputDataTypes(DataType*       outputTypes,
                               int32_t         nbOutputs,
                               DataType const* inputTypes,
                               int32_t         nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs,
                            int32_t          nbInputs,
                            DimsExprs const* shapeInputs,
                            int32_t          nbShapeInputs,
                            DimsExprs*       outputs,
                            int32_t          nbOutputs,
                            IExprBuilder&    exprBuilder) noexcept override;

    bool supportsFormatCombination(int32_t                        pos,
                                   DynamicPluginTensorDesc const* inOut,
                                   int32_t                        nbInputs,
                                   int32_t                        nbOutputs) noexcept override;

    int32_t getNbOutputs() const noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
                            int32_t                        nbInputs,
                            DynamicPluginTensorDesc const* outputs,
                            int32_t                        nbOutputs) const noexcept override;

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;

    int32_t getNbTactics() noexcept override;

    char const* getTimingCacheID() noexcept override;

    int32_t getFormatCombinationLimit() noexcept override;

    char const* getMetadataString() noexcept override;

    // IPluginV3OneRuntime methods
    int32_t setTactic(int32_t tactic) noexcept override;

    int32_t onShapeChange(PluginTensorDesc const* in,
                          int32_t                 nbInputs,
                          PluginTensorDesc const* out,
                          int32_t                 nbOutputs) noexcept override;

    int32_t enqueue(PluginTensorDesc const* inputDesc,
                    PluginTensorDesc const* outputDesc,
                    void const* const*      inputs,
                    void* const*            outputs,
                    void*                   workspace,
                    cudaStream_t            stream) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;
};

class CUDNNAttentionPluginCreator: public IPluginCreatorV3One {
private:
    PluginFieldCollection    mFC;
    std::vector<PluginField> mPluginAttributes;

public:
    CUDNNAttentionPluginCreator();

    ~CUDNNAttentionPluginCreator() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;
};

namespace plugin {
class ThreadSafeLoggerFinder {
private:
    ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex     mMutex;

public:
    ThreadSafeLoggerFinder() = default;

    void     setLoggerFinder(ILoggerFinder* finder);
    ILogger* getLogger() noexcept;
};

}  // namespace plugin

extern "C" __attribute__((visibility("default"))) void setLoggerFinder(nvinfer1::ILoggerFinder* finder);
extern "C" __attribute__((visibility("default"))) IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);

}  // namespace nvinfer1