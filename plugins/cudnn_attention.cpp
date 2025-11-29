//
// Created by ubuntu on 25-09-06.
//
#include "cudnn_attention.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace plugin_utils {

struct MHAParams {
    int64_t        b;
    int64_t        n;
    int64_t        s_q;
    int64_t        s_kv;
    int64_t        d;
    fe::DataType_t dataType;
};

template<typename T>
struct ParamsWrapper {
    T pod;
    static_assert(std::is_standard_layout_v<T>, "ParamsWrapper cannot wrap non-POD data");

    ParamsWrapper(): pod()
    {
        std::memset(&(pod), 0, sizeof(pod));
    }

    ParamsWrapper(const ParamsWrapper& other)
    {
        std::memcpy(&(pod), &(other.pod), sizeof(pod));
    }

    ParamsWrapper(ParamsWrapper&& other) noexcept
    {
        std::memcpy(&(pod), &(other.pod), sizeof(pod));
    }

    ParamsWrapper& operator=(const ParamsWrapper& other)
    {
        std::memcpy(&(pod), &(other.pod), sizeof(pod));
        return *this;
    }

    ParamsWrapper& operator=(ParamsWrapper&& other) noexcept
    {
        std::memcpy(&(pod), &(other.pod), sizeof(pod));
        return *this;
    }

    inline friend bool operator==(const ParamsWrapper& lhs, const ParamsWrapper& rhs) noexcept
    {
        auto ptr1 = reinterpret_cast<const uint8_t*>(&(lhs.pod));
        auto ptr2 = reinterpret_cast<const uint8_t*>(&(rhs.pod));
        return std::memcmp(ptr1, ptr2, sizeof(lhs.pod)) == 0;
    }
};

struct MHACacheKeyWrapper: ParamsWrapper<MHAParams> {
    MHACacheKeyWrapper(
        int64_t b, int64_t n, int64_t s_q, int64_t s_kv, int64_t d, fe::DataType_t dataType = fe::DataType_t::BFLOAT16)
    {
        pod.b        = b;
        pod.n        = n;
        pod.s_q      = s_q;
        pod.s_kv     = s_kv;
        pod.d        = d;
        pod.dataType = dataType;
    }
};

template<typename ParamsWrapper>
struct ParamsWrapperHash {
    // Params must be a POD because we read out its memory
    // contents as char* when hashing
    static_assert(std::is_standard_layout_v<decltype(ParamsWrapper::pod)>, "ParamsWrapper cannot wrap non-POD data");

    size_t operator()(const ParamsWrapper& params_wrapper) const
    {
        auto     ptr   = reinterpret_cast<const uint8_t*>(&(params_wrapper.pod));
        uint32_t value = 0x811C9DC5;
        for (int i = 0; i < sizeof(params_wrapper.pod); ++i) {
            value ^= ptr[i];
            value *= 0x01000193;
        }
        return (size_t)value;
    }
};

template<typename T, typename KeyType>
struct MHAGraphCache {
    std::unordered_map<KeyType, T, ParamsWrapperHash<KeyType>> engine_cache;
    int                                                        count = 0;
    int                                                        hits  = 0;

    // no mutexes here as caches are now thread local for v8, can also return a
    // pointer to the Execution Plan if we know it will not be invalidated by
    // another thread
    T* find(const KeyType& key)
    {
        count++;
        auto it = engine_cache.find(key);
        if (it == engine_cache.end()) {
            return nullptr;
        }
        hits++;
        return &(it->second);
    }

    void update(const KeyType& key, T& results)
    {
        engine_cache.erase(key);
        engine_cache.emplace(key, std::move(results));
    }
};

// @eqy: use thread local caches as cuDNN Execution Plans are not guaranteed to
// be thread safe across all engines see Limitations in
// https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html
// We also leak the caches to workaround potential teardown race issues.

MHAGraphCache<std::shared_ptr<fe::graph::Graph>, MHACacheKeyWrapper>& getMHAGraphCache_()
{
    thread_local MHAGraphCache<std::shared_ptr<fe::graph::Graph>, MHACacheKeyWrapper>& instance =
        *new MHAGraphCache<std::shared_ptr<fe::graph::Graph>, MHACacheKeyWrapper>;
    return instance;
}

std::shared_ptr<fe::graph::Graph>
buildGraph(int64_t b, int64_t n, int64_t s_q, int64_t s_kv, int64_t d, fe::DataType_t dataType, cudnnHandle_t& handle)
{
    auto mha_graph = std::make_shared<fe::graph::Graph>();
    // We're baking in float accumulation and scale types
    // in theory the graph may support other types, but they
    // have not been tested
    mha_graph->set_io_data_type(dataType)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                            .set_uid('S')
                                            .set_name("attn_scale")
                                            .set_dim({1, 1, 1, 1})
                                            .set_stride({1, 1, 1, 1})
                                            .set_is_pass_by_value(true)
                                            .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_opt = fe::graph::SDPA_attributes()
                        .set_name("CUDNN_SDPA")
                        .set_generate_stats(false)
                        .set_is_inference(true)
                        .set_alibi_mask(false)
                        .set_padding_mask(false)
                        .set_causal_mask(false)
                        .set_causal_mask_bottom_right(false)
                        .set_attn_scale(attn_scale)
                        .set_implementation(fe::AttentionImplementation_t::AUTO);

    auto Q_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Q")
                                    .set_uid('Q')
                                    .set_dim({b, n, s_q, d})
                                    .set_stride({s_q * n * d, d, n * d, 1}));

    auto K_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("K")
                                    .set_uid('K')
                                    .set_dim({b, n, s_kv, d})
                                    .set_stride({s_kv * n * d, d, n * d, 1}));

    auto V_ = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("V")
                                    .set_uid('V')
                                    .set_dim({b, n, s_kv, d})
                                    .set_stride({s_kv * n * d, d, n * d, 1}));

    auto [O_, Stats] = mha_graph->sdpa(Q_, K_, V_, sdpa_opt);
    assert(Stats == nullptr);
    O_->set_output(true).set_name("O").set_uid('O').set_dim({b, n, s_q, d}).set_stride({s_q * n * d, d, n * d, 1});

    GRAPH_CHECK(mha_graph->validate());
    GRAPH_CHECK(mha_graph->build_operation_graph(handle));
    GRAPH_CHECK(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
    GRAPH_CHECK(mha_graph->check_support(handle));
    GRAPH_CHECK(mha_graph->build_plans(handle));

    return mha_graph;
}

}  // namespace plugin_utils

namespace nvinfer1 {

CUDNNAttentionPlugin::CUDNNAttentionPlugin(const std::string& name): mName(name)
{
    WHERE_AM_I();
    initFieldsToSerialize();
    mHandle = nullptr;
}

CUDNNAttentionPlugin::~CUDNNAttentionPlugin() noexcept
{
    if (mHandle) {
        CUDNN_CHECK(cudnnDestroy(mHandle));
    }
}

void CUDNNAttentionPlugin::initializeContext()
{
    CUDNN_CHECK(cudnnCreate(&mHandle));
}

void CUDNNAttentionPlugin::initFieldsToSerialize()
{
    WHERE_AM_I();
    mFCToSerialize.nbFields = 0;
    mFCToSerialize.fields   = nullptr;
}

IPluginCapability* CUDNNAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    WHERE_AM_I();
    switch (type) {
        case PluginCapabilityType::kBUILD: {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        case PluginCapabilityType::kRUNTIME: {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        case PluginCapabilityType::kCORE: {
            return static_cast<IPluginV3OneCore*>(this);
        }
    }
    return nullptr;
}

CUDNNAttentionPlugin* CUDNNAttentionPlugin::clone() noexcept
{
    WHERE_AM_I();
    auto clone_p = new CUDNNAttentionPlugin(*this);
    clone_p->initFieldsToSerialize();
    return clone_p;
}

char const* CUDNNAttentionPlugin::getPluginName() const noexcept
{
    WHERE_AM_I();
    return "CUDNNAttention";
}

char const* CUDNNAttentionPlugin::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return "1";
}

char const* CUDNNAttentionPlugin::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return "";
}

int32_t CUDNNAttentionPlugin::configurePlugin(DynamicPluginTensorDesc const* in,
                                              int32_t                        nbInputs,
                                              DynamicPluginTensorDesc const* out,
                                              int32_t                        nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CUDNNAttentionPlugin::getOutputDataTypes(DataType*       outputTypes,
                                                 int32_t         nbOutputs,
                                                 DataType const* inputTypes,
                                                 int32_t         nbInputs) const noexcept
{
    WHERE_AM_I();
    if (nbInputs != 3) {
        std::cerr << "CUDNNAttentionPlugin only support 3 inputs, but got " << nbInputs << " inputs!\n";
        return -1;
    }
    if (nbOutputs != 1) {
        std::cerr << "CUDNNAttentionPlugin only support 1 outputs, but got " << nbOutputs << " outputs!\n";
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t CUDNNAttentionPlugin::getOutputShapes(DimsExprs const* inputs,
                                              int32_t          nbInputs,
                                              DimsExprs const* shapeInputs,
                                              int32_t          nbShapeInputs,
                                              DimsExprs*       outputs,
                                              int32_t          nbOutputs,
                                              IExprBuilder&    exprBuilder) noexcept
{
    WHERE_AM_I();
    if (nbInputs != 3 || nbOutputs != 1) {
        std::cerr << "nbInputs/nbOutputs should be equal to 3/1, but got nbInputs=" << nbInputs
                  << " nbOutputs=" << nbOutputs << '\n';
        return -1;
    }
    outputs[0] = inputs[0];
    return 0;
}

bool CUDNNAttentionPlugin::supportsFormatCombination(int32_t                        pos,
                                                     DynamicPluginTensorDesc const* inOut,
                                                     int32_t                        nbInputs,
                                                     int32_t                        nbOutputs) noexcept
{
    WHERE_AM_I();
    bool res{false};
    switch (pos) {
        case 0:  // q
            res = (inOut[0].desc.type == DataType::kBF16 || inOut[0].desc.type == DataType::kHALF)
                  && inOut[0].desc.format == TensorFormat::kLINEAR;
            break;
        case 1:  // k
            res = inOut[1].desc.type == inOut[0].desc.type && inOut[1].desc.format == inOut[0].desc.format;
            break;
        case 2:  // v
            res = inOut[2].desc.type == inOut[0].desc.type && inOut[2].desc.format == inOut[0].desc.format;
            break;
        case 3:  // o
            res = inOut[3].desc.type == inOut[0].desc.type && inOut[3].desc.format == inOut[0].desc.format;
            break;
        default:  // should NOT be here!
            std::cerr << "pos=" << pos << " is not supported!\n";
            return res;
    }
    PRINT_FORMAT_COMBINATION();
    return res;
}

int32_t CUDNNAttentionPlugin::getNbOutputs() const noexcept
{
    WHERE_AM_I();
    return 1;
}

size_t CUDNNAttentionPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs,
                                              int32_t                        nbInputs,
                                              DynamicPluginTensorDesc const* outputs,
                                              int32_t                        nbOutputs) const noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CUDNNAttentionPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CUDNNAttentionPlugin::getNbTactics() noexcept
{
    WHERE_AM_I();
    return 0;
}

char const* CUDNNAttentionPlugin::getTimingCacheID() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t CUDNNAttentionPlugin::getFormatCombinationLimit() noexcept
{
    WHERE_AM_I();
    return kDEFAULT_FORMAT_COMBINATION_LIMIT;
}

char const* CUDNNAttentionPlugin::getMetadataString() noexcept
{
    WHERE_AM_I();
    return nullptr;
}

int32_t CUDNNAttentionPlugin::setTactic(int32_t tactic) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CUDNNAttentionPlugin::onShapeChange(PluginTensorDesc const* in,
                                            int32_t                 nbInputs,
                                            PluginTensorDesc const* out,
                                            int32_t                 nbOutputs) noexcept
{
    WHERE_AM_I();
    return 0;
}

int32_t CUDNNAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc,
                                      PluginTensorDesc const* outputDesc,
                                      void const* const*      inputs,
                                      void* const*            outputs,
                                      void*                   workspace,
                                      cudaStream_t            stream) noexcept
{
    WHERE_AM_I();

#ifndef NDEBUG
    NANO_SECOND t0, t1, t2, t3, t4, t5, t6;

    t0 = std::chrono::high_resolution_clock::now();
#endif

    int64_t b          = inputDesc[0].dims.d[0];
    int64_t s_q        = inputDesc[0].dims.d[1];
    int64_t n          = inputDesc[0].dims.d[2];
    int64_t d          = inputDesc[0].dims.d[3];
    int64_t s_kv       = inputDesc[1].dims.d[1];
    float   attn_scale = 1.0f / std::sqrt(static_cast<float>(d));
    CUDNN_CHECK(cudnnSetStream(mHandle, stream));
    fe::DataType_t dataType = dataTypeToFeType(inputDesc[0].type);

#ifndef NDEBUG
    t1 = std::chrono::high_resolution_clock::now();
#endif

    plugin_utils::MHACacheKeyWrapper key = plugin_utils::MHACacheKeyWrapper(b, n, s_q, s_kv, d, dataType);

#ifndef NDEBUG
    t2 = std::chrono::high_resolution_clock::now();
#endif

    auto graph_ptr = plugin_utils::getMHAGraphCache_().find(key);

#ifndef NDEBUG
    t3 = std::chrono::high_resolution_clock::now();
#endif

    std::shared_ptr<fe::graph::Graph> mha_graph;
    if (graph_ptr) {
        mha_graph = *graph_ptr;
    }
    else {
        mha_graph = plugin_utils::buildGraph(b, n, s_q, s_kv, d, dataType, mHandle);
    }

#ifndef NDEBUG
    t4 = std::chrono::high_resolution_clock::now();
#endif

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack_data = {
        {'Q', const_cast<void*>(inputs[0])},
        {'K', const_cast<void*>(inputs[1])},
        {'V', const_cast<void*>(inputs[2])},
        {'O', outputs[0]},
        {'S', &attn_scale},
    };

    GRAPH_CHECK(mha_graph->execute(mHandle, pack_data, workspace));

#ifndef NDEBUG
    t5 = std::chrono::high_resolution_clock::now();
#endif

    plugin_utils::getMHAGraphCache_().update(key, mha_graph);

#ifndef NDEBUG
    t6 = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(5) << "setHandle[" << costStr(t0, t1) << "] | buildCache["
              << costStr(t1, t2) << "] | getCache[" << costStr(t2, t3) << "] | buildGraph[" << costStr(t3, t4)
              << "] | executeGraph[" << costStr(t4, t5) << "] | updateCache[" << costStr(t5, t6) << "]\n";
#endif
    return 0;
}

IPluginV3* CUDNNAttentionPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    WHERE_AM_I();
    CUDNNAttentionPlugin* ret = clone();
    ret->initializeContext();
    return ret;
}

PluginFieldCollection const* CUDNNAttentionPlugin::getFieldsToSerialize() noexcept
{
    WHERE_AM_I();
    return &mFCToSerialize;
}

CUDNNAttentionPluginCreator::CUDNNAttentionPluginCreator()
{
    WHERE_AM_I();
}

CUDNNAttentionPluginCreator::~CUDNNAttentionPluginCreator() noexcept
{
    WHERE_AM_I();
}

IPluginV3* CUDNNAttentionPluginCreator::createPlugin(char const*                  name,
                                                     PluginFieldCollection const* fc,
                                                     TensorRTPhase                phase) noexcept
{
    WHERE_AM_I();
    return new CUDNNAttentionPlugin(name);
}

PluginFieldCollection const* CUDNNAttentionPluginCreator::getFieldNames() noexcept
{
    WHERE_AM_I();
    return &mFC;
}

char const* CUDNNAttentionPluginCreator::getPluginName() const noexcept
{
    WHERE_AM_I();
    return "CUDNNAttention";
}

char const* CUDNNAttentionPluginCreator::getPluginVersion() const noexcept
{
    WHERE_AM_I();
    return "1";
}

char const* CUDNNAttentionPluginCreator::getPluginNamespace() const noexcept
{
    WHERE_AM_I();
    return "";
}

namespace plugin {
void ThreadSafeLoggerFinder::setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr) {
        mLoggerFinder = finder;
    }
}

ILogger* ThreadSafeLoggerFinder::getLogger() noexcept
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr) {
        return mLoggerFinder->findLogger();
    }
    return nullptr;
}

ThreadSafeLoggerFinder gLoggerFinder;

ILogger* getPluginLogger()
{
    return gLoggerFinder.getLogger();
}
}  // namespace plugin

extern "C" __attribute__((visibility("default"))) void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    nvinfer1::plugin::gLoggerFinder.setLoggerFinder(finder);
}

extern "C" __attribute__((visibility("default"))) IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static CUDNNAttentionPluginCreator    creator;
    static IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = {&creator};
    return kPLUGIN_CREATOR_LIST;
}

REGISTER_TENSORRT_PLUGIN(CUDNNAttentionPluginCreator);

}  // namespace nvinfer1
