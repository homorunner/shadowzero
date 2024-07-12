#pragma once

#include <onnxruntime_cxx_api.h>

#include "core/evaluator/base.h"

const char* inputNames[] = {"in"};
const char* outputNames[] = {"v", "p"};

void exp(std::vector<float>& arr) {
  for (auto& i : arr) {
    i = std::exp(i);
  }
}

void exp_fast(std::vector<float>& arr) {
  [[assume(arr.size() % 128 == 0)]];
  for (auto& i : arr) {
    i = std::exp(i);
  }
}

// naive onnx evaluator
class OnnxEvaluator : public EvaluatorBase {
 public:
  OnnxEvaluator(std::string model_path_, const std::array<int, 3>& input_size, int output_action_size,
                bool cpu_only = false, int device_id = 0, bool warmup = true, bool verbose = true)
      : model_path(model_path_), pi_size(output_action_size) {
    d[0] = 1;
    d[1] = input_size[0];
    d[2] = input_size[1];
    d[3] = input_size[2];

#ifdef USE_CUDA
#endif

    const OrtApi& api = Ort::GetApi();

    Ort::Env env;
    api.CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_evaluator", (OrtEnv**)(&env));
    
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions sessionOptions;

    // Legacy way to use CudaProviderOption.
    OrtCUDAProviderOptions cudaProviderOptions;
    cudaProviderOptions.device_id = device_id;
    cudaProviderOptions.gpu_mem_limit = SIZE_MAX;
    sessionOptions.AppendExecutionProvider_CUDA(cudaProviderOptions);

    // Use CudaProviderOptionsV2 to enable cuda graph in cuda provider option.
    // OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    // api.CreateCUDAProviderOptions(&cuda_options);
    // std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
    // std::vector<const char*> keys{ "cudnn_conv_algo_search" }; // "enable_cuda_graph"};
    // std::vector<const char*> values{ "HEURISTIC" }; // "1"};
    // api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1);
    // api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(sessionOptions), rel_cuda_options.get());

    session = std::make_unique<Ort::Session>(env, model_path_.c_str(), sessionOptions);

    // warm up the model
    if (warmup) {
      if (verbose) std::cout << "Warming up." << std::endl;

      std::vector<float> input(d[1] * d[2] * d[3], 1.0f);
      auto inputTensor =
          Ort::Value::CreateTensor<float>(allocator.GetInfo(), input.data(), d[1] * d[2] * d[3] * sizeof(float), d, 4);
      session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 2);

      if (verbose) std::cout << "Warm up ok." << std::endl;
    }
  }

  void evaluate(std::function<void(float*)> canonicalize,
                std::function<void(const float*, const float*)> process_result, uint64_t hashval = 0) {
    std::vector<float> input(d[1] * d[2] * d[3], 0), v(2), pi(pi_size);

    // Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    auto inputTensor =
        Ort::Value::CreateTensor<float>(memory_info, input.data(), d[1] * d[2] * d[3] * sizeof(float), d, 4);

    canonicalize(input.data());

    int64_t v_shape[2] = {1, 2};
    int64_t pi_shape[2] = {1, pi_size};
    Ort::Value outputTensors[2] = {
        Ort::Value::CreateTensor<float>(memory_info, v.data(), 2 * sizeof(float), v_shape, 2),
        Ort::Value::CreateTensor<float>(memory_info, pi.data(), pi_size * sizeof(float), pi_shape, 2)};

    Ort::RunOptions run_options;
    model_mutex.lock();
    session->Run(run_options, inputNames, &inputTensor, 1, outputNames, outputTensors, 2);
    model_mutex.unlock();
    exp(v);
    exp_fast(pi);

    process_result(pi.data(), v.data());
  }

  void evaluateN(int N, std::function<void(float*)>* canonicalizes,
                 std::function<void(const float*, const float*)>* process_results) {
    std::vector<float> input(N * d[1] * d[2] * d[3], 0), v(N * 2), pi(N * pi_size);

    // Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    int64_t input_shape[4] = {N, d[1], d[2], d[3]};

    auto inputTensor =
        Ort::Value::CreateTensor<float>(memory_info, input.data(), N * d[1] * d[2] * d[3] * sizeof(float), input_shape, 4);

    for (int i = 0; i < N; i++) {
      canonicalizes[i](input.data() + i * d[1] * d[2] * d[3]);
    }

    int64_t v_shape[2] = {N, 2};
    int64_t pi_shape[2] = {N, pi_size};
    Ort::Value outputTensors[2] = {
        Ort::Value::CreateTensor<float>(memory_info, v.data(), N * 2 * sizeof(float), v_shape, 2),
        Ort::Value::CreateTensor<float>(memory_info, pi.data(), N * pi_size * sizeof(float), pi_shape, 2)};

    Ort::RunOptions run_options;
    session->Run(run_options, inputNames, &inputTensor, 1, outputNames, outputTensors, 2);
    exp(v);
    exp_fast(pi);

    for (int i = 0; i < N; i++) {
      process_results[i](pi.data() + i * pi_size, v.data() + i * 2);
    }
  }

 private:
  std::string model_path;
  std::unique_ptr<Ort::Session> session;
  int64_t d[4], pi_size;
  std::mutex model_mutex;
};
