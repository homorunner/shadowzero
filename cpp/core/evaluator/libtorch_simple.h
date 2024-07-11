#pragma once

#include "core/evaluator/base.h"
#include "core/util/libtorch.h"

// naive libtorch evaluator
class LibtorchEvaluator : public EvaluatorBase {
 public:
  LibtorchEvaluator(std::string model_path_, const std::array<int, 3>& dimentions, bool cpu_only = false,
                    bool warmup = true, bool verbose = true)
      : device("cpu"), model_path(model_path_) {
    torch::print_libtorch_version();

#ifdef USE_CUDA
    if (!cpu_only && torch::cuda::is_available()) {
      torch::print_CUDA_cuDNN_info();
      if (verbose) std::cout << "Using CUDA." << std::endl;
      device = torch::Device("cuda:0");
      options = options.device(device);
    } else {
      if (verbose) std::cout << "Using CPU." << std::endl;
    }
#endif
    c10::InferenceMode guard;
    model = torch::jit::load(model_path, device);
    if (model.is_training()) {
      if (verbose) std::cout << "Warning: Model is in training mode. Calling eval()." << std::endl;
      model.eval();
    }

    d1 = dimentions[0];
    d2 = dimentions[1];
    d3 = dimentions[2];

    // undocumented API that may be useful to optimize the model
    torch::jit::optimize_for_inference(model);

    // warm up the model
    if (warmup) {
      if (verbose) std::cout << "Warming up." << std::endl;
      std::vector<torch::jit::IValue> inputs = {torch::ones({1, d1, d2, d3}, options)};
      auto _ = model.forward(inputs).toTuple();
      if (_->elements().size() > 0) {
        if (verbose) std::cout << "Warm up ok." << std::endl;
      }
    }
  }

  // TODO: maybe not pass std function as value here, e.g. use template.
  //       maybe return std::future<void> here.
  void evaluate(std::function<void(float*)> canonicalize,
                std::function<void(const float*, const float*)> process_result, uint64_t hashval = 0) {
    c10::InferenceMode guard;
    auto input = torch::zeros({1, d1, d2, d3});
    canonicalize(input.data_ptr<float>());

    std::vector<torch::jit::IValue> inputs = {input.to(device)};
    model_mutex.lock();
    auto outputs = model.forward(inputs).toTuple();
    model_mutex.unlock();
    auto v = torch::exp(outputs->elements()[0].toTensor()).cpu();
    auto pi = torch::exp(outputs->elements()[1].toTensor()).cpu();

    process_result(pi.data_ptr<float>(), v.data_ptr<float>());
  }

  void evaluateN(int N, std::function<void(float*)>* canonicalizes,
                 std::function<void(const float*, const float*)>* process_results) {
    for (int i = 0; i < N; i++) {
      evaluate(canonicalizes[i], process_results[i]);
    }
  }

 private:
  torch::Device device;
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat);

  int d1, d2, d3;

  std::mutex model_mutex;
  torch::jit::script::Module model;
  std::string model_path;
};