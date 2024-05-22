#pragma once

#include "core/evaluator/base.h"
#include "core/util/libtorch.h"

// enhanced evaluator that uses a separate thread to evaluate the model
class QueuedLibtorchEvaluator : public EvaluatorBase {
 public:
  QueuedLibtorchEvaluator(std::string model_path,
                          const std::array<int, 3>& dimentions,
                          bool cpu_only = false, int device_id = 0,
                          bool warmup = true, bool verbose = true)
      : device("cpu") {
    torch::print_libtorch_version();

#ifdef USE_CUDA
    if (!cpu_only && torch::cuda::is_available()) {
      torch::print_CUDA_cuDNN_info();
      if (verbose) std::cout << "Using CUDA." << std::endl;
      device = torch::Device("cuda:" + std::to_string(device_id));
      options = options.device(device);
    } else {
      if (verbose) std::cout << "Using CPU." << std::endl;
    }
#endif
    c10::InferenceMode guard;
    model = torch::jit::load(model_path, device);
    if (model.is_training()) {
      if (verbose)
        std::cout << "Warning: Model is in training mode. Calling eval()."
                  << std::endl;
      model.eval();
    }

    d1 = dimentions[0];
    d2 = dimentions[1];
    d3 = dimentions[2];
    dx = d1 * d2 * d3;

    // undocumented API that may be useful to optimize the model
    torch::jit::optimize_for_inference(model);

    // warm up the model
    if (warmup) {
      if (verbose) std::cout << "Warming up." << std::endl;
      std::vector<torch::jit::IValue> inputs = {
          torch::ones({1, d1, d2, d3}, options)};
      auto _ = model.forward(inputs).toTuple();
      if (_->elements().size() > 0) {
        if (verbose) std::cout << "Warm up ok." << std::endl;
      }
    }

    // initialize thread control
    job_done[0] = job_done[1] = false;
    working_input_size = 0;
    stop_eval = false;

    eval_thread = std::make_unique<std::thread>([this]() {
      c10::InferenceMode guard;
      while (!stop_eval) {
        if (working_input_size == 0) {
          continue;
        }
        input_mutex.lock();
        auto input = torch::from_blob(working_input.data(),
                                      {working_input_size, d1, d2, d3});
        // auto input_guard = torch::from_blob(working_input_guard.data(),
        //                                     {working_input_size, 1});
        if (device.is_cpu()) {
          input = input.clone();
          // input_guard = input_guard.clone();
        } else {
          input = input.to(device);
          // input_guard = input_guard.to(device);
        }
        std::vector<torch::jit::IValue> inputs = {input};
        // std::vector<torch::jit::IValue> inputs_guard = {input_guard};
        total_working_input_size += working_input_size;
        working_input.clear();
        // working_input_guard.clear();
        working_input_size = 0;
        working_index += 1;
        job_done[(working_index + 1) % 64] = false;
        input_mutex.unlock();

        working_input_count++;
        auto outputs = model.forward(inputs).toTuple();

        // TODO: move torch::exp to inside the model
        output_v[working_index % 64] =
            torch::exp(outputs->elements()[0].toTensor()).cpu();
        output_pi[working_index % 64] =
            torch::exp(outputs->elements()[1].toTensor()).cpu();
        // output_guard[working_index % 64] = inputs_guard[0].toTensor().cpu();
        job_done[working_index % 64] = true;
        job_done[working_index % 64].notify_all();
      }
    });
  }

  ~QueuedLibtorchEvaluator() {
    stop_eval = true;
    eval_thread->join();
  }

  // TODO: maybe not pass std function as value here, e.g. use template.
  //       maybe return std::future<void> here.
  void evaluate(std::function<void(float*)> canonicalize,
                std::function<void(const float*, const float*)> process_result,
                uint64_t hashval = 0) {
    c10::InferenceMode guard;
    while (true) {
      input_mutex.lock();
      int current_size = working_input_size;
      working_input.resize((working_input_size + 1) * dx, 0);
      canonicalize(working_input.data() + working_input_size * dx);
      working_input_size += 1;
      // float guard_val = randn(10086);
      // working_input_guard.push_back(guard_val);
      uint8_t my_index_ = (working_index + 1) % 64;
      input_mutex.unlock();

      job_done[my_index_].wait(false);

      // if (output_v[my_index_].size(0) < current_size) {
      //   std::cerr << "Output size mismatch!" << std::endl;
      //   continue;
      // }
      // if (output_guard[my_index_].size(0) < current_size) {
      //   std::cerr << "Guard size mismatch!" << std::endl;
      //   continue;
      // }
      // if (auto g = output_guard[my_index_][current_size].item<float>();
      //     g != guard_val) {
      //   std::cerr << "Guard value mismatch!" << g << guard_val << std::endl;
      //   continue;
      // }
      process_result(output_pi[my_index_][current_size].data_ptr<float>(),
                     output_v[my_index_][current_size].data_ptr<float>());
      break;
    }
  }

  void evaluateN(
      int N, std::function<void(float*)>* canonicalizes,
      std::function<void(const float*, const float*)>* process_results) {
    c10::InferenceMode guard;
    input_mutex.lock();
    int current_size = working_input_size;
    working_input.resize((working_input_size + N) * dx, 0);
    for (int i = 0; i < N; i++) {
      canonicalizes[i](working_input.data() + (working_input_size + i) * dx);
    }
    working_input_size += N;
    uint8_t my_index_ = (working_index + 1) % 64;
    input_mutex.unlock();

    job_done[my_index_].wait(false);

    assert(output_v[my_index_].size(0) >= current_size + N - 1);
    for (int i = 0; i < N; i++) {
      process_results[i](
          output_pi[my_index_][current_size + i].data_ptr<float>(),
          output_v[my_index_][current_size + i].data_ptr<float>());
    }
  }

  std::string statistics() {
    std::stringstream ss;
    ss << "Aaverage input size: "
       << total_working_input_size / (double)working_input_count;
    return ss.str();
  }

 private:
  torch::Device device;
  torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat);

  int d1, d2, d3, dx;

  torch::jit::script::Module model;

  std::mutex input_mutex;
  volatile int working_index = 0;
  std::atomic<bool> job_done[64];
  std::vector<float> working_input;
  volatile int working_input_size;
  // std::vector<float> working_input_guard; // debug only
  torch::Tensor output_pi[64], output_v[64];
  // torch::Tensor output_guard[64];  // debug only

  std::unique_ptr<std::thread> eval_thread;
  std::atomic<bool> stop_eval;

  int total_working_input_size = 0, working_input_count = 0;
};