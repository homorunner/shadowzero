#include <onnxruntime_cxx_api.h>
#include "game/shadow.h"

#include <iostream>
#include <vector>

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

  Ort::SessionOptions sessionOptions;
  OrtCUDAProviderOptions cudaProviderOptions;
  cudaProviderOptions.device_id = 0;
  cudaProviderOptions.gpu_mem_limit = SIZE_MAX;
  sessionOptions.AppendExecutionProvider_CUDA(cudaProviderOptions);

  const char* modelPath = "testdata/example.onnx";

  Ort::Session session(env, modelPath, sessionOptions);
  Ort::AllocatorWithDefaultOptions allocator;

  auto inputName = session.GetInputNameAllocated(0, allocator);
  std::vector<int64_t> inputNodeDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  for (size_t i = 0; i < inputNodeDims.size(); i++) {
    if (inputNodeDims[i] == -1) {
      inputNodeDims[i] = 1;
    }
  }
  size_t inputTensorSize = 1;
  for (size_t i = 0; i < inputNodeDims.size(); i++) {
    inputTensorSize *= inputNodeDims[i];
  }
  std::cout << "Input size: " << inputTensorSize << std::endl;
  std::vector<float> inputTensorValues(inputTensorSize);

  auto game = Shadow::GameState();
  game.Canonicalize(inputTensorValues.data());

  std::vector<Ort::Value> inputTensors;
  inputTensors.push_back(Ort::Value::CreateTensor<float>(allocator.GetInfo(), inputTensorValues.data(), inputTensorSize,
                                                         inputNodeDims.data(), inputNodeDims.size()));

  const char* inputNames[] = {inputName.get()};
  const char* outputNames[] = {"v", "p"};

  // auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames,
  //                                  &inputTensors[0], 1, outputNames, 2);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<float> v(2), pi(1024);
  int64_t v_shape[2] = {1, 2};
  int64_t pi_shape[2] = {1, 1024};
  Ort::Value outputTensors[2] = {
      Ort::Value::CreateTensor<float>(memory_info, v.data(), 2 * sizeof(float), v_shape, 2),
      Ort::Value::CreateTensor<float>(memory_info, pi.data(), 1024 * sizeof(float), pi_shape, 2)};
  session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensors[0], 1, outputNames, outputTensors, 2);

  auto output1shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  auto output1 = (float*)outputTensors[0].GetTensorRawData();
  std::cout << "Output 1 shape: " << output1shape[0] << " " << output1shape[1] << std::endl;
  std::cout << "Output 1 value: " << output1[0] << " " << output1[1] << std::endl;

  auto output2shape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
  auto output2 = (float*)outputTensors[1].GetTensorRawData();
  std::cout << "Output 2 shape: " << output2shape[0] << " " << output2shape[1] << std::endl;
  std::cout << "Output 2 value: " << output2[0] << " " << output2[1] << "..." << std::endl;

  return 0;
}