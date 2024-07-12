#include "core/evaluator/libtorch_simple.h"
#include "core/evaluator/onnx_simple.h"
#include "game/shadow.h"

void run_libtorch(const std::string& model, int num) {
  auto game = Shadow::GameState();
  auto evaluator = LibtorchEvaluator(model, Shadow::CANONICAL_SHAPE);
  auto result = 0.0;
  for (int i = 0; i < num; i++) {
    evaluator.evaluate(std::bind(&Shadow::GameState::Canonicalize, &game, std::placeholders::_1),
      [&result](const float* pi, const float* v) {
        result += pi[0] + v[0];
      });
  }
  std::cout << "result = " << result << std::endl;
}

void run_onnx(const std::string& model, int num) {
  auto game = Shadow::GameState();
  auto evaluator = OnnxEvaluator(model, Shadow::CANONICAL_SHAPE, Shadow::NUM_ACTIONS);
  auto result = 0.0;
  for (int i = 0; i < num; i++) {
    evaluator.evaluate(std::bind(&Shadow::GameState::Canonicalize, &game, std::placeholders::_1),
      [&result](const float* pi, const float* v) {
        result += pi[0] + v[0];
      });
  }
  std::cout << "result = " << result << std::endl;
}

int main(int argc, const char** argv) {
  int NumIterations = 10000;
  if (argc == 3 && strcmp(argv[1], "-i") == 0) {
    NumIterations = std::atoi(argv[2]);
  }

  {
    auto start = high_resolution_clock::now();
    run_onnx("testdata/example.onnx", NumIterations);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "Time for onnx: " << duration << "ms\nIteration: " << NumIterations << std::endl;
  }

  {
    auto start = high_resolution_clock::now();
    run_libtorch("testdata/example.pt", NumIterations);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    std::cout << "Time for libtorch: " << duration << "ms\nIteration: " << NumIterations << std::endl;
  }
  return 0;
}
