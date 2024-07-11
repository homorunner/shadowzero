#include "core/evaluator/libtorch_simple.h"
#include "core/evaluator/onnx_simple.h"
#include "game/shadow.h"

void run_libtorch(const std::string& model, int num) {
  auto game = Shadow::GameState();
  auto evaluator = LibtorchEvaluator(model, Shadow::CANONICAL_SHAPE);
  for (int i = 0; i < num; i++) {
    evaluator.evaluate(std::bind(&Shadow::GameState::Canonicalize, &game, std::placeholders::_1),
                       [](const float* pi, const float* v) {
                         std::cout << v[0] << " " << v[1] << std::endl;
                         std::cout << pi[0] << " " << pi[1] << " " << pi[2] << "..." << std::endl;
                       });
  }
}

void run_onnx(const std::string& model, int num) {
  auto game = Shadow::GameState();
  auto evaluator = OnnxEvaluator(model, Shadow::CANONICAL_SHAPE, Shadow::NUM_ACTIONS);
  for (int i = 0; i < num; i++) {
    evaluator.evaluate(std::bind(&Shadow::GameState::Canonicalize, &game, std::placeholders::_1),
                       [](const float* pi, const float* v) {
                         std::cout << v[0] << " " << v[1] << std::endl;
                         std::cout << pi[0] << " " << pi[1] << " " << pi[2] << "..." << std::endl;
                       });
  }
}

int main(int argc, const char** argv) {
  int NumIterations = 10000;
  if (argc == 3 && strcmp(argv[1], "-i") == 0) {
    NumIterations = std::atoi(argv[2]);
  }

  auto start = high_resolution_clock::now();
  run_libtorch("testdata/example.pt", NumIterations);
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  std::cout << "Time: " << duration << "ms\nIteration: " << NumIterations << std::endl;

  start = high_resolution_clock::now();
  run_onnx("testdata/example.onnx", NumIterations);
  end = high_resolution_clock::now();
  duration = duration_cast<milliseconds>(end - start).count();
  std::cout << "Time: " << duration << "ms\nIteration: " << NumIterations << std::endl;
  return 0;
}
