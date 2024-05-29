#include "core/evaluator/libtorch_simple.h"
#include "game/shadow.h"

int main(int argc, const char** argv) {
  int NumIterations = 10000;
  if (argc == 3 && strcmp(argv[1], "-i") == 0) {
    NumIterations = std::atoi(argv[2]);
  }

  std::string model = "testdata/example_shadow_model.pt";

  auto evaluator = LibtorchEvaluator(model, Shadow::CANONICAL_SHAPE);
  auto game = Shadow::GameState();

  auto start = high_resolution_clock::now();

  for (int i = 0; i < NumIterations; i++) {
    evaluator.evaluate(std::bind(&Shadow::GameState::Canonicalize, &game,
                                 std::placeholders::_1),
                       [](const float* input, const float* output) {});
  }

  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  std::cout << "Time: " << duration << "ms\nIteration: " << NumIterations
            << std::endl;
  return 0;
}
