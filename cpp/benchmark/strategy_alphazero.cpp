#include "core/algorithm/strategy_alphazero.h"

#include "core/evaluator/dummy.h"
#include "game/shadow.h"

int main(int argc, const char** argv) {
  int NumIterations = 100000;
  if (argc == 3 && strcmp(argv[1], "-i") == 0) {
    NumIterations = std::atoi(argv[2]);
  }
  
  alphazero::Algorithm<Shadow::GameState, 0> algorithm;
  auto evaluator = DummyEvaluator(Shadow::NUM_PLAYERS, Shadow::NUM_ACTIONS);
  auto game = Shadow::GameState();
  auto context = algorithm.compute(game, evaluator);

  auto start = high_resolution_clock::now();
  context->step(/*iterations=*/NumIterations);
  auto best_move = context->best_move();
  auto best_value = context->best_value();
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  std::cout << "Time: " << duration << "ms\nIteration: " << NumIterations
            << "\nBest move: "
            << game.action_to_string(best_move)
            << "\nBest value: " << best_value << std::endl;
  return 0;
}
