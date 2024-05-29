#include "core/algorithm/strategy_alphazero.h"
#include "core/evaluator/libtorch_queued.h"
#include "core/util/io.h"
#include "game/shadow.h"

void interactive(const char* model) {
  alphazero::Algorithm<Shadow::GameState, 3> algorithm;
  QueuedLibtorchEvaluator evaluator(model, Shadow::CANONICAL_SHAPE);
  auto game = std::make_shared<Shadow::GameState>();

  std::vector<std::shared_ptr<Shadow::GameState>> history;
  std::vector<std::string> history_moves;
  while (true) {
  start:;
    std::cout << game->ToString() << "\n";
    auto valid_moves = game->Valid_moves();
    for (int i = 0; i < game->Num_actions(); i++) {
      if (valid_moves[i]) {
        std::cout << game->action_to_string(i) << " ";
      }
    }
    Shadow::ActionType action;
    auto context = algorithm.compute(*game, evaluator);
    while (true) {
      std::cout << "\nInput action (y to think): ";
      std::string move;
      std::cin >> move;
      if (move == "y" || move == "Y") {
        std::atomic<bool> stop(false);

        std::thread t([&]() {
          int iter = 0;
          puts("Thinking...\n\n\n");
          for (; !stop.load(); iter++) {
            context->step(64, /*root_noise_enabled=*/true);
            context->show_actions(5, /*move_up_cursor=*/true);
          }
        });

        std::getchar();
        std::getchar();
        printf("\33[F");
        stop = true;
        t.join();
      } else if (move == "b" || move == "B") {
        game = history.back();
        history.pop_back();
        history_moves.pop_back();
        goto start;
      } else if (move == "save" || move == "dump") {
        dumpGame("game.txt", history_moves);
        continue;
      } else if (move == "load") {
        game = loadGame("game.txt", history_moves, history,
                        std::bind(&Shadow::GameState::string_to_action,
                                  game.get(), std::placeholders::_1));
        goto start;
      } else {
        action = game->string_to_action(move);
        if (action < 0 || action >= game->Num_actions()) {
          continue;
        }
        if (action != Shadow::MOVE_PASS && !game->Valid_moves()[action]) {
          std::cerr << "Invalid move." << std::endl;
          continue;
        }
        history_moves.push_back(move);
        break;
      }
    }
    history.push_back(game->Copy());
    game->Move(action);
  }
}

int main(int argc, const char** argv) {
  if (argc >= 2) {
    interactive(argv[1]);
  } else {
    interactive("");
  }
  return 0;
}
