#include "core/algorithm/strategy_alphazero.h"
#include "core/evaluator/libtorch_queued.h"
#include "core/util/io.h"
#include "game/connect4.h"

void interactive(const char* model) {
  alphazero::Algorithm<Connect4::GameState, 3> algorithm;
  QueuedLibtorchEvaluator evaluator(model, Connect4::CANONICAL_SHAPE);
  auto game = std::make_shared<Connect4::GameState>();

  std::vector<std::shared_ptr<Connect4::GameState>> history;
  std::vector<std::string> history_moves;
  while (true) {
  start:;
    std::cout << game->ToString() << "\n";
    auto valid_moves = game->Valid_moves();
    Connect4::ActionType action;
    auto context = algorithm.compute(*game, evaluator);
    while (true) {
      std::cout << "\nInput action (y to think): ";
      std::string move;
      std::cin >> move;
      if (move == "y" || move == "Y") {
        std::atomic<bool> stop(false);

        std::thread t([&]() {
          int iter = 0;
          puts("Thinking...");
          for (; !stop.load(); iter++) {
            context->step(320, /*root_noise_enabled=*/true);
            context->show_actions(5, /*move_up_cursor=*/!!iter);
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
                        std::bind(&Connect4::GameState::string_to_action, std::placeholders::_1, std::placeholders::_2));
        goto start;
      } else {
        bool check = true;
        if (move.back() == '!') {
          check = false;
          move.pop_back();
        }
        action = game->string_to_action(move);
        if (action < 0 || action >= game->Num_actions()) {
          continue;
        }
        if (check && action != Connect4::MOVE_PASS && !game->Valid_moves()[action]) {
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
