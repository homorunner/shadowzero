#include "core/algorithm/strategy_alphazero.h"
#include "core/evaluator/libtorch_queued.h"
#include "core/util/io.h"
#include "game/shadow.h"

constexpr bool DEBUG_SHOW_ACTIONS_PER_TURN = false;

constexpr int ALPHAZERO_NUM_PLAYOUT = 5000;
constexpr float ALPHAZERO_TEMPERATURE_START = 0.5f;
constexpr float ALPHAZERO_TEMPERATURE_END = 0.2f;
constexpr float ALPHAZERO_TEMPERATURE_LAMBDA = -0.01f;

using Game = Shadow::GameState;
using Algorithm = alphazero::Algorithm<Game, 0>;

int main(int argc, const char** argv) {
  bool DEBUG_SHOW_GAMEBOARD = false;
  int THREAD_COUNT = 32;

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <model>" << std::endl;
    return 1;
  }

  auto model = argv[1];

  if (argc == 3 && std::string(argv[2]) == "--show-board") {
    DEBUG_SHOW_GAMEBOARD = true;
    THREAD_COUNT = 1;
  }

  Algorithm* zero[2];
  zero[0] = new Algorithm(3.0f, 0.25f);
  zero[1] = new Algorithm(3.0f, 0.25f);
  QueuedLibtorchEvaluator evaluator(model, Shadow::CANONICAL_SHAPE,
                                    /*use_cpu_only=*/false);

  float win_count[2] = {0};
  float total_score[2][2] = {0};
  int total_count[2][2] = {0};
  std::mutex mutex;
  std::vector<std::thread> threads;
  for (int t = 0; t < THREAD_COUNT; t++) {
    threads.emplace_back(
        [&](int index) {
          for (int round = index;; round++) {
            Game game;
            float temperature = ALPHAZERO_TEMPERATURE_START;
            int turn;
            bool first_play_evaluator = round % 2;
            int valid_move_count;

            for (turn = 0; !game.End(); turn++) {
              // check if no valid move
              auto valid_moves = game.Valid_moves();
              valid_move_count = 0;
              for (int i = 0; i < Shadow::NUM_ACTIONS; i++) {
                if (valid_moves[i]) {
                  valid_move_count++;
                }
              }
              if (valid_move_count == 0) {
                break;
              }

              // first 1-4 step is randomized
              if (turn <= rand() % 4) {
                int action_index = rand() % valid_move_count;
                for (int i = 0; i < Shadow::NUM_ACTIONS; i++) {
                  if (valid_moves[i]) {
                    if (action_index-- == 0) {
                      game.Move(i);
                      break;
                    }
                  }
                }

                if (DEBUG_SHOW_GAMEBOARD) {
                  std::cout << game.ToString() << std::endl;
                }
                continue;
              }

              temperature = std::exp(ALPHAZERO_TEMPERATURE_LAMBDA * turn) * (temperature - ALPHAZERO_TEMPERATURE_END) +
                            ALPHAZERO_TEMPERATURE_END;

              auto context = zero[game.Current_player() ^ first_play_evaluator]->compute(game, evaluator);

              context->step(ALPHAZERO_NUM_PLAYOUT, false);
              auto action = context->select_move(temperature);

              if (action < 0 || action >= Shadow::NUM_ACTIONS || !valid_moves[action]) {
                std::cout << "Invalid move " << action << std::endl;
                break;
              }

              game.Move(action);

              if constexpr (DEBUG_SHOW_ACTIONS_PER_TURN) {
                std::cout << "Turn " << turn << ", action=" << game.action_to_string(action) << std::endl;
              }
              if (DEBUG_SHOW_GAMEBOARD) {
                std::cout << game.ToString() << std::endl;
              }
            }

            float score;

            if (valid_move_count == 0) {
              score = game.Current_player() == 0 ? 0.0f : 1.0f;
            } else {
              score = game.Score();
            }

            if (score == 0.5f) {
              std::cout << "Draw" << std::endl;
            } else if (score > 0.5f) {
              std::cout << "Winner is O" << std::endl;
            } else {
              std::cout << "Winner is X" << std::endl;
            }

            mutex.lock();
            win_count[first_play_evaluator] += score;
            win_count[!first_play_evaluator] += 1 - score;
            if (first_play_evaluator == 0) {
              total_count[0][0]++;
              total_count[1][1]++;
              total_score[0][0] += score;
              total_score[1][1] += 1 - score;
            } else {
              total_count[0][1]++;
              total_count[1][0]++;
              total_score[0][1] += 1 - score;
              total_score[1][0] += score;
            }
            mutex.unlock();

            std::cout << "Win count: " << win_count[0] << " - " << win_count[1] << std::endl;
          }
        },
        t);
  }
  for (auto& t : threads) {
    t.join();
  }

  return 0;
}
