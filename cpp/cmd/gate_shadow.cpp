#include "core/algorithm/strategy_alphazero.h"
#include "core/evaluator/libtorch_queued.h"
#include "core/util/io.h"
#include "game/shadow.h"

constexpr bool DEBUG_SHOW_ACTIONS_PER_TURN = false;

constexpr int ALPHAZERO_NUM_PLAYOUT = 300;
constexpr float ALPHAZERO_TEMPERATURE_START = 0.5f;
constexpr float ALPHAZERO_TEMPERATURE_END = 0.2f;
constexpr float ALPHAZERO_TEMPERATURE_LAMBDA = -0.01f;

using Game = Shadow::GameState;
using Algorithm = alphazero::Algorithm<Game, 0>;

int main(int argc, const char** argv) {
  bool DEBUG_SHOW_GAMEBOARD = false;
  bool OUTPUT_BEST = false;
  bool OUTPUT_DATA = false;
  bool USE_TWO_GPU = false;
  int THREAD_COUNT = 32;
  std::string OUTPUT_BEST_FILE = "best_model.txt";
  std::string OUTPUT_DATA_FILE = "gating_data.txt";

  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <number_iteration> <model1> <model2>"
              << std::endl;
    return 1;
  }

  int GATING_TOTAL_ROUND = std::stoi(argv[1]);
  int GATING_AT_LEAST_WIN = (GATING_TOTAL_ROUND + 1) / 2;

  auto model_left = argv[2], model_right = argv[3];

  if (argc == 5 && std::string(argv[4]) == "--show-board") {
    DEBUG_SHOW_GAMEBOARD = true;
    THREAD_COUNT = 1;
  }
  if (argc >= 5 && std::string(argv[4]) == "--output-best") {
    OUTPUT_BEST = true;
    if (argc >= 6) {
      OUTPUT_BEST_FILE = argv[5];
    }
  }
  if (argc >= 5 && std::string(argv[4]) == "--output-data") {
    OUTPUT_DATA = true;
    if (argc >= 6) {
      OUTPUT_DATA_FILE = argv[5];
    }
  } else if (argc >= 7 && std::string(argv[6]) == "--output-data") {
    OUTPUT_DATA = true;
    if (argc >= 8) {
      OUTPUT_DATA_FILE = argv[7];
    }
  }

  if (torch::cuda::is_available() && torch::cuda::device_count() >= 2) {
    USE_TWO_GPU = true;
  }

  Algorithm zero;
  QueuedLibtorchEvaluator* evaluator[2];
  evaluator[0] =
      new QueuedLibtorchEvaluator(model_left, Shadow::CANONICAL_SHAPE,
                                  /*use_cpu_only=*/false);
  evaluator[1] =
      new QueuedLibtorchEvaluator(model_right, Shadow::CANONICAL_SHAPE,
                                  /*use_cpu_only=*/false,
                                  /*device_id=*/USE_TWO_GPU ? 1 : 0);

  float win_count[2] = {0};
  float total_score[2][2] = {0};
  int total_count[2][2] = {0};
  std::mutex mutex;
  std::vector<std::thread> threads;
  for (int t = 0; t < THREAD_COUNT; t++) {
    threads.emplace_back(
        [&](int index) {
          for (int round = index; round < GATING_TOTAL_ROUND;
               round += THREAD_COUNT) {
            if (win_count[0] >= GATING_AT_LEAST_WIN ||
                win_count[1] >= GATING_AT_LEAST_WIN) {
              break;
            }

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

              temperature = std::exp(ALPHAZERO_TEMPERATURE_LAMBDA * turn) *
                                (temperature - ALPHAZERO_TEMPERATURE_END) +
                            ALPHAZERO_TEMPERATURE_END;

              auto context = zero.compute(
                  game,
                  *evaluator[game.Current_player() ^ first_play_evaluator]);
              context->step(ALPHAZERO_NUM_PLAYOUT, false);
              auto action = context->select_move(temperature);

              if (action < 0 || action >= Shadow::NUM_ACTIONS ||
                  !valid_moves[action]) {
                std::cout << "Invalid move " << action << std::endl;
                break;
              }

              game.Move(action);

              if constexpr (DEBUG_SHOW_ACTIONS_PER_TURN) {
                std::cout << "Turn " << turn
                          << ", action=" << game.action_to_string(action)
                          << std::endl;
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

            std::cout << "Win count: " << win_count[0] << " - " << win_count[1]
                      << std::endl;
          }
        },
        t);
  }
  for (auto& t : threads) {
    t.join();
  }

  if (OUTPUT_BEST) {
    writeStringToFile(OUTPUT_BEST_FILE, win_count[0] >= GATING_AT_LEAST_WIN
                                            ? model_left
                                            : model_right);
  }

  if (OUTPUT_DATA) {
    auto data = std::format(
        "[[model]]\npath = \"{}\"\nfirstplay_count = {}\nfirstplay_score = "
        "{}\nsecondplay_count = {}\nsecondplay_score = {}\n"
        "[[model]]\npath = \"{}\"\nfirstplay_count = {}\nfirstplay_score = "
        "{}\nsecondplay_count = {}\nsecondplay_score = {}\n",
        model_left, total_count[0][0], total_score[0][0], total_count[0][1],
        total_score[0][1], model_right, total_count[1][0], total_score[1][0],
        total_count[1][1], total_score[1][1]);
    writeStringToFile(OUTPUT_DATA_FILE, data);
  }

  return 0;
}
