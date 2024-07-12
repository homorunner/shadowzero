#include "core/algorithm/strategy_alphazero.h"
#include "core/evaluator/libtorch_queued.h"
#include "game/shadow.h"

#include "core/util/argh.h"

constexpr bool DEBUG_SHOW_ACTIONS_PER_TURN = false;
constexpr bool DEBUG_SHOW_GAMEBOARD = false;

constexpr int PLAYOUT_NUM = 2000;
constexpr int PLAYOUT_CAP_NUM = 180;
constexpr float PLAYOUT_CAP_PERCENT = 0.75f;
constexpr float TEMPERATURE_START = 1.0f;
constexpr float TEMPERATURE_END = 0.2f;
constexpr float TEMPERATURE_LAMBDA = -0.01f;

constexpr int WORKER_THREADS = 32;
constexpr int GPU_EVALUATOR_COUNT = 1;
constexpr int CPU_EVALUATOR_COUNT = 0;

using Game = Shadow::GameState;
using Algorithm = alphazero::Algorithm<Game, 0>;

auto init_rand() {
  std::uniform_real_distribution<float> rd(0, 1);
  std::default_random_engine re(std::random_device{}());
  return std::bind(rd, re);
}

int count_current_dataset(const char* output_dir) {
  for (int i = 0;; i++) {
    const auto pattern = std::format("_{:04d}_", i);
    bool found = false;
    for (const auto& entry : std::filesystem::directory_iterator(output_dir)) {
      if (entry.is_regular_file() && entry.path().string().find(pattern) != std::string::npos) {
        found = true;
      }
    }
    if (!found) {
      return i;
    }
  }
}

int main(int argc, const char** argv) {
  argh::parser cmd({"-m", "--model", "-o", "--output-dir", "-c", "--count"});
  cmd.parse(argc, argv);
  auto model = cmd({"-m", "--model"}).str();
  auto output_dir = cmd({"-o", "--output-dir"}).str();
  int gen_dataset_count;
  cmd({"-c", "--count"}, 1024) >> gen_dataset_count;
  if (model.empty() || output_dir.empty()) {
    std::cout << "Usage: " << argv[0] << " <model> <output_dir>" << std::endl;
    return 1;
  }

  auto rand = init_rand();
  c10::InferenceMode guard;
  Algorithm algorithm;
  QueuedLibtorchEvaluator* evaluators[GPU_EVALUATOR_COUNT + CPU_EVALUATOR_COUNT];
  for (int i = 0; i < CPU_EVALUATOR_COUNT; i++) {
    evaluators[i] = new QueuedLibtorchEvaluator(model, Shadow::CANONICAL_SHAPE,
                                                /*cpu_only=*/true);
  }
  for (int i = CPU_EVALUATOR_COUNT; i < CPU_EVALUATOR_COUNT + GPU_EVALUATOR_COUNT; i++) {
    evaluators[i] = new QueuedLibtorchEvaluator(model, Shadow::CANONICAL_SHAPE, /*cpu_only=*/false,
                                                /*device_id=*/i - CPU_EVALUATOR_COUNT);
  }

  if (!std::filesystem::exists(output_dir)) {
    std::filesystem::create_directories(output_dir);
  }
  std::atomic<int> dataset_id = count_current_dataset(output_dir.c_str());

  std::atomic<bool> stop = false;

  auto work = [&](int evaluator_id) {
    while (!stop) {
      Game game;
      float temperature = TEMPERATURE_START;
      int turn;
      std::vector<std::unique_ptr<Algorithm::Context>> contexts;
      int valid_move_count;

      for (turn = 0; !stop && !game.End(); turn++) {
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

        // capped is used in playout cap randomization
        bool capped = rand() < PLAYOUT_CAP_PERCENT;

        temperature = std::exp(TEMPERATURE_LAMBDA * turn) * (temperature - TEMPERATURE_END) + TEMPERATURE_END;

        auto context = algorithm.compute(game, *evaluators[evaluator_id]);
        context->step(capped ? PLAYOUT_CAP_NUM : PLAYOUT_NUM,
                      /*root_noise_enabled=*/!capped,
                      /*force_playout=*/!capped);
        auto action = context->select_move(temperature);

        if (action < 0 || action >= Shadow::NUM_ACTIONS || !valid_moves[action]) {
          std::cout << "Invalid move " << action << std::endl;
          stop = true;
          break;
        }

        game.Move(action);

        if (!capped && !context->is_ended_state()) {
          contexts.emplace_back(std::move(context));
        }

        if constexpr (DEBUG_SHOW_GAMEBOARD) {
          std::cout << game.ToString() << std::endl;
        }
      }

      if (stop) {
        break;
      }

      if (contexts.empty()) {
        std::cout << "No context to save." << std::endl;
        continue;
      }

      float score;

      if (valid_move_count == 0) {
        score = game.Current_player() == 0 ? 0.0f : 1.0f;
      } else {
        score = game.Score();
      }

      int n = contexts.size();
      const int kSymmetry = Shadow::NUM_SYMMETRIES;
      at::Tensor canonical = torch::zeros(
          {n * kSymmetry, Shadow::CANONICAL_SHAPE[0], Shadow::CANONICAL_SHAPE[1], Shadow::CANONICAL_SHAPE[2]},
          torch::kFloat);
      at::Tensor policy = torch::empty({n * kSymmetry, Shadow::NUM_ACTIONS}, torch::kFloat);
      at::Tensor values = torch::zeros({n * kSymmetry, 2}, torch::kFloat);
      for (int i = 0; i < n; i++) {
        auto& context = contexts[i];
        context->game->Canonicalize(canonical.mutable_data_ptr<float>() + i * kSymmetry * Shadow::CANONICAL_SHAPE[0] *
                                                                              Shadow::CANONICAL_SHAPE[1] *
                                                                              Shadow::CANONICAL_SHAPE[2]);
        context->mcts.set_probs(policy.mutable_data_ptr<float>() + i * kSymmetry * Shadow::NUM_ACTIONS,
                                /*temp=*/1.0f, /*prune_forced_count=*/true);
        values[i * kSymmetry][context->game->Current_player()] = score;
        values[i * kSymmetry][!context->game->Current_player()] = 1.0f - score;

        context->game->create_symmetry_boards(canonical[i * kSymmetry + 1].mutable_data_ptr<float>(),
                                              canonical[i * kSymmetry].mutable_data_ptr<float>());
        context->game->create_symmetry_actions(policy[i * kSymmetry + 1].mutable_data_ptr<float>(),
                                               policy[i * kSymmetry].mutable_data_ptr<float>());
        context->game->create_symmetry_values(values[i * kSymmetry + 1].mutable_data_ptr<float>(),
                                              values[i * kSymmetry].mutable_data_ptr<float>());
      }

      // check for possible nan
      if (torch::any(torch::isnan(canonical)).item<bool>() || torch::any(torch::isnan(policy)).item<bool>() ||
          torch::any(torch::isnan(values)).item<bool>()) {
        std::cout << "Nan detected, skip." << std::endl;
        continue;
      }

      int index = dataset_id.fetch_add(1);
      if (index >= gen_dataset_count) {
        stop = true;
      }

      std::cout << "Iteration " << index << ", score is " << score << std::endl;

      auto c_path = std::format("{}/c_{:04d}_{}.pt", output_dir, index, n);
      auto p_path = std::format("{}/p_{:04d}_{}.pt", output_dir, index, n);
      auto v_path = std::format("{}/v_{:04d}_{}.pt", output_dir, index, n);
      torch::pickle_save(canonical, c_path);
      torch::pickle_save(policy, p_path);
      torch::pickle_save(values, v_path);
    }
  };  // work

  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < WORKER_THREADS; thread_id++) {
    threads.emplace_back(work, thread_id % (GPU_EVALUATOR_COUNT + CPU_EVALUATOR_COUNT));
  }
  threads.emplace_back([&]() {
    while (!stop && dataset_id.load() < gen_dataset_count) {
      std::this_thread::sleep_for(std::chrono::seconds(10));
      for (int i = 0; i < CPU_EVALUATOR_COUNT + GPU_EVALUATOR_COUNT; i++) {
        std::cout << "Evaluator " << i << ": " << evaluators[i]->statistics() << std::endl;
      }
    }
  });
  for (auto& thread : threads) {
    thread.join();
  }

  return 0;
}
