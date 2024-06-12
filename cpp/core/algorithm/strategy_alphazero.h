#pragma once

#include "core/evaluator/base.h"
#include "core/util/common.h"

namespace alphazero {

/* NOISE_ALPHA_RATIO needs to be changed across different games.
   Related papers:
   [1] Domain Knowledge in Exploration Noise in AlphaZero.
       Eric W., George D., Aaron T., Abtin M.
   [2] https://medium.com/oracledevs/lessons-from-alpha-zero-part
       -6-hyperparameter-tuning-b1cfcbe4ca9a
*/
const float NOISE_ALPHA_RATIO = 10.83f;

const float CPUCT = 1.25;
const float FPU_REDUCTION = 0.25;

struct ValueType {
  // v should be the winrate/score for player 0.
  // In this project, we only consider two-player zero sum games.
  // e.g. value(0) + value(1) = 1.
  float v;
  ValueType() = default;
  explicit ValueType(float v_) : v(v_) {}
  explicit ValueType(int index, float v_) { set(index, v_); }
  explicit ValueType(bool index, float v_) { set(index, v_); }
  explicit ValueType(float v0, float v1) { v = v0 / (v0 + v1); }
  float operator()(int index) const { return get(index); }
  float get(int index) const {
    assert(index == 0 || index == 1);
    return index == 0 ? v : 1 - v;
  }
  template <class T0, class T1>
  void set(T0 a, T1 b);
  void set(int index, float v_) {
    assert(index == 0 || index == 1);
    v = index == 0 ? v_ : 1 - v_;
  }
  void set(bool index, float v_) { v = index == 0 ? v_ : 1 - v_; }
  void set(float v0, float v1) { v = v0 / (v0 + v1); }
};

struct Node {
  Node() = default;
  explicit Node(int m) : move(m) {}

  float q = 0;
  float v = 0;
  float policy = 0;
  int move = 0;
  int n = 0;
  bool player = 0;
  bool ended = false;
  ValueType value;
  std::vector<Node> children{};

  void add_children(const std::vector<uint8_t>& valids) noexcept {
    for (size_t w = 0; w < valids.size(); ++w) {
      if (valids[w]) {
        children.emplace_back(w);
      }
    }
    static auto rd = std::default_random_engine(std::time(0));
    std::shuffle(children.begin(), children.end(), rd);
  }
  size_t size() const noexcept { return children.size(); }
  void update_policy(const std::vector<float>& pi) noexcept {
    for (auto& c : children) {
      c.policy = pi[c.move];
    }
  }
  float uct(float sqrt_parent_n, float cpuct, float fpu_value) const noexcept {
    return (n == 0 ? fpu_value : q) + cpuct * policy * sqrt_parent_n / (n + 1);
  }
  Node* best_child(float cpuct, float fpu_reduction,
                   bool force_playout) noexcept {
    float seen_policy = 0.0f;
    for (auto& c : children) {
      if (c.n > 0) {
        if (force_playout && c.n < std::sqrt(2 * c.policy * (this->n - c.n))) {
          return &c;
        }
        seen_policy += c.policy;
      }
    }
    auto fpu_value = v - fpu_reduction * std::sqrt(seen_policy);
    auto sqrt_n = std::sqrt((float)n);
    auto best_i = 0;
    auto best_uct = children[0].uct(sqrt_n, cpuct, fpu_value);
    for (size_t i = 1; i < children.size(); ++i) {
      auto uct = children[i].uct(sqrt_n, cpuct, fpu_value);
      if (uct > best_uct) {
        best_uct = uct;
        best_i = i;
      }
    }
    return &children[best_i];
  }
};

template <class GameState>
class MCTS {
 public:
  MCTS(float cpuct, int num_moves, float epsilon = 0,
       float root_policy_temp = 1.4, float fpu_reduction = 0)
      : cpuct_(cpuct),
        num_moves_(num_moves),
        current_(&root_),
        epsilon_(epsilon),
        root_policy_temp_(root_policy_temp),
        fpu_reduction_(fpu_reduction) {}

  std::unique_ptr<GameState> find_leaf(const GameState& gs,
                                       bool force_playout = false) {
    current_ = &root_;
    auto leaf = gs.Copy();

    while (current_->n > 0 && !current_->ended) {
      path_.push_back(current_);
      auto fpu_reduction = fpu_reduction_;
      // root fpu is half-ed.
      if (current_ == &root_) {
        fpu_reduction /= 2;
      }
      // fpu of failing node is half-ed.
      if (current_->n > 0 && current_->v < 0.2) {
        fpu_reduction /= 2;
      }
      current_ = current_->best_child(cpuct_, fpu_reduction, force_playout);

      leaf->Move(current_->move);
    }

    if (current_->n == 0 && !current_->ended) {
      current_->player = leaf->Current_player();
      current_->ended = leaf->End();
      if (current_->ended) {
        current_->value = ValueType(leaf->Score());
      } else {
        current_->add_children(leaf->Valid_moves());
        if (!current_->children.size()) {
          current_->ended = true;
          current_->value = ValueType(leaf->Current_player(),
                                      0);  // player with no valid moves loses.
        }
      }
    }
    return leaf;
  }

  void process_result(const float* pi, size_t size_pi, const float* v,
                      bool root_noise_enabled = false) {
    ValueType value(current_->value);

    if (!current_->ended) {
      value = ValueType(v[current_->player], v[!current_->player]);
      // Rescale pi based on valid moves.
      std::vector<float> scaled(size_pi, 0);
      float sum = 0;
      for (auto& c : current_->children) {
        sum += pi[c.move];
      }
      for (auto& c : current_->children) {
        scaled[c.move] = pi[c.move] / sum;
      }
      if (current_ == &root_) {
        sum = 0;
        for (auto& c : current_->children) {
          scaled[c.move] = std::pow(scaled[c.move], 1.0 / root_policy_temp_);
          sum += scaled[c.move];
        }
        for (auto& c : current_->children) {
          scaled[c.move] = scaled[c.move] / sum;
        }
        current_->update_policy(scaled);
        if (root_noise_enabled) {
          add_root_noise();
        }
      } else {
        current_->update_policy(scaled);
      }
    }

    while (!path_.empty()) {
      // TODO: update parent->values when all subtree have been visited.
      auto* parent = path_.back();
      path_.pop_back();
      auto v = value(parent->player);
      current_->q = (current_->q * current_->n + v) / (current_->n + 1);
      if (current_->n == 0) {
        current_->v = value(current_->player);
      }
      ++current_->n;
      current_ = parent;
    }
    ++depth_;
    ++root_.n;
  }

  void add_root_noise() {
    auto dist =
        std::gamma_distribution<float>{NOISE_ALPHA_RATIO / root_.size(), 1.0};
    static auto re = std::default_random_engine(std::time(0));
    std::vector<float> noise(num_moves_, 0);
    float sum = 0;
    for (auto& c : root_.children) {
      noise[c.move] = dist(re);
      sum += noise[c.move];
    }
    for (auto& c : root_.children) {
      c.policy = c.policy * (1 - epsilon_) + epsilon_ * noise[c.move] / sum;
    }
  }

  float root_value() const {
    float q = 0;
    for (const auto& c : root_.children) {
      if (c.n > 0 && c.q > q) {
        q = c.q;
      }
    }
    return q;
  }

  std::vector<Node>& root_children() noexcept { return root_.children; }

  std::vector<int> counts() const noexcept {
    std::vector<int> result(num_moves_, 0);
    for (const auto& c : root_.children) {
      if (c.n > 0) {
        result[c.move] = c.n;
      }
    }
    return result;
  }

  std::vector<int> policy_pruned_counts() const noexcept {
    std::vector<int> result(num_moves_, 0);
    const Node* best_child = nullptr;
    int best_child_visit = 0;
    float sqrt_root_n = std::sqrt((float)root_.n);

    for (auto& c : root_.children) {
      if (c.n > best_child_visit) {
        best_child_visit = c.n;
        best_child = &c;
      }
    }

    if (best_child == nullptr) {
      return result;
    }

    float best_child_uct = best_child->uct(sqrt_root_n, cpuct_, fpu_reduction_);

    for (auto& c : root_.children) {
      if (c.n > 0) {
        if (&c == best_child) {
          result[c.move] = c.n;
        } else {
          int visits = c.n;

          int lower_bound = std::ceil(cpuct_ * c.policy * sqrt_root_n /
                                      (best_child_uct - c.q));

          result[c.move] = std::min(visits, lower_bound);

          // prune visit count equal to 1
          if (result[c.move] <= 1) {
            result[c.move] = 0;
          }
        }
      }
    }

    return result;
  }

  std::vector<float> probs(float temp) const noexcept {
    std::vector<float> probs(num_moves_, 0);
    set_probs(probs.data(), temp);
    return probs;
  }

  void set_probs(float* buffer, float temp,
                 bool prune_forced_count = false) const noexcept {
    auto counts =
        prune_forced_count ? this->policy_pruned_counts() : this->counts();

    if (temp < 1e-7f) {
      auto best_moves = std::vector<int>{0};
      auto best_count = counts[0];
      for (auto m = 1; m < num_moves_; ++m) {
        if (counts[m] > best_count) {
          best_count = counts[m];
          best_moves = {m};
        } else if (counts[m] == best_count) {
          best_moves.push_back(m);
        }
      }

      memset(buffer, 0, num_moves_ * sizeof(float));
      for (auto m : best_moves) {
        buffer[m] = 1.0f / best_moves.size();
      }
    } else {
      float sum = 0;
      for (int i = 0; i < num_moves_; i++) {
        sum += counts[i];
      }
      for (int i = 0; i < num_moves_; i++) {
        buffer[i] = counts[i] / sum;
      }

      if (temp != 1.0f) {
        sum = 0;
        for (int i = 0; i < num_moves_; i++) {
          sum += std::pow(buffer[i], 1.0f / temp);
        }
        for (int i = 0; i < num_moves_; i++) {
          buffer[i] = std::pow(buffer[i], 1.0f / temp) / sum;
        }
      }
    }
  }

  int depth() const noexcept { return depth_; }

  static int pick_move(const std::vector<float>& p) {
    std::uniform_real_distribution<float> dist{0.0F, 1.0F};
    static std::random_device rd{};
    static std::mt19937 re{rd()};
    auto choice = dist(re);
    auto sum = 0.0f;
    for (size_t m = 0; m < p.size(); ++m) {
      sum += p[m];
      if (sum > choice) {
        return m;
      }
    }
    // Due to floating point error we didn't pick a move.
    // Pick the last valid move.
    for (size_t m = p.size() - 1; m >= 0; --m) {
      if (p[m] > 0) {
        return m;
      }
    }
    assert(false);
    __builtin_unreachable();
  }

 private:
  float cpuct_;
  int num_moves_;

  int depth_ = 0;

 public:
  Node root_ = Node{};
  Node* current_;

 private:
  std::vector<Node*> path_{};
  float epsilon_;
  float root_policy_temp_;
  float fpu_reduction_;
};

template <class GameState, int SpecThreadCount>
class Algorithm {
 public:
  Algorithm() {}

  struct Context {
    Context(std::unique_ptr<GameState> game_, EvaluatorBase* evaluator_)
        : game(std::move(game_)),
          evaluator(evaluator_),
          mcts(
              /*cpuct=*/CPUCT,
              /*num_moves=*/game->Num_actions(),
              /*epsilon=*/0.25f,
              /*root_policy_temp=*/1.4f,
              /*fpu_reduction=*/FPU_REDUCTION) {
      for (int i = 0; i < SpecThreadCount; ++i) {
        specs[i] = std::make_unique<MCTS<GameState>>(
            /*cpuct=*/CPUCT,
            /*num_moves=*/game->Num_actions(),
            /*epsilon=*/0.25f,
            /*root_policy_temp=*/1.4f,
            /*fpu_reduction=*/FPU_REDUCTION);
      }
    }

    void step(int iterations, bool root_noise_enabled = false,
              bool force_playout = false) {
      if constexpr (SpecThreadCount == 0) {
        step_singlespec(iterations, root_noise_enabled, force_playout);
      } else {
        step_multispec(iterations, root_noise_enabled);
      }
    }

    void step_singlespec(int iterations, bool root_noise_enabled,
                         bool force_playout) {
      for (int iter = 0; iter < iterations; iter++) {
        auto leaf = mcts.find_leaf(*game, force_playout);

        if (mcts.current_->ended) {
          mcts.process_result(nullptr, 0, nullptr, root_noise_enabled);
          continue;
        }

        evaluator->evaluate(
            std::bind(&GameState::Canonicalize, *leaf, std::placeholders::_1),
            std::bind(&MCTS<GameState>::process_result, &mcts,
                      std::placeholders::_1, game->Num_actions(),
                      std::placeholders::_2, root_noise_enabled),
            leaf->Hash());
      }
    }

    void step_multispec(int iterations, bool root_noise_enabled) {
      // initalize spec trees with most p-value moves.
      if constexpr (SpecThreadCount > 0) {
        if (!spec_initialized) {
          mcts.find_leaf(*game);
          evaluator->evaluate(
              std::bind(&GameState::Canonicalize, *game, std::placeholders::_1),
              [this](const float* pi, const float* v) {
                auto& children = mcts.root_children();
                int count = children.size();
                std::vector<int> idx(count);
                for (int i = 0; i < count; i++) {
                  idx[i] = children[i].move;
                }
                std::sort(idx.begin(), idx.end(),
                          [&](int a, int b) { return pi[a] > pi[b]; });
                count = std::min(count - 1, SpecThreadCount);
                children.erase(
                    std::remove_if(
                        children.begin(), children.end(),
                        [&idx, count](const alphazero::Node& node) {
                          return std::find(idx.begin(), idx.begin() + count,
                                           node.move) != idx.begin() + count;
                        }),
                    children.end());
                mcts.process_result(pi, game->Num_actions(), v);
                for (int i = 0; i < count; i++) {
                  specs[i]->root_children().emplace_back(idx[i]);
                  specs[i]->root_.player = game->Current_player();
                  specs[i]->process_result(pi, game->Num_actions(), v);
                }
              },
              game->Hash());
          spec_initialized = true;
        }
      }

      std::array<std::unique_ptr<GameState>, SpecThreadCount + 1> leaves;
      std::array<std::atomic<bool>, SpecThreadCount> ins;
      std::vector<std::thread> threads;
      int specCount = 0;
      for (int i = 0; i < SpecThreadCount; i++) {
        if (specs[i]->root_.children.empty()) {
          continue;
        }
        specCount++;
        threads.push_back(std::thread(
            [&](int i) {
              for (int iter = 0; iter < iterations; iter++) {
                ins[i].wait(false);
                leaves[i] = specs[i]->find_leaf(*game);
                ins[i].store(false);
                ins[i].notify_one();
              }
            },
            i));
      }

      for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < specCount; i++) {
          ins[i].store(true);
          ins[i].notify_one();
        }
        leaves[specCount] = mcts.find_leaf(*game);
        for (int i = 0; i < specCount; i++) {
          ins[i].wait(true);
        }

        std::array<std::function<void(const float*, const float*)>,
                   SpecThreadCount + 1>
            process_results;
        std::array<std::function<void(float*)>, SpecThreadCount + 1>
            canonicalizes;
        for (int i = 0; i < specCount; i++) {
          canonicalizes[i] = std::bind(&GameState::Canonicalize, *leaves[i],
                                       std::placeholders::_1);
          process_results[i] =
              std::bind(&MCTS<GameState>::process_result, specs[i].get(),
                        std::placeholders::_1, game->Num_actions(),
                        std::placeholders::_2, false);
        }
        canonicalizes[specCount] =
            std::bind(&GameState::Canonicalize, *leaves[specCount],
                      std::placeholders::_1);
        process_results[specCount] = std::bind(
            &MCTS<GameState>::process_result, &mcts, std::placeholders::_1,
            game->Num_actions(), std::placeholders::_2, root_noise_enabled);
        evaluator->evaluateN(specCount + 1, canonicalizes.data(),
                             process_results.data());
      }

      for (auto& t : threads) {
        t.join();
      }
    }

    void show_actions(int show_count, bool move_up_cursor,
                      bool prune_forced_count = false) {
      int specs_count = 0;
      for (int i = 0; i < SpecThreadCount; i++) {
        if (specs[i]->root_.children.empty()) {
          break;
        }
        specs_count++;
      }
      if (move_up_cursor) {
        int lines_count = specs_count + 1 + (mcts.root_children().size() > 1);
        printf("\33[%dF", lines_count);
      }
      for (int i = 0; i < specs_count; i++) {
        auto counts =
            inversed_map(prune_forced_count ? specs[i]->policy_pruned_counts()
                                            : specs[i]->counts());
        auto root_value = specs[i]->root_value();
        printf("Action: [%s], Winrate:  %.4f\n",
               game->action_to_string(counts.begin()->second).c_str(),
               root_value);
      }
      auto counts = inversed_map(
          prune_forced_count ? mcts.policy_pruned_counts() : mcts.counts());
      printf("Action: ");
      if (counts.size() > 1) putchar('\n');
      for (auto iter = counts.rbegin(); show_count && iter != counts.rend();
           iter++, show_count--) {
        printf("[%s]: %d , ", game->action_to_string(iter->second).c_str(),
               iter->first);
      }
      auto root_value = mcts.root_value();
      printf("Winrate: %.4f\n", root_value);
    }

    int best_move() {
      float best_value = 0;
      int best_action = 0;
      for (const auto& c : mcts.root_.children) {
        if (c.n > 0 && c.q > best_value) {
          best_value = c.q;
          best_action = c.move;
        }
      }
      for (auto& spec : specs) {
        for (const auto& c : spec->root_.children) {
          if (c.n > 0 && c.q > best_value) {
            best_value = c.q;
            best_action = c.move;
          }
        }
      }
      return best_action;
    }

    float best_value() {
      float best_value = 0;
      for (const auto& c : mcts.root_.children) {
        if (c.n > 0 && c.q > best_value) {
          best_value = c.q;
        }
      }
      for (auto& spec : specs) {
        for (const auto& c : spec->root_.children) {
          if (c.n > 0 && c.q > best_value) {
            best_value = c.q;
          }
        }
      }
      return best_value;
    }

    int select_move(float temperature) {
      auto probs = mcts.probs(temperature);
      return MCTS<GameState>::pick_move(probs);
    }

    std::unique_ptr<GameState> game;
    // Algorithm *base; // currently not used variable, commented
    EvaluatorBase* evaluator;
    MCTS<GameState> mcts;
    std::array<std::unique_ptr<MCTS<GameState>>, SpecThreadCount> specs;
    bool spec_initialized = false;
  };

  std::unique_ptr<Context> compute(const GameState& game,
                                   EvaluatorBase& evaluator) {
    auto context = std::make_unique<Context>(game.Copy(), &evaluator);
    // context->base = this;
    return context;
  }
};

}  // namespace alphazero
