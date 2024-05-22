#pragma once

#include "core/util/common.h"

// An example evaluator that does nothing
class DummyEvaluator : public EvaluatorBase {
 public:
  DummyEvaluator(int v_size_, int pi_size_)
      : v_size(v_size_), pi_size(pi_size_) {}
  void evaluate(std::function<void(float*)> canonicalize,
                std::function<void(const float*, const float*)> process_result,
                uint64_t hashval = 0) {
    auto v = std::make_unique<float[]>(v_size);
    auto pi = std::make_unique<float[]>(pi_size);
    for (int i = 0; i < v_size; i++) v[i] = 1.0 / v_size;
    for (int i = 0; i < pi_size; i++) pi[i] = 1.0 / pi_size;
    process_result(v.get(), pi.get());
  }
  void evaluateN(
      int N, std::function<void(float*)>* games,
      std::function<void(const float*, const float*)>* process_results) {
    for (int i = 0; i < N; i++) {
      evaluate(games[i], process_results[i]);
    }
  }

 private:
  int v_size;
  int pi_size;
};