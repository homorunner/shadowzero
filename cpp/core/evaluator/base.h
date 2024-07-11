#pragma once

#include "core/util/common.h"

// Evaluator interface
class EvaluatorBase {
 public:
  virtual void evaluate(std::function<void(float*)> canonicalize,
                        std::function<void(const float*, const float*)> process_result, uint64_t hashval) = 0;
  virtual void evaluateN(int N, std::function<void(float*)>* games,
                         std::function<void(const float*, const float*)>* process_results) = 0;
};