#pragma once

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using namespace std::chrono_literals;  // for 1s, 10ms, etc.

int randn(int n) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  return std::uniform_int_distribution<int>(0, n - 1)(gen);
}

std::multimap<int, int> inversed_map(const std::vector<int>& vec) {
  std::multimap<int, int> result;
  for (int i = (int)vec.size() - 1; i >= 0; i--) {
    if (vec[i] != 0) {
      result.insert({vec[i], i});
    }
  }
  return result;
}
