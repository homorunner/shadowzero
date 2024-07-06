#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

std::string readStringFromFile(const std::string& filename) {
  std::ifstream fin(filename);
  if (!fin) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return "";
  }
  std::string line;
  std::getline(fin, line);
  return line;
}

void writeStringToFile(const std::string& filename, const std::string& str) {
  std::ofstream fout(filename, std::ios::trunc);
  if (!fout) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  fout << str;
}

void dumpGame(const char* filename,
              const std::vector<std::string>& history_moves) {
  std::ofstream ofs(filename, std::ios::trunc);
  if (!ofs.is_open()) {
    std::cerr << "Failed to open " << filename << " to write, dump failed."
              << std::endl;
    return;
  }
  for (auto i : history_moves) {
    ofs << i << '\n';
  }
  ofs.close();
}

template <class Game, class ToActionFn>
std::shared_ptr<Game> loadGame(
    const char* filename, std::vector<std::string>& history_moves,
    std::vector<std::shared_ptr<Game>>& history,
    ToActionFn string_to_action) {
  auto game = std::make_shared<Game>();
  history.clear();
  history_moves.clear();
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open " << filename << " to read, load failed."
              << std::endl;
    return game;
  }
  std::string move;
  while (ifs >> move) {
    int action = string_to_action(game.get(), move);
    if (action < 0 || action >= game->Num_actions()) {
      std::cerr << "Warning: invalid action, skipped." << std::endl;
      continue;
    }
    if (!game->Valid_moves()[action]) {
      std::cerr << "Warning: invalid move, skipped." << std::endl;
      continue;
    }
    history.push_back(game->Copy());
    history_moves.push_back(move);
    game->Move(action);
  }
  return game;
}
