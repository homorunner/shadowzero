#pragma once

#include "core/util/common.h"

// this is an implementation of N,M,k game
namespace Connect4 {

constexpr const int N = 5;
constexpr const int K = 4;

constexpr const int NUM_ACTIONS = 25;
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 2;

constexpr const std::array<int, 3> CANONICAL_SHAPE = {N, N + 1, N * 2};

using ActionType = int;
const ActionType MOVE_PASS = -1;

class GameState {
 private:
  bool current_player;
  int8_t round;
  int8_t piece[N][N][N][2];

 public:
  GameState() {
    current_player = 0;
    round = 0;
    memset(piece, 0, sizeof(piece));

    // force this opening
    Move(string_to_action("c3"));
    Move(string_to_action("b2"));
  }

  std::string action_to_string(const ActionType action) {
    if (action == MOVE_PASS) return "pass";
    return std::string(1, 'a' + action % N) + (char)('1' + action / N);
  }

  ActionType string_to_action(const std::string& action) {
    if (action == "pass") return MOVE_PASS;
    return (action[0] - 'a') + (action[1] - '1') * N;
  }

  std::unique_ptr<GameState> Copy() const {
    return std::make_unique<GameState>(*this);
  }

  uint64_t Hash() const noexcept {
    // not implemented
    return 0;
  }

  bool Current_player() const noexcept { return current_player; }

  int Num_actions() const noexcept { return NUM_ACTIONS; }

  std::vector<uint8_t> Valid_moves() const {
    auto valids = std::vector<uint8_t>(NUM_ACTIONS, 0);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        if (piece[N - 1][i][j][0] == 0 && piece[N - 1][i][j][1] == 0) {
          valids[i * N + j] = 1;
        }
      }
    }
    return valids;
  }

  void Move(ActionType action) {
    if (action == MOVE_PASS) {
      current_player = !current_player;
      round += 1;
      return;
    }

    int x = action / N;
    int y = action % N;
    for (int i = 0; i < N; i++) {
      if (piece[i][x][y][0] == 0 && piece[i][x][y][1] == 0) {
        piece[i][x][y][current_player] = 1;
        break;
      }
    }

    current_player = !current_player;
    round += 1;
  }

  int winner() const {
    if (round == N * N * N)
      return 1;  // if the board is full, we consider the second player as
                 // winner

    // TODO: maybe we only check around the last move.
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dz = -1; dz <= 1; dz++) {
          if (!dx && !dy && !dz) continue;
          for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
              for (int k = 0; k < N; k++) {
                if (i + dx * (K - 1) < 0 || i + dx * (K - 1) >= N) continue;
                if (j + dy * (K - 1) < 0 || j + dy * (K - 1) >= N) continue;
                if (k + dz * (K - 1) < 0 || k + dz * (K - 1) >= N) continue;

                int count[2] = {0, 0};
                for (int l = 0; l < K; l++) {
                  count[0] += piece[i + dx * l][j + dy * l][k + dz * l][0];
                  count[1] += piece[i + dx * l][j + dy * l][k + dz * l][1];
                }

                if (count[0] == K) return 0;
                if (count[1] == K) return 1;
              }
            }
          }
        }
      }
    }

    return -1;
  }

  bool End() const { return winner() != -1; }

  bool Winner() const {
    assert(End());
    return winner();
  }

  float Score() const {
    assert(End());
    return 1.0 - winner();
  }

  void Canonicalize(float* storage) const noexcept {
    // if we update this function (e.g. add more channels), we need to update
    // this variable
    assert(CANONICAL_SHAPE[0] == N && CANONICAL_SHAPE[1] == N + 1 &&
           CANONICAL_SHAPE[2] == N * 2);

    float(&out)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]] =
        *reinterpret_cast<float(*)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]]
                                  [CANONICAL_SHAPE[2]]>(storage);

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          out[i][j][k] = piece[i][j][k][0];
          out[i][j][k + N] = piece[i][j][k][1];
        }
      }
      for (int k = 0; k < N+N; k++) {
        out[i][N][k] = current_player;
      }
    }
  }

  std::string ToString() const {
    std::string out;

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        out += (char)('a' + j);
      }
      out += '\n';
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          out += piece[i][j][k][0] ? 'X' : (piece[i][j][k][1] ? 'O' : '.');
        }
        out += ' ';
        out += (char)('1' + j);
        out += '\n';
      }
      out += '\n';
    }

    out += "Player: ";
    out += (current_player ? 'X' : 'O');

    return out;
  }

  void create_symmetry_values(float* dst, const float* src) const {
    for (int i = 0; i < NUM_SYMMETRIES; i++) {
      for (int j = 0; j < 2; j++) {
        dst[i * 2 + j] = src[j];
      }
    }
  }

  void create_symmetry_board(float* dst, const float* src) const {
    // if we change the shape, we need to update this function
    assert(CANONICAL_SHAPE[0] == N && CANONICAL_SHAPE[1] == N + 1 &&
           CANONICAL_SHAPE[2] == N * 2);

    float(&out)[N][N+1][N * 2] = *reinterpret_cast<float(*)[N][N+1][N * 2]>(dst);
    const float(&in)[N][N+1][N * 2] = *reinterpret_cast<const float(*)[N][N+1][N * 2]>(src);

    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
          out[i][j][k] = in[i][k][j];
          out[i][j][k + N] = in[i][k][j + N];
        }
      }
      for (int k = 0; k < N * 2; k++) {
        out[i][N][k] = in[i][N][k];
      }
    }
  }

  void create_symmetry_action(float* dst, const float* src) const {
    // if we change the shape, we need to update this function
    assert(CANONICAL_SHAPE[0] == N && CANONICAL_SHAPE[1] == N + 1 &&
           CANONICAL_SHAPE[2] == N * 2);
    
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        dst[i * N + j] = src[j * N + i];
      }
    }
  }

  void create_symmetry_boards(float* dst, const float* src) const {
    create_symmetry_board(dst, src);
  }

  void create_symmetry_actions(float* dst, const float* src) const {
    create_symmetry_action(dst, src);
  }
};

}  // namespace Connect4
