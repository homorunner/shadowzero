#pragma once

#include "core/util/common.h"
#include "core/util/xxhash64.h"

namespace Shadow {

constexpr const int NUM_ACTIONS = 8 * 16 * 16;
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 1;

constexpr const std::array<int, 3> CANONICAL_SHAPE = {32, 4, 4};

using ActionType = int;

const int8_t dirx[16] = {0, 1, 0, -1, 1, 1, -1, -1, 0, 2, 0, -2, 2, 2, -2, -2};
const int8_t diry[16] = {1, 0, -1, 0, 1, -1, 1, -1, 2, 0, -2, 0, 2, -2, 2, -2};
const int8_t first_step_x[16] = {0, 1, 0, -1, 1, 1, -1, -1,
                                 0, 1, 0, -1, 1, 1, -1, -1};
const int8_t first_step_y[16] = {1, 0, -1, 0, 1, -1, 1, -1,
                                 1, 0, -1, 0, 1, -1, 1, -1};
const char* pos_name[2] = {"12345678abcdefgh", "abcdefgh12345678"};
const char* dir_name[16] = {"d",  "r",  "u",  "l",  "dr",  "ur",  "dl",  "ul",
                            "d2", "r2", "u2", "l2", "dr2", "ur2", "dl2", "ul2"};
const char* dir_name_alt[16] = {"down", "right", "up",  "left", "rd", "ru",
                                "ld",   "lu",    "d2",  "r2",   "u2", "l2",
                                "rd2",  "ru2",   "ld2", "lu2"};

// the position of the piece is 0 ~ 15, so 16 means it was captured.
const int CAPTURED = 16;

std::string action_to_string(const ActionType action, const bool player = 0) {
  return std::string(1, pos_name[0][action % 128 / 16]) +
         pos_name[player][action % 16] + dir_name[action / 128];
}

ActionType string_to_action(const std::string& action, const bool player = 0) {
  int ret = (action[0] - '1') * 16;
  for (int i = 0; i < 16; i++) {
    if (action[1] == pos_name[player][i]) {
      ret += i;
      break;
    }
  }
  for (int i = 0; i < 16; i++) {
    if (action.substr(2) == dir_name[i] || action.substr(2) == dir_name_alt[i]) {
      return ret + i * 128;
    }
  }
  return -1;
}

void sort4(int8_t* arr) {
  if (arr[0] > arr[1]) std::swap(arr[0], arr[1]);
  if (arr[2] > arr[3]) std::swap(arr[2], arr[3]);
  if (arr[0] > arr[2]) std::swap(arr[0], arr[2]);
  if (arr[1] > arr[3]) std::swap(arr[1], arr[3]);
  if (arr[1] > arr[2]) std::swap(arr[1], arr[2]);
}

using ActionType = Shadow::ActionType;

class GameState {
 private:
  bool current_player;
  int8_t round;
  int8_t piece[2][16];  // player_id, piece_id. piece 0~7 is moveable, piece
                        // 8~15 is shadow.

 public:
  GameState() {
    current_player = 0;
    round = 0;
    piece[0][0] = 0;
    piece[0][1] = 1;
    piece[0][2] = 2;
    piece[0][3] = 3;
    piece[0][4] = 0;
    piece[0][5] = 1;
    piece[0][6] = 2;
    piece[0][7] = 3;
    piece[0][8] = 0;
    piece[0][9] = 1;
    piece[0][10] = 2;
    piece[0][11] = 3;
    piece[0][12] = 0;
    piece[0][13] = 1;
    piece[0][14] = 2;
    piece[0][15] = 3;
    piece[1][0] = 12;
    piece[1][1] = 13;
    piece[1][2] = 14;
    piece[1][3] = 15;
    piece[1][4] = 12;
    piece[1][5] = 13;
    piece[1][6] = 14;
    piece[1][7] = 15;
    piece[1][8] = 12;
    piece[1][9] = 13;
    piece[1][10] = 14;
    piece[1][11] = 15;
    piece[1][12] = 12;
    piece[1][13] = 13;
    piece[1][14] = 14;
    piece[1][15] = 15;
  }

  std::unique_ptr<GameState> Copy() const {
    return std::make_unique<GameState>(*this);
  }

  uint64_t Hash() const noexcept {
    // not implemented
    assert(false);
    return 0;
  }

  bool Current_player() { return current_player; }

  int Num_actions() { return NUM_ACTIONS; }

  std::vector<uint8_t> Valid_moves() const {
    auto valids = std::vector<uint8_t>(NUM_ACTIONS, 0);
    bool vshadow = (round / 6) % 2 == 0;

    uint8_t board[2][4][4][4] = {0};

    for (int p = 0; p < 2; p++) {
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          board[p][i][piece[p][i * 4 + j] % 4][piece[p][i * 4 + j] / 4] = 1;
        }
      }
    }

    for (int dir = 0; dir < 16; dir++) {
      int dx = dirx[dir], dy = diry[dir];
      for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 16; b++) {
          int board_a = current_player ? a / 4 + 2 : a / 4;
          int board_b = b / 4;

          [[assume(board_a < 4)]];
          [[assume(board_b < 4)]];

          // if b is not the shadow of a, it can't move
          if (vshadow) {
            if (board_a == board_b || board_a % 2 == board_b % 2) {
              continue;
            }
          } else {  // !vshadow
            if (board_b / 2 == board_a / 2) {
              continue;
            }
          }

          int piece_a = piece[current_player][current_player ? a + 8 : a];
          int piece_b = piece[current_player][b];

          // if captured, it can't move
          if (piece_a == CAPTURED || piece_b == CAPTURED) {
            continue;
          }

          int ax = piece_a % 4;
          int ay = piece_a / 4;
          int bx = piece_b % 4;
          int by = piece_b / 4;

          // if move out of board, it can't move
          if (ax + dx < 0 || ax + dx >= 4 || ay + dy < 0 || ay + dy >= 4) {
            continue;
          }
          if (bx + dx < 0 || bx + dx >= 4 || by + dy < 0 || by + dy >= 4) {
            continue;
          }

          // if anything in a's way, it can't move
          if (board[0][board_a][ax + first_step_x[dir]]
                   [ay + first_step_y[dir]] ||
              board[1][board_a][ax + first_step_x[dir]]
                   [ay + first_step_y[dir]] ||
              board[0][board_a][ax + dx][ay + dy] ||
              board[1][board_a][ax + dx][ay + dy]) {
            continue;
          }

          // if friendly piece in b's way, it can't move
          if (board[current_player][board_b][bx + first_step_x[dir]]
                   [by + first_step_y[dir]] ||
              board[current_player][board_b][bx + dx][by + dy]) {
            continue;
          }

          // if two opponent pieces in b's way, it can't move
          int count = board[!current_player][board_b][bx + dx][by + dy];
          if (dir >= 8 && board[!current_player][board_b][bx + first_step_x[dir]][by + first_step_y[dir]])
            count++;
          
          if(int cx = bx + dx + first_step_x[dir], cy = by + dy + first_step_y[dir]; cx >= 0 && cx < 4 && cy >= 0 && cy < 4 && board[!current_player][board_b][cx][cy]) {
            count ++;
          }
          if (count >= 2) {
            continue;
          }

          valids[dir * 128 + a * 16 + b] = 1;
        }
      }
    }

    return valids;
  }

  void Move(ActionType action) {
    auto a = action % 128 / 16, b = action % 16, dir = action / 128;

    if (current_player) a += 8;
    auto& piece_a = piece[current_player][a];
    auto ax = piece_a % 4;
    auto ay = piece_a / 4;
    piece_a = (ax + dirx[dir]) + (ay + diry[dir]) * 4;

    auto& piece_b = piece[current_player][b];
    auto bx = piece_b % 4;
    auto by = piece_b / 4;
    piece_b = (bx + dirx[dir]) + (by + diry[dir]) * 4;
    for (int i = b / 4 * 4; i < b / 4 * 4 + 4; i++) {
      if (piece[!current_player][i] == piece_b ||
          piece[!current_player][i] ==
              (bx + first_step_x[dir]) + (by + first_step_y[dir]) * 4) {
        auto cx = bx + first_step_x[dir] + dirx[dir];
        auto cy = by + first_step_y[dir] + diry[dir];
        if (cx >= 0 && cx < 4 && cy >= 0 && cy < 4) {
          piece[!current_player][i] = cx + cy * 4;
        } else {
          piece[!current_player][i] = CAPTURED;
        }
      }
    }

    sort4(&piece[current_player][a / 4 * 4]);
    sort4(&piece[current_player][b / 4 * 4]);
    sort4(&piece[!current_player][b / 4 * 4]);

    current_player = !current_player;
    round += 1;
  }

  bool End() const {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 16; j += 4) {
        if (piece[i][j] == CAPTURED && piece[i][j + 1] == CAPTURED &&
            piece[i][j + 2] == CAPTURED && piece[i][j + 3] == CAPTURED) {
          return true;
        }
      }
    }

    return false;
  }

  float Score() const {
    assert(End());

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 16; j += 4) {
        if (piece[i][j] == CAPTURED && piece[i][j + 1] == CAPTURED &&
            piece[i][j + 2] == CAPTURED && piece[i][j + 3] == CAPTURED) {
          return i == 0 ? 0.0f : 1.0f;
        }
      }
    }

    return false;
  }

  void Canonicalize(float* storage) const {
    float(&out)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]] =
        *reinterpret_cast<float(*)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]]
                                  [CANONICAL_SHAPE[2]]>(storage);

    for (int i = 0; i < 16; i++) {
      if (piece[current_player][i] != CAPTURED)
        out[i][piece[current_player][i] % 4][piece[current_player][i] / 4] =
            1.0f;
    }

    for (int i = 0; i < 16; i++) {
      if (piece[0][i] != CAPTURED)
        out[16 + i / 4][piece[0][i] % 4][piece[0][i] / 4] = 1.0f;
      if (piece[1][i] != CAPTURED)
        out[20 + i / 4][piece[1][i] % 4][piece[1][i] / 4] = 1.0f;
    }

    if (current_player) {
      std::fill(&out[24][0][0], &out[25][0][0], 1.0f);
    }

    if ((round / 6) % 2 == 0) {
      std::fill(&out[25][0][0], &out[26][0][0], 1.0f);
    }

    for (int i = 0; i < 6; i++) {
      if (round % 6 <= i) {
        std::fill(&out[26 + i][0][0], &out[27 + i][0][0], 1.0f);
      }
    }
  }

  std::string ToString() const noexcept {
    std::string out;

    char board[8][8];
    std::fill(board[0], board[0] + 8 * 8, '.');

    for (int i = 0; i < 4; i++) {
      int offset_x = i == 1 || i == 3 ? 4 : 0;
      int offset_y = i >= 2 ? 4 : 0;
      for (int j = i * 4; j < i * 4 + 4; j++) {
        if (piece[0][j] != CAPTURED) {
          int x = piece[0][j] % 4;
          int y = piece[0][j] / 4;
          board[y + offset_y][x + offset_x] = (offset_y ? 'a' - 8 : '1') + j;
        }

        if (piece[1][j] != CAPTURED) {
          int x = piece[1][j] % 4;
          int y = piece[1][j] / 4;
          board[y + offset_y][x + offset_x] = (offset_y ? '1' - 8 : 'a') + j;
        }
      }
    }

    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        out += board[i][j];
        if (j == 3) out += ' ';
      }
      out += '\n';
      if (i == 3) out += '\n';
    }

    out += "Player: " + std::to_string(int(current_player)) + '\n';
    out += "Shadow: " + std::string((round / 6) % 2 ? "H" : "V") + '\n';

    return out;
  }

  static std::string action_to_string(ActionType action, bool player) {
    return Shadow::action_to_string(action, player);
  }

  static ActionType string_to_action(const std::string& str, bool player) {
    return Shadow::string_to_action(str, player);
  }
};

void create_symmetry_values(float* dst, const float* src) {
  for (int i = 0; i < NUM_SYMMETRIES; i++) {
    for (int j = 0; j < 2; j++) {
      dst[i * 2 + j] = src[j];
    }
  }
}

// flip the boards horizontally
void create_symmetry_board(float* dst, const float* src) {
  // to be implemented
}

// flip the actions horizontally
void create_symmetry_action(float* dst, const float* src) {
  // to be implemented
}

void create_symmetry_boards(float* dst, const float* src) {
  create_symmetry_board(dst, src);
}

void create_symmetry_actions(float* dst, const float* src) {
  create_symmetry_action(dst, src);
}

}  // namespace Shadow
