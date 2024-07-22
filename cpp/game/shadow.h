#pragma once

#include "core/util/common.h"

namespace Shadow {

constexpr const int NUM_ACTIONS = 8 * 8 * 16;
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 2;

constexpr const std::array<int, 3> CANONICAL_SHAPE = {25, 4, 4};
constexpr const int CANONICAL_EXTRA_COUNT = 14;

using ActionType = int;

const ActionType MOVE_PASS = -1;

const int8_t dirx[16] = {0, 1, 0, -1, 1, 1, -1, -1, 0, 2, 0, -2, 2, 2, -2, -2};
const int8_t diry[16] = {1, 0, -1, 0, 1, -1, 1, -1, 2, 0, -2, 0, 2, -2, 2, -2};
const int8_t first_step_x[16] = {0, 1, 0, -1, 1, 1, -1, -1, 0, 1, 0, -1, 1, 1, -1, -1};
const int8_t first_step_y[16] = {1, 0, -1, 0, 1, -1, 1, -1, 1, 0, -1, 0, 1, -1, 1, -1};
const char* pos_a_name = {"12345678"};
const char* pos_b_name[2][2] = {{"5678efgh", "1234abcd"}, {"abcdefgh", "abcdefgh"}};
const int8_t pos_b_index[2][2][8] = {
    {{4, 5, 6, 7, 12, 13, 14, 15}, {0, 1, 2, 3, 8, 9, 10, 11}},
    {{8, 9, 10, 11, 12, 13, 14, 15}, {8, 9, 10, 11, 12, 13, 14, 15}},
};
const char* dir_name[2][16] = {
    {"d", "r", "u", "l", "dr", "ur", "dl", "ul", "d2", "r2", "u2", "l2", "dr2", "ur2", "dl2", "ul2"},
    {"u", "l", "d", "r", "ul", "dl", "ur", "dr", "u2", "l2", "d2", "r2", "ul2", "dl2", "ur2", "dr2"}};
const char* dir_name_alt[2][16] = {
    {"down", "right", "up", "left", "rd", "ru", "ld", "lu", "d2", "r2", "u2", "l2", "rd2", "ru2", "ld2", "lu2"},
    {"up", "left", "down", "right", "lu", "ld", "ru", "rd", "u2", "l2", "d2", "r2", "lu2", "ld2", "ru2", "rd2"}};

// the position of the piece is 0 ~ 15, so 16 means it was captured.
const int CAPTURED = 16;

void sort4(int8_t* arr) {
  if (arr[0] > arr[1]) std::swap(arr[0], arr[1]);
  if (arr[2] > arr[3]) std::swap(arr[2], arr[3]);
  if (arr[0] > arr[2]) std::swap(arr[0], arr[2]);
  if (arr[1] > arr[3]) std::swap(arr[1], arr[3]);
  if (arr[1] > arr[2]) std::swap(arr[1], arr[2]);
}

class GameState {
 private:
  bool current_player;
  bool __unused;
  int16_t round;
  int8_t piece[2][16];  // player_id, piece_id. piece 0~7 is moveable, piece
                        // 8~15 is shadow.

 public:
  GameState() {
    current_player = 0;
    round = 0;
    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < 16; i++) {
        piece[j][i] = i % 4;
      }
    }
  }

  GameState(bool current_player, int16_t round, int8_t piece[2][16]) : current_player(current_player), round(round) {
    std::memcpy(this->piece, piece, sizeof(this->piece));
  }

  std::string action_to_string(const ActionType action) {
    if (action == MOVE_PASS) return "pass";
    return std::string(1, pos_a_name[action % 64 / 8]) +
           pos_b_name[(round / 12) % 2][(action % 64 / 8) >= 4][action % 8] + dir_name[current_player][action / 64];
  }

  ActionType string_to_action(const std::string& action) {
    if (action == "pass") return MOVE_PASS;
    int a = action[0] - '1';
    int ret = a * 8;
    bool found = false;
    for (int i = 0; i < 8; i++) {
      if (action[1] == pos_b_name[(round / 12) % 2][a >= 4][i]) {
        ret += i;
        found = true;
        break;
      }
    }
    if (!found) return -1;
    for (int i = 0; i < 16; i++) {
      if (action.substr(2) == dir_name[current_player][i] || action.substr(2) == dir_name_alt[current_player][i]) {
        return ret + i * 64;
      }
    }
    return -1;
  }

  std::unique_ptr<GameState> Copy() const { return std::make_unique<GameState>(*this); }

  uint64_t Hash() const noexcept {
    // not implemented
    return 0;
  }

  bool Current_player() const noexcept { return current_player; }

  int Num_actions() const noexcept { return NUM_ACTIONS; }

  std::vector<uint8_t> Valid_moves() const {
    auto valids = std::vector<uint8_t>(NUM_ACTIONS, 0);
    bool vshadow = (round / 12) % 2 == 0;

    uint8_t board[2][4][4][4] = {0};
    int8_t position[2][16];

    for (int i = 0; i < 16; i++) {
      position[0][i] = piece[current_player][i];
      position[1][i] = piece[!current_player][i] == CAPTURED ? CAPTURED : 15 - piece[!current_player][i];
    }

    for (int p = 0; p < 2; p++) {
      for (int i = 0; i < 16; i++) {
        if (position[p][i] != CAPTURED) board[p][p ? 3 - i / 4 : i / 4][position[p][i] % 4][position[p][i] / 4] = 1;
      }
    }

    for (int dir = 0; dir < 16; dir++) {
      auto dx = dirx[dir], dy = diry[dir];
      for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
          int board_a = a / 4;
          int board_b = vshadow ? a < 4 ? (b < 4 ? 1 : 3) : (b < 4 ? 0 : 2) : b / 4 + 2;

          auto position_a = position[0][a];
          auto position_b = position[0][board_b * 4 + b % 4];

          // if captured, it can't move
          if (position_a == CAPTURED || position_b == CAPTURED) {
            continue;
          }

          auto ax = position_a % 4;
          auto ay = position_a / 4;
          auto bx = position_b % 4;
          auto by = position_b / 4;

          // if b move out of board, it can't move
          if (bx + dx < 0 || bx + dx >= 4 || by + dy < 0 || by + dy >= 4) {
            continue;
          }

          bool move_out_flag = false;

          // if anything in a's way, it can't move
          if (ax + first_step_x[dir] < 0 || ax + first_step_x[dir] >= 4 || ay + first_step_y[dir] < 0 ||
              ay + first_step_y[dir] >= 4) {
          } else if (board[0][board_a][ax + first_step_x[dir]][ay + first_step_y[dir]] ||
                     board[1][board_a][ax + first_step_x[dir]][ay + first_step_y[dir]]) {
            continue;
          }
          if (ax + dx < 0 || ax + dx >= 4 || ay + dy < 0 || ay + dy >= 4) {
            move_out_flag = true;
          } else if (board[0][board_a][ax + dx][ay + dy] || board[1][board_a][ax + dx][ay + dy]) {
            continue;
          }

          // if friendly piece in b's way, it can't move
          if (board[0][board_b][bx + first_step_x[dir]][by + first_step_y[dir]] ||
              board[0][board_b][bx + dx][by + dy]) {
            continue;
          }

          // if two opponent pieces in b's way, it can't move
          int count = board[1][board_b][bx + dx][by + dy];
          if (dir >= 8 && board[1][board_b][bx + first_step_x[dir]][by + first_step_y[dir]]) count++;
          if (count == 0 && move_out_flag) {
            continue;
          }

          if (int cx = bx + dx + first_step_x[dir], cy = by + dy + first_step_y[dir];
              cx >= 0 && cx < 4 && cy >= 0 && cy < 4 && (board[0][board_b][cx][cy] || board[1][board_b][cx][cy])) {
            count++;
          }
          if (count >= 2) {
            continue;
          }

          valids[dir * 64 + a * 8 + b] = 1;
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

    int a = action % 64 / 8, b = pos_b_index[(round / 12) % 2][a >= 4][action % 8], dir = action / 64;

    assert(a >= 0 && a < 8);
    assert(b >= 0 && b < 16);
    assert(dir >= 0 && dir < 16);

    auto& piece_a = piece[current_player][a];
    assert(piece_a >= 0 && piece_a < 16);
    auto ax = piece_a % 4;
    auto ay = piece_a / 4;
    if (ax + dirx[dir] < 0 || ax + dirx[dir] >= 4 || ay + diry[dir] < 0 || ay + diry[dir] >= 4) {
      piece_a = CAPTURED;
    } else {
      piece_a = (ax + dirx[dir]) + (ay + diry[dir]) * 4;
      assert(piece_a >= 0 && piece_a < 16);
    }

    auto& piece_b = piece[current_player][b];
    assert(piece_b >= 0 && piece_b < 16);
    auto bx = piece_b % 4;
    auto by = piece_b / 4;
    if (bx + dirx[dir] < 0 || bx + dirx[dir] >= 4 || by + diry[dir] < 0 || by + diry[dir] >= 4) {
      piece_b = CAPTURED;
    } else {
      piece_b = (bx + dirx[dir]) + (by + diry[dir]) * 4;
      assert(piece_b >= 0 && piece_b < 16);
    }

    int op = (3 - b / 4) * 4;
    for (int i = op; i < op + 4; i++) {
      if (15 - piece[!current_player][i] == piece_b ||
          15 - piece[!current_player][i] == (bx + first_step_x[dir]) + (by + first_step_y[dir]) * 4) {
        auto cx = bx + first_step_x[dir] + dirx[dir];
        auto cy = by + first_step_y[dir] + diry[dir];
        if (cx >= 0 && cx < 4 && cy >= 0 && cy < 4) {
          piece[!current_player][i] = 15 - cx - cy * 4;
        } else {
          piece[!current_player][i] = CAPTURED;
        }
        break;
      }
    }

    sort4(&piece[current_player][a / 4 * 4]);
    sort4(&piece[current_player][b / 4 * 4]);
    sort4(&piece[!current_player][op]);

    current_player = !current_player;
    round += 1;
  }

  bool End() const {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 16; j += 4) {
        if (piece[i][j] == CAPTURED && piece[i][j + 1] == CAPTURED && piece[i][j + 2] == CAPTURED &&
            piece[i][j + 3] == CAPTURED) {
          return true;
        }
      }
    }

    return false;
  }

  bool Winner() const {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 16; j += 4) {
        if (piece[i][j] == CAPTURED && piece[i][j + 1] == CAPTURED && piece[i][j + 2] == CAPTURED &&
            piece[i][j + 3] == CAPTURED) {
          return !i;
        }
      }
    }

    std::unreachable();
  }

  float Score() const {
    assert(End());

    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 16; j += 4) {
        if (piece[i][j] == CAPTURED && piece[i][j + 1] == CAPTURED && piece[i][j + 2] == CAPTURED &&
            piece[i][j + 3] == CAPTURED) {
          return i == 0 ? 0.0f : 1.0f;
        }
      }
    }

    return false;
  }

  void Canonicalize(float* storage) const noexcept {
    // if we update this function (e.g. add more channels), we need to update
    // this variable
    assert(CANONICAL_SHAPE[0] == 25 && CANONICAL_SHAPE[1] == 4 && CANONICAL_SHAPE[2] == 4);

    float(&out)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]] =
        *reinterpret_cast<float(*)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]]>(storage);

    float capture_count[2] = {0.0f, 0.0f};

    // channel 0..15 represent each piece position of the current player
    // channel 16..19 represent current player board 0-3
    // channel 20..23 represent opponent player board 0-3
    for (int i = 0; i < 16; i++) {
      if (piece[current_player][i] != CAPTURED) {
        out[i][piece[current_player][i] / 4][piece[current_player][i] % 4] = 1.0f;
        out[16 + i / 4][piece[current_player][i] / 4][piece[current_player][i] % 4] = 1.0f;
      } else {
        capture_count[0] += 1.0f;
      }
      if (piece[!current_player][i] != CAPTURED) {
        out[20 + i / 4][piece[!current_player][i] / 4][piece[!current_player][i] % 4] = 1.0f;
      } else {
        capture_count[1] += 1.0f;
      }
    }

    // channel 24 is the linear vector
    // vector 0 represent the shadow direction(0: V, 1: H)
    // vector 1..11 represent the round count to next shadow
    // vector 12..13 represent the capture count of player 0 and player 1.
    float* linear = &out[24][0][0];
    linear[0] = (round / 12) % 2 ? 1.0f : 0.0f;
    for (int i = 1; i < 12; i++) {
      linear[i] = round % 12 >= i ? 1.0f : 0.0f; // TODO: maybe accelerated a little bit here, because initial value is 0
    }
    linear[12] = capture_count[current_player];
    linear[13] = capture_count[!current_player];

    // if we add more vector, we need to update this variable
    assert(CANONICAL_EXTRA_COUNT == 14);
  }

  std::string ToString() const {
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
          board[y + offset_y][x + offset_x] = (i >= 2 ? 'a' - 8 : '1') + j;
        }

        if (piece[1][j] != CAPTURED) {
          int x = (15 - piece[1][j]) % 4;
          int y = (15 - piece[1][j]) / 4;
          board[y + 4 - offset_y][x + 4 - offset_x] = (i >= 2 ? 'a' - 8 : '1') + j;
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
    out += "Shadow: " + std::string((round / 12) % 2 ? "H" : "V") + '\n';

    return out;
  }

  void create_symmetry_values(float* dst, const float* src) const {
    for (int i = 0; i < NUM_SYMMETRIES; i++) {
      for (int j = 0; j < 2; j++) {
        dst[i * 2 + j] = src[j];
      }
    }
  }

  // change board orders (a,b,c,d -> b,a,d,c)
  void create_symmetry_board(float* dst, const float* src) const {
    // if we change the shape, we need to update this function
    assert(CANONICAL_SHAPE[0] == 25);

    float(&out)[25][4][4] = *reinterpret_cast<float(*)[25][4][4]>(dst);
    const float(&in)[25][4][4] = *reinterpret_cast<const float(*)[25][4][4]>(src);

    // note that we assert the canonical board is already filled with zeros.
    assert(out[0][0][0] == 0.0f && out[24][3][3] == 0.0f);

    for (int i = 0; i < 16; i++) {
      float* dst_ptr = &out[i ^ 4][0][0];
      const float* src_ptr = &in[i][0][0];
      std::memcpy(dst_ptr, src_ptr, sizeof(float) * 16);
    }
    for (int i = 16; i < 24; i++) {
      float* dst_ptr = &out[i ^ 1][0][0];
      const float* src_ptr = &in[i][0][0];
      std::memcpy(dst_ptr, src_ptr, sizeof(float) * 16);
    }
    float* dst_ptr = &out[24][0][0];
    const float* src_ptr = &in[24][0][0];
    for (int i = 0; i < CANONICAL_EXTRA_COUNT; i++) {
      dst_ptr[i] = src_ptr[i];
    }
  }

  // flip the actions horizontally
  void create_symmetry_action(float* dst, const float* src) const {
    // if we change the shape, we need to update this function
    assert(CANONICAL_SHAPE[0] == 25);

    bool vshadow = (round / 12) % 2 == 0;

    for (int dir = 0; dir < 16; dir++) {
      for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
          int idx = dir * 64 + a * 8 + b;

          int new_a = a ^ 4;
          int new_b = vshadow ? b : b ^ 4;
          int new_idx = dir * 64 + new_a * 8 + new_b;

          dst[new_idx] = src[idx];
        }
      }
    }
  }

  void create_symmetry_boards(float* dst, const float* src) const { create_symmetry_board(dst, src); }

  void create_symmetry_actions(float* dst, const float* src) const { create_symmetry_action(dst, src); }
};

}  // namespace Shadow
