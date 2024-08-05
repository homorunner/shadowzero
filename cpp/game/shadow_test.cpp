#include "game/shadow.h"

#include <iostream>

#include "gtest/gtest.h"

using namespace Shadow;

TEST(GameShadow, TestInitState) {
  GameState game;

  EXPECT_EQ('\n' + game.ToString(), R"(
1234 5678
.... ....
.... ....
hgfe dcba

abcd efgh
.... ....
.... ....
8765 4321
Player: 0
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  int valid = 0;
  for (auto m : valid_moves) {
    valid += m;
  }
  EXPECT_EQ(valid, 29 * 8);  // 232

  auto ended = game.End();
  EXPECT_FALSE(ended);
}

TEST(GameShadow, TestFirstStep) {
  GameState game;

  game.Move(game.string_to_action("15d"));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .567
4... 8...
.... ....
hgfe dcba

abcd efgh
.... ....
.... ....
8765 4321
Player: 1
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  int valid = 0;
  for (auto m : valid_moves) {
    valid += m;
  }
  EXPECT_EQ(valid, 236);

  auto ended = game.End();
  EXPECT_FALSE(ended);
}

TEST(GameShadow, TestFirstStep2) {
  GameState game;

  game.Move(game.string_to_action("5ad"));

  EXPECT_EQ('\n' + game.ToString(), R"(
1234 .567
.... 8...
.... ....
hgfe dcba

.abc efgh
d... ....
.... ....
8765 4321
Player: 1
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  int valid = 0;
  for (auto m : valid_moves) {
    valid += m;
  }
  EXPECT_EQ(valid, 224);

  auto ended = game.End();
  EXPECT_FALSE(ended);
}

TEST(GameShadow, TestMovePiece) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("8du2"));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 5678
4... d...
.... ....
hgfe .cba

abcd efgh
8... ....
.... ....
.765 4321
Player: 0
Shadow: V
)");
}

TEST(GameShadow, TestCapturePiece) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("6bul2"));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .567
4... d...
.... ....
hgfe cb.a

abcd efgh
8... ....
.... ....
76.5 4321
Player: 0
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  int count = 0;
  for (auto m : valid_moves) {
    count += m;
  }
  EXPECT_EQ(count, 171);
}

TEST(GameShadow, TestCapturePiece2) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("6bul2"));
  game.Move(game.string_to_action("7bdl"));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .56.
4... d.7.
.... ....
hgfe cb.a

a.bc efgh
d... ....
.... ....
76.5 4321
Player: 1
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  int count = 0;
  for (auto m : valid_moves) {
    count += m;
  }
  EXPECT_EQ(count, 173);
}

TEST(GameShadow, TestValidMove) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("6bul2"));
  game.Move(game.string_to_action("7bdl"));

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[game.string_to_action("36u2")]);
  EXPECT_FALSE(valid_moves[game.string_to_action("37u2")]);  // can not push two piece at a time

  EXPECT_TRUE(valid_moves[game.string_to_action("63lu")]);
  EXPECT_FALSE(valid_moves[game.string_to_action("73lu")]);   // can not move outside
  EXPECT_FALSE(valid_moves[game.string_to_action("53lu2")]);  // can not move outside
  EXPECT_FALSE(valid_moves[game.string_to_action("73lu2")]);  // can not move outside

  EXPECT_TRUE(valid_moves[game.string_to_action("5al")]);
  EXPECT_FALSE(valid_moves[game.string_to_action("5bl")]);  // can not push friendly piece
  EXPECT_EQ(-1, game.string_to_action("56u"));              // can not move the same board
  EXPECT_EQ(-1, game.string_to_action("2du"));              // can not move board of the same shadow
}

TEST(GameShadow, TestValidMove2) {
  GameState game;

  game.Move(game.string_to_action("8ddl2"));
  game.Move(game.string_to_action("5bul2"));

  EXPECT_EQ('\n' + game.ToString(), R"(
1234 567.
.... d...
.... ....
hgfe cb.a

abc. efgh
.8.. ....
.d.. ....
765. 4321
Player: 0
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  EXPECT_FALSE(valid_moves[game.string_to_action("7bd")]);
  EXPECT_FALSE(valid_moves[game.string_to_action("7bd2")]);
  EXPECT_TRUE(valid_moves[game.string_to_action("7adr")]);
  EXPECT_TRUE(valid_moves[game.string_to_action("7cdl")]);
}

TEST(GameShadow, TestShadow) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("48u"));
  game.Move(game.string_to_action("48d"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  for (int i = 0; i < 6; i++) game.Move(MOVE_PASS);

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .567
4... 8...
.... ....
hgfe dcba

abcd efgh
.... ....
...8 ...4
765. 321.
Player: 0
Shadow: H
)");

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[game.string_to_action("1ad2")]);  // now 1 and a are in different shadow
  EXPECT_TRUE(valid_moves[game.string_to_action("1ed2")]);
  EXPECT_EQ(-1,
            game.string_to_action("15d2"));  // now 1 and 5 are in same shadow
  EXPECT_EQ(-1, game.string_to_action("51d2"));
}

TEST(GameShadow, TestShadow2) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("48u"));
  game.Move(game.string_to_action("48d"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  for (int i = 0; i < 6; i++) game.Move(MOVE_PASS);
  game.Move(game.string_to_action("1ad"));

  EXPECT_EQ('\n' + game.ToString(), R"(
..12 .567
34.. 8...
.... ....
hgfe dcba

.abc efgh
d... ....
...8 ...4
765. 321.
Player: 1
Shadow: H
)");

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[game.string_to_action("2au")]);
  EXPECT_TRUE(valid_moves[game.string_to_action("2eu")]);
  EXPECT_TRUE(valid_moves[game.string_to_action("8du")]);
  EXPECT_TRUE(valid_moves[game.string_to_action("8hu")]);
  EXPECT_EQ(-1, game.string_to_action("15r"));
  EXPECT_EQ(-1, game.string_to_action("72u"));
}

TEST(GameShadow, TestShadow3) {
  GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("48u"));
  game.Move(game.string_to_action("48d"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  for (int i = 0; i < 6; i++) game.Move(MOVE_PASS);
  game.Move(game.string_to_action("1ad"));
  game.Move(game.string_to_action("3du"));
  game.Move(game.string_to_action("4du"));
  game.Move(game.string_to_action("4du"));
  game.Move(game.string_to_action("1ad"));
  game.Move(game.string_to_action("4dd2"));
  for (int i = 0; i < 6; i++) game.Move(MOVE_PASS);

  EXPECT_EQ('\n' + game.ToString(), R"(
..12 5678
34.. ....
.... ....
hgfe dcba

.abc efgh
d... ....
...8 ...4
765. 321.
Player: 0
Shadow: V
)");

  auto valid_moves = game.Valid_moves();
  int count = 0;
  for (auto m : valid_moves) {
    count += m;
  }
  EXPECT_EQ(count, 200);
}

TEST(GameShadow, TestGameEnd) {
  bool current_player = 0;
  int16_t round = 22;
  int8_t pieces[2][16] = {
      {0, CAPTURED, CAPTURED, CAPTURED, 0, CAPTURED, CAPTURED, CAPTURED, 0, CAPTURED, CAPTURED, CAPTURED, 0, CAPTURED,
       CAPTURED, CAPTURED},
      {0, CAPTURED, CAPTURED, CAPTURED, 0, CAPTURED, CAPTURED, CAPTURED, 0, CAPTURED, CAPTURED, CAPTURED, 0, CAPTURED,
       CAPTURED, CAPTURED},
  };
  GameState game(current_player, round, pieces);

  EXPECT_FALSE(game.End());
  game.Move(game.string_to_action("1adr"));
  EXPECT_FALSE(game.End());
  game.Move(game.string_to_action("5eul"));
  EXPECT_FALSE(game.End());
  game.Move(game.string_to_action("5adr2"));
  EXPECT_TRUE(game.End());

  EXPECT_EQ(game.Score(), 1.0f);
}

std::string StringFromCanonical(const float* canonical) {
  const float(&in)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]] =
      *reinterpret_cast<const float(*)[CANONICAL_SHAPE[0]][CANONICAL_SHAPE[1]][CANONICAL_SHAPE[2]]>(canonical);
  // get round and current player
  int16_t round = 0;
  const float* linear = &in[24][0][0];
  for (int i = 1; i < 12; i++) {
    if (linear[i]) round++;
  }
  bool current_player = round % 2;
  bool hshadow = linear[0];
  if (hshadow) round += 12;

  // get board
  bool board[2][4][4][4];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        board[current_player][i][j][k] = in[16 + i][j][k];
        board[!current_player][i][j][k] = in[20 + i][j][k];
      }
    }
  }

  int index[2][4] = {0};
  int8_t pieces[2][16];
  for (int p = 0; p < 2; p++) {
    for (int i = 0; i < 4; i++) {
      for (int x = 0; x < 4; x++) {
        for (int y = 0; y < 4; y++) {
          if (board[p][i][x][y]) {
            pieces[p][i * 4 + (index[p][i]++)] = x * 4 + y;
          }
        }
      }
    }
  }

  float capture_count[2] = {0};
  for (int p = 0; p < 2; p++) {
    for (int i = 0; i < 4; i++) {
      while (index[p][i] < 3) {
        pieces[p][i * 4 + (index[p][i]++)] = CAPTURED;
        capture_count[p]++;
      }
    }
  }

  if (capture_count[current_player] != linear[13]) {
    return "capture_count[current_player] != linear[13]";
  }
  if (capture_count[!current_player] != linear[14]) {
    return "capture_count[!current_player] != linear[14]";
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 16; j += 4) {
      sort4(pieces[i] + j);
    }
  }

  GameState game(current_player, round, pieces);

  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        if (in[i][j][k] == 1.f) {
          if (pieces[current_player][i] != (j * 4 + k)) {
            return "pieces[current_player][i] != (j * 4 + k)";
          }
        } else if (in[i][j][k] != 0.f) {
          return "in[i][j][k] != 0.f && in[i][j][k] != 1.f";
        }
      }
    }
  }

  return game.ToString();
}

TEST(GameShadow, TestCanonical) {
  GameState game;

  for (int i = 0; i < 10; i++) {
    auto valid_moves = game.Valid_moves();
    int action = -1;
    for (int i = 0; i < NUM_ACTIONS; i++) {
      if (valid_moves[37 * i % NUM_ACTIONS]) {
        action = 37 * i % NUM_ACTIONS;
        break;
      }
    }
    if (action == -1) break;
    game.Move(action);

    float canonical[CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1] * CANONICAL_SHAPE[2]] = {0};
    game.Canonicalize(canonical);

    auto fromCanonical = StringFromCanonical(canonical);
    EXPECT_EQ(fromCanonical, game.ToString());
  }
}

TEST(GameShadow, TestSymmetry) {
  GameState game;

  float canonical[CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1] * CANONICAL_SHAPE[2]] = {0};
  game.Canonicalize(canonical);

  float symmetry[1][CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1] * CANONICAL_SHAPE[2]] = {0};
  game.create_symmetry_boards(&symmetry[0][0], canonical);

  auto fromSymmetry = StringFromCanonical(symmetry[0]);
  EXPECT_EQ(fromSymmetry, game.ToString());
}

TEST(GameShadow, TestSymmetry2) {
  GameState game;
  game.Move(game.string_to_action("1edr"));
  game.Move(game.string_to_action("1eul"));

  GameState game2;
  game2.Move(game2.string_to_action("5adr"));
  game2.Move(game2.string_to_action("5aul"));

  float canonical[CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1] * CANONICAL_SHAPE[2]] = {0};
  game.Canonicalize(canonical);

  float symmetry[1][CANONICAL_SHAPE[0] * CANONICAL_SHAPE[1] * CANONICAL_SHAPE[2]] = {0};
  game.create_symmetry_boards(&symmetry[0][0], canonical);

  auto fromSymmetry = StringFromCanonical(symmetry[0]);
  EXPECT_EQ(fromSymmetry, game2.ToString());
}

TEST(GameShadow, TestSymmetryAction) {
  GameState game;
  game.Move(game.string_to_action("1edr"));
  game.Move(game.string_to_action("1eul"));

  GameState game2;
  game2.Move(game2.string_to_action("5adr"));
  game2.Move(game2.string_to_action("5aul"));

  float actions[NUM_ACTIONS] = {0};
  for (int i = 0; i < NUM_ACTIONS; i++) {
    actions[i] = i;
  }

  float symmetry[1][NUM_ACTIONS] = {0};
  game.create_symmetry_actions(&symmetry[0][0], actions);

  EXPECT_EQ(symmetry[0][game2.string_to_action("36d")], actions[game.string_to_action("72d")]);
  EXPECT_EQ(symmetry[0][game2.string_to_action("8dr2")], actions[game.string_to_action("4hr2")]);
}

TEST(GameShadow, TestSymmetryAction2) {
  GameState game;
  game.Move(game.string_to_action("1edr"));
  game.Move(game.string_to_action("1eul"));
  for (int i = 0; i < 12; i++) game.Move(MOVE_PASS);

  GameState game2;
  game2.Move(game2.string_to_action("5adr"));
  game2.Move(game2.string_to_action("5aul"));
  for (int i = 0; i < 12; i++) game2.Move(MOVE_PASS);

  float actions[NUM_ACTIONS] = {0};
  for (int i = 0; i < NUM_ACTIONS; i++) {
    actions[i] = i;
  }

  float symmetry[1][NUM_ACTIONS] = {0};
  game.create_symmetry_actions(&symmetry[0][0], actions);

  EXPECT_EQ(symmetry[0][game2.string_to_action("3cd")], actions[game.string_to_action("7gd")]);
  EXPECT_EQ(symmetry[0][game2.string_to_action("8dr2")], actions[game.string_to_action("4hr2")]);
}
