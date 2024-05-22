#include "game/shadow.h"

#include <iostream>

#include "gtest/gtest.h"

using namespace Shadow;

TEST(GameShadow, TestInitState) {
  Shadow::GameState game;

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
  Shadow::GameState game;

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
  EXPECT_EQ(valid, 29 * 8);  // 232

  auto ended = game.End();
  EXPECT_FALSE(ended);
}

TEST(GameShadow, TestFirstStep2) {
  Shadow::GameState game;

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
  EXPECT_EQ(valid, 220);  // 1xu2...8; 3xul2...4; 232-12=220

  auto ended = game.End();
  EXPECT_FALSE(ended);
}

TEST(GameShadow, TestMovePiece) {
  Shadow::GameState game;

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
  Shadow::GameState game;

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
  EXPECT_EQ(count, 168);
}

TEST(GameShadow, TestCapturePiece2) {
  Shadow::GameState game;

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
  EXPECT_EQ(count, 168);
}

TEST(GameShadow, TestValidMove) {
  Shadow::GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("6bul2"));
  game.Move(game.string_to_action("7bdl"));

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[game.string_to_action("36u2")]);
  EXPECT_FALSE(valid_moves[game.string_to_action(
      "37u2")]);  // can not push two piece at a time

  EXPECT_TRUE(valid_moves[game.string_to_action("63lu")]);
  EXPECT_FALSE(
      valid_moves[game.string_to_action("73lu")]);  // can not move outside
  EXPECT_FALSE(
      valid_moves[game.string_to_action("53lu2")]);  // can not move outside
  EXPECT_FALSE(
      valid_moves[game.string_to_action("73lu2")]);  // can not move outside

  EXPECT_TRUE(valid_moves[game.string_to_action("5al")]);
  EXPECT_FALSE(valid_moves[game.string_to_action(
      "5bl")]);                                 // can not push friendly piece
  EXPECT_EQ(-1, game.string_to_action("56u"));  // can not move the same board
  EXPECT_EQ(-1, game.string_to_action(
                    "2du"));  // can not move board of the same shadow
}

TEST(GameShadow, TestValidMove2) {
  Shadow::GameState game;

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
  Shadow::GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("48u"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));

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
  EXPECT_TRUE(valid_moves[game.string_to_action(
      "1ad2")]);  // now 1 and a are in different shadow
  EXPECT_TRUE(valid_moves[game.string_to_action("1ed2")]);
  EXPECT_EQ(-1,
            game.string_to_action("15d2"));  // now 1 and 5 are in same shadow
  EXPECT_EQ(-1, game.string_to_action("51d2"));
}

TEST(GameShadow, TestShadow2) {
  Shadow::GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("48u"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
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
  Shadow::GameState game;

  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("48u"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15d"));
  game.Move(game.string_to_action("15u"));
  game.Move(game.string_to_action("1ad"));
  game.Move(game.string_to_action("3du"));
  game.Move(game.string_to_action("4du"));
  game.Move(game.string_to_action("4du"));
  game.Move(game.string_to_action("1ad"));
  game.Move(game.string_to_action("4dd2"));

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
  EXPECT_EQ(count, 190);
}

TEST(GameShadow, TestGameEnd) {
  Shadow::GameState game;

  EXPECT_FALSE(game.End());

  // game.Move(game.string_to_action("..."));

  // EXPECT_TRUE(game.End());
  // EXPECT_EQ(game.Score(), 1.0f);
}
