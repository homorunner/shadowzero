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
abcd efgh

abcd efgh
.... ....
.... ....
1234 5678
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

  game.Move(string_to_action("15d"));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .567
4... 8...
.... ....
abcd efgh

abcd efgh
.... ....
.... ....
1234 5678
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

  game.Move(string_to_action("5ad"));

  EXPECT_EQ('\n' + game.ToString(), R"(
1234 .567
.... 8...
.... ....
abcd efgh

.abc efgh
d... ....
.... ....
1234 5678
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

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("1eu2", game.Current_player()));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 5678
4... e...
.... ....
abcd .fgh

abcd efgh
1... ....
.... ....
.234 5678
Player: 0
Shadow: V
)");
}

TEST(GameShadow, TestCapturePiece) {
  Shadow::GameState game;

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("3gul2", game.Current_player()));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .567
4... e...
.... ....
abcd fg.h

abcd efgh
1... ....
.... ....
23.4 5678
Player: 0
Shadow: V
)");
}

TEST(GameShadow, TestCapturePiece2) {
  Shadow::GameState game;

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("3gul2", game.Current_player()));
  game.Move(string_to_action("7bdl", game.Current_player()));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .56.
4... e.7.
.... ....
abcd fg.h

a.bc efgh
d... ....
.... ....
12.3 5678
Player: 1
Shadow: V
)");
}

TEST(GameShadow, TestValidMove) {
  Shadow::GameState game;

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("3gul2", game.Current_player()));
  game.Move(string_to_action("7bdl", game.Current_player()));

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[string_to_action("52u2", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("51u2", game.Current_player())]); // can not push two piece at a time

  EXPECT_TRUE(valid_moves[string_to_action("72lu", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("71lu", game.Current_player())]); // can not move outside
  EXPECT_FALSE(valid_moves[string_to_action("72lu2", game.Current_player())]); // can not move outside
  EXPECT_FALSE(valid_moves[string_to_action("63lu2", game.Current_player())]); // can not move outside

  EXPECT_TRUE(valid_moves[string_to_action("3hl", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("36l", game.Current_player())]); // can not push friendly piece
  EXPECT_FALSE(valid_moves[string_to_action("23l", game.Current_player())]); // can not move the same board
  EXPECT_FALSE(valid_moves[string_to_action("2bu", game.Current_player())]); // can not move board of the same shadow
}


TEST(GameShadow, TestShadow) {
  Shadow::GameState game;

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15u", game.Current_player()));
  game.Move(string_to_action("48u", game.Current_player()));
  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15u", game.Current_player()));

  EXPECT_EQ('\n' + game.ToString(), R"(
.123 .567
4... 8...
.... ....
abcd efgh

abcd efgh
.... ....
1... 5...
.234 .678
Player: 0
Shadow: H
)");

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[string_to_action("1ad2", game.Current_player())]); // now 1 and a are in different shadow
  EXPECT_TRUE(valid_moves[string_to_action("1ed2", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("15d2", game.Current_player())]); // now 1 and 5 are in same shadow
  EXPECT_FALSE(valid_moves[string_to_action("51d2", game.Current_player())]);
}


TEST(GameShadow, TestShadow2) {
  Shadow::GameState game;

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15u", game.Current_player()));
  game.Move(string_to_action("48u", game.Current_player()));
  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15u", game.Current_player()));
  game.Move(string_to_action("1ad", game.Current_player()));

  EXPECT_EQ('\n' + game.ToString(), R"(
..12 .567
34.. 8...
.... ....
abcd efgh

.abc efgh
d... ....
1... 5...
.234 .678
Player: 1
Shadow: H
)");

  auto valid_moves = game.Valid_moves();
  EXPECT_TRUE(valid_moves[string_to_action("2au", game.Current_player())]);
  EXPECT_TRUE(valid_moves[string_to_action("2eu", game.Current_player())]);
  EXPECT_TRUE(valid_moves[string_to_action("8du", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("15r", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("1au", game.Current_player())]);
  EXPECT_FALSE(valid_moves[string_to_action("1eu", game.Current_player())]);
}

TEST(GameShadow, TestShadow3) {
  Shadow::GameState game;

  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15u", game.Current_player()));
  game.Move(string_to_action("48u", game.Current_player()));
  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15d", game.Current_player()));
  game.Move(string_to_action("15u", game.Current_player()));
  game.Move(string_to_action("1ad", game.Current_player()));
  game.Move(string_to_action("6eu", game.Current_player()));
  game.Move(string_to_action("4du", game.Current_player()));
  game.Move(string_to_action("6eu", game.Current_player()));
  game.Move(string_to_action("1ad", game.Current_player()));
  game.Move(string_to_action("5ed2", game.Current_player()));

  EXPECT_EQ('\n' + game.ToString(), R"(
..12 5678
34.. ....
.... ....
abcd efgh

.abc efgh
d... ....
1... 5...
.234 .678
Player: 0
Shadow: V
)");
}