# DDT

Implementation of an array of distributed gaming algorithms.

# Design concepts

- Easy to compile.
- Easy to deploy.
- Full CI and unit test.

# Algorithm interfaces

Every algorithm should implement these interfaces:

```cpp
ComputationContext compute(const GameState&);
void ComputationContext::step();
ActionType ComputationContext::best_move();
ActionType ComputationContext::select_move();
```

The `compute` function returns a `ComputationContext` object that contains action probability information.

The `step` function perform a one-turn compute to the computation context. For example, one rollout for MCTS based algorithm, or one search for DFS based algorithm.

# Game interfaces

Every game should implement these interfaces:

```cpp
void Init();
bool Is_simultaneous();
int Num_actions();
int Current_player();
std::vector<uint8_t> Valid_moves();
std::vector<ActionType> Current_actions();
bool End();
float Score();
void Canonicalize(float*);
std::unique_ptr<GameState> Copy();
uint64_t Hash();
```

The `Is_simultaneous` function returns if the current turn of the game is performed simultaneously.

The `Current_player` function returns the `player_id` for current player. Note that some games may support only 2 players.

The `Valid_moves` function returns current actions as a list or vector. Should call only when the current turn is not simultaneous.

The `Current_actions` function returns current actions as a list or vector for a specific player. If game is not simultaneous, it default returns the actions for current player.

## TODO

- Optimize when only one move is available.
- Support multiple GPU evaluator.
- Check why there's a win in `./build/gate_legacy ./testdata/quoridor_baseline_2d4c200iter.pt ./testdata/quoridor_legacy_baseline.pt` (1-23)
- Take use of c++23 assume() and unreachable() to optimize.
