# DDT

Implementation of several algorithms that solves two player zero-sum games.

Full observative, turn-based games are currently supported, but simultaneous games and partial information games are planned to be supported. 

# Algorithm interfaces

Every algorithm should implement these interfaces:

```cpp
ComputationContext compute(const GameState& game);
void ComputationContext::step(int iterations);
ActionType ComputationContext::best_move();
float ComputationContext::best_value();
ActionType ComputationContext::select_move(float temperature);
```

The `compute` function returns a `ComputationContext` object that contains action probability information.

The `step` function perform a fixed-turn compute to the computation context based on `iterations` parameter.

# Game interfaces

Every game should implement these interfaces:

```cpp
int Num_actions();
bool Current_player();
std::vector<uint8_t> Valid_moves();
bool End();
bool Winner();
float Score();
void Canonicalize(float*);
std::unique_ptr<GameState> Copy();
HashType Hash();

std::string ToString();
std::string action_to_string(const ActionType action);
ActionType string_to_action(const std::string& action);

void create_symmetry_values(float* dst, const float* src);
void create_symmetry_boards(float* dst, const float* src);
void create_symmetry_actions(float* dst, const float* src);
```

