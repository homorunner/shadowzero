project(
  'dgo',
  'cpp',
  version : '0.0.1',
  default_options : [
    'cpp_std=c++20'
  ]
)

cc         = meson.get_compiler('cpp')

if not cc.has_header('format')
  error('C++ compiler too old, try newer compiler like g++ v13.0 or later. You may use command like "CXX=g++-13 meson build " to specify the compiler.')
endif

includes = []
link_args = []

gtest_proj = subproject('gtest')
gtest      = gtest_proj.get_variable('gtest_main_dep')

# These dependencies are for libtorch libraries
# Usage :
# executable(
#   ...
#   dependencies : [torch_dep, torch_cpu_dep, torch_cuda_dep, c10_dep], // torch_cuda_dep is optional here
#   link_args: link_args,
# )
torch_dep      = dependency('Torch', method: 'cmake')
torch_cuda_dep  = cc.find_library(
  'torch_cuda',
  dirs: join_paths(meson.current_source_dir(), 'subprojects\\libtorch\\lib'),
  required: false
)
torch_cpu_dep  = cc.find_library(
  'torch_cpu',
  dirs: join_paths(meson.current_source_dir(), 'subprojects\\libtorch\\lib'),
  required: true
)
c10_dep        = cc.find_library(
  'c10',
  dirs: join_paths(meson.current_source_dir(), 'subprojects\\libtorch\\lib'),
  required: true
)
# End of dependencies for libtorch libraries

if get_option('use_cuda')
  link_args += ['-Wl,--no-as-needed']
  add_project_arguments('-DUSE_CUDA=1', language: 'cpp')
endif


##################
# Benchmark for algorithms
##################

benchmark_alphazero = executable(
  'benchmark_alphazero',
  'benchmark/strategy_alphazero.cpp',
)

benchmark_libtorch_basic = executable(
  'benchmark_libtorch_basic',
  'benchmark/evaluator_libtorch_basic.cpp',
  dependencies : [torch_dep, torch_cpu_dep, torch_cuda_dep, c10_dep],
  link_args: link_args,
)



##################
# Tests for algorithms
##################

strategy_alphazero_test = executable(
  'strategy_alphazero_test',
  'core/algorithm/strategy_alphazero_test.cpp',
  dependencies: [torch_dep, torch_cpu_dep, torch_cuda_dep, c10_dep, gtest],
)
test('strategy_alphazero', strategy_alphazero_test, workdir : meson.project_source_root())


##################
# Tests for games
##################

game_shadow_test = executable(
  'game_shadow_test',
  'game/shadow_test.cpp',
  dependencies: gtest
)
test('game_shadow', game_shadow_test, workdir : meson.project_source_root())


##################
# Executables
##################

selfplay_shadow = executable(
  'selfplay_shadow',
  'cmd/selfplay_shadow.cpp',
  dependencies : [torch_dep, torch_cpu_dep, torch_cuda_dep, c10_dep],
	link_args: link_args,
)

gate_shadow = executable(
  'gate_shadow',
  'cmd/gate_shadow.cpp',
  dependencies : [torch_dep, torch_cpu_dep, torch_cuda_dep, c10_dep],
	link_args: link_args,
)

shadow = executable(
  'shadow',
  'cmd/game_shadow.cpp',
  dependencies : [torch_dep, torch_cpu_dep, torch_cuda_dep, c10_dep],
	link_args: link_args,
)
