"""Milestone 2 check: run from the project root with  python tests/test_milestone2.py

For each of the 12 saved maps it picks 3-7 distributors (fixed by the map's seed) and verifies:
  1. both routes are valid (real roads, start at Office, visit every required distributor)
  2. A* is optimal: its cost equals uniform-cost search (A* with h = 0)
  3. A* cost <= BFS cost on every map
  4. A* never expands more states than uniform-cost search
  5. BFS uses the fewest roads (<= A* roads)
"""
import glob
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from agents import AStarAgent, BFSAgent
from environment import MapEnvironment
from problem import RouteProblem


def assert_valid(problem, path):
    assert path[0] == problem.start_node
    for a, b in zip(path, path[1:]):
        assert b in problem.env.get_neighbors(a), f"{a}->{b} is not a road"
    assert problem.required <= set(path), "route misses a required distributor"


files = sorted(glob.glob(os.path.join(ROOT, "maps", "map_seed_*.json")))
assert len(files) >= 10, f"need at least 10 maps in maps/, found {len(files)}"

header = f"{'Seed':>5} {'Nodes':>5} {'Tgts':>4} | {'A* cost':>8} {'BFS cost':>8} {'Saved':>6} | {'A* exp':>6} {'UCS exp':>7} {'BFS exp':>7} | {'A* ms':>7} {'BFS ms':>7}"
print(header)
print("-" * len(header))

for f in files:
    env = MapEnvironment.load_from_json(f)
    rng = random.Random(env.seed)
    targets = rng.sample(env.distributors(), rng.randint(3, 7))
    problem = RouteProblem(env, "Office", targets)

    a, u, b = AStarAgent(), AStarAgent(use_heuristic=False), BFSAgent()
    a.solve(problem)
    u.solve(problem)
    b.solve(problem)
    sa, su, sb = a.get_statistics(), u.get_statistics(), b.get_statistics()

    assert sa["found"] and sb["found"]
    assert_valid(problem, sa["path"])
    assert_valid(problem, sb["path"])
    assert abs(sa["path_cost"] - su["path_cost"]) < 1e-6, "A* is not optimal"
    assert sa["path_cost"] <= sb["path_cost"] + 1e-9, "A* costlier than BFS"
    assert sa["nodes_expanded"] <= su["nodes_expanded"], "heuristic did not help"
    assert sb["path_edges"] <= sa["path_edges"], "BFS did not minimise roads"

    saved = 100 * (sb["path_cost"] - sa["path_cost"]) / sb["path_cost"]
    print(f"{env.seed:>5} {len(env.location_names):>5} {len(targets):>4} | "
          f"{sa['path_cost']:>8.2f} {sb['path_cost']:>8.2f} {saved:>5.1f}% | "
          f"{sa['nodes_expanded']:>6} {su['nodes_expanded']:>7} {sb['nodes_expanded']:>7} | "
          f"{sa['runtime_ms']:>7.2f} {sb['runtime_ms']:>7.2f}")

print("\nExample route (last map):")
print("  A*  :", " -> ".join(sa["path"]))
print("  BFS :", " -> ".join(sb["path"]))
print("\nMILESTONE 2 PASSED")
