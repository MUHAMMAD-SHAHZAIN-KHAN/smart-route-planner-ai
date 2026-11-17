"""Milestone 1 check: run from the project root with  python tests/test_milestone1.py"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from environment import MapEnvironment
from problem import RouteProblem


def connected(env):
    seen, stack = {"Office"}, ["Office"]
    while stack:
        for nb in env.get_neighbors(stack.pop()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(env.location_names)


# 1. Same seed -> same map; different seed -> different map
a = MapEnvironment().generate_map(42, 10)
b = MapEnvironment().generate_map(42, 10)
c = MapEnvironment().generate_map(43, 10)
assert a.coordinates == b.coordinates and a.edge_costs == b.edge_costs
assert a.coordinates != c.coordinates
print("OK  seeded maps are reproducible")

# 2. Connected for many seeds and sizes; edge cost >= straight-line distance (admissible)
for seed in range(100):
    for n in (8, 12, 20):
        env = MapEnvironment().generate_map(seed, n)
        assert connected(env), (seed, n)
        for key, cost in env.edge_costs.items():
            x, y = tuple(key)
            assert cost >= env.estimate_cost(x, y), (seed, n, x, y)
print("OK  300 maps connected; heuristic admissible on every edge")

# 3. JSON round trip
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "m.json")
    a.save_to_json(p)
    r = MapEnvironment.load_from_json(p)
    assert r.coordinates == a.coordinates and r.edge_costs == a.edge_costs and r.graph.keys() == a.graph.keys()
print("OK  save/load JSON round trip")

# 4. Manual walk: goal flips only when every required distributor is visited
env = MapEnvironment().generate_map(42, 8)
print("\nOffice neighbours (seed 42):")
for nb in env.get_neighbors("Office"):
    print(f"  Office -> {nb}: cost {env.get_edge_cost('Office', nb)}")

required = ["Dist_2", "Dist_5"]
prob = RouteProblem(env, "Office", required)
state = prob.initial_state()
assert not prob.is_goal(state)

# shortest hop route to each required node, built by simple BFS just for this test
def hop_path(src, dst):
    prev, queue = {src: None}, [src]
    for cur in queue:
        for nb in env.get_neighbors(cur):
            if nb not in prev:
                prev[nb] = cur
                queue.append(nb)
    path, n = [], dst
    while n is not None:
        path.append(n)
        n = prev[n]
    return path[::-1]

walk = ["Office"]
for target in required:
    walk += hop_path(walk[-1], target)[1:]
for i, nxt in enumerate(walk[1:], 1):
    assert nxt in prob.get_actions(state)
    state = prob.apply_action(state, nxt)
    print(f"  step {i}: at {state[0]:7s} visited={sorted(state[1])} goal={prob.is_goal(state)}")
assert prob.is_goal(state)
print("path:", " -> ".join(walk), "| cost:", prob.path_cost(walk))

# 5. Heuristic never overestimates the true remaining cost along that walk
total = prob.path_cost(walk)
assert prob.heuristic(prob.initial_state()) <= total
assert prob.heuristic(state) == 0
print("OK  RouteProblem works; goal flips only after all required visited; no return to office needed")
print("\nMILESTONE 1 PASSED")
