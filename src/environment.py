"""MapEnvironment: generates and manages the simulated territory map.

The map is a connected weighted graph.
- Node 0 is the "Office"; every other node is a distributor "Dist_1", "Dist_2", ...
- Each node has (x, y) coordinates on a 100 x 100 grid.
- Every edge cost is >= the straight-line distance between its endpoints, so the
  Euclidean distance is an ADMISSIBLE heuristic (it never overestimates).
"""

import json
import math
import random


class MapEnvironment:
    def __init__(self):
        self.seed = None
        self.location_names = []   # index -> name, e.g. ["Office", "Dist_1", ...]
        self.coordinates = {}      # name -> (x, y)
        self.graph = {}            # name -> list of neighbour names (adjacency list)
        self.edge_costs = {}       # frozenset({a, b}) -> travel cost

    # ------------------------------------------------------------------ build
    def generate_map(self, seed, num_nodes=12, extra_edge_ratio=0.5):
        """Create a reproducible, connected, weighted map.

        seed             -> same seed always gives the same map
        num_nodes        -> total locations including the office (8 to 20)
        extra_edge_ratio -> extra random edges added after the spanning tree,
                            as a fraction of num_nodes (more = denser roads)
        """
        if not 8 <= num_nodes <= 20:
            raise ValueError("num_nodes must be between 8 and 20 (see proposal Section 6)")

        rng = random.Random(seed)  # local RNG: does not disturb global random state
        self.seed = seed
        self.location_names = ["Office"] + [f"Dist_{i}" for i in range(1, num_nodes)]
        self.coordinates = {
            name: (round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2))
            for name in self.location_names
        }
        self.graph = {name: [] for name in self.location_names}
        self.edge_costs = {}

        # 1) Spanning tree (Prim's algorithm on straight-line distance) -> guarantees connectivity
        in_tree = {self.location_names[0]}
        while len(in_tree) < num_nodes:
            best = None
            for a in in_tree:
                for b in self.location_names:
                    if b in in_tree:
                        continue
                    d = self._euclid(a, b)
                    if best is None or d < best[0]:
                        best = (d, a, b)
            _, a, b = best
            self._add_edge(a, b, rng)
            in_tree.add(b)

        # 2) Extra edges between nearby nodes so there are several possible routes
        pairs = [
            (self._euclid(a, b), a, b)
            for i, a in enumerate(self.location_names)
            for b in self.location_names[i + 1:]
            if frozenset((a, b)) not in self.edge_costs
        ]
        pairs.sort()
        candidates = pairs[: num_nodes * 2]       # only the closest pairs are realistic roads
        rng.shuffle(candidates)
        for _, a, b in candidates[: int(num_nodes * extra_edge_ratio)]:
            self._add_edge(a, b, rng)
        return self

    def _add_edge(self, a, b, rng):
        # cost = straight-line distance * detour factor (>= 1.1), rounded UP to 2 decimals
        raw = self._euclid(a, b) * rng.uniform(1.1, 1.5)
        self.edge_costs[frozenset((a, b))] = math.ceil(raw * 100) / 100
        self.graph[a].append(b)
        self.graph[b].append(a)

    def _euclid(self, a, b):
        (x1, y1), (x2, y2) = self.coordinates[a], self.coordinates[b]
        return math.hypot(x1 - x2, y1 - y2)

    # ----------------------------------------------------- methods from proposal
    def get_neighbors(self, node):
        return list(self.graph[node])

    def get_edge_cost(self, a, b):
        return self.edge_costs[frozenset((a, b))]

    def estimate_cost(self, node, goal):
        """Heuristic: straight-line distance. Admissible because every edge cost
        is at least the straight-line distance between its endpoints."""
        return self._euclid(node, goal)

    def distributors(self):
        return [n for n in self.location_names if n != "Office"]

    # --------------------------------------------------------------- JSON store
    def save_to_json(self, path):
        data = {
            "seed": self.seed,
            "location_names": self.location_names,
            "coordinates": self.coordinates,
            "edges": [
                {"a": sorted(k)[0], "b": sorted(k)[1], "cost": c}
                for k, c in sorted(self.edge_costs.items(), key=lambda kv: sorted(kv[0]))
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        env = cls()
        env.seed = data["seed"]
        env.location_names = data["location_names"]
        env.coordinates = {k: tuple(v) for k, v in data["coordinates"].items()}
        env.graph = {n: [] for n in env.location_names}
        for e in data["edges"]:
            env.edge_costs[frozenset((e["a"], e["b"]))] = e["cost"]
            env.graph[e["a"]].append(e["b"])
            env.graph[e["b"]].append(e["a"])
        return env
