"""RouteProblem: defines the search problem (states, actions, goal, cost).

State  = (current_node, frozenset of required distributors already visited)
Action = move to a neighbouring node (the officer may pass through any node;
         a distributor only counts as visited if it is in the required list).
Goal   = all required distributors visited. The officer does NOT return to the
         office (decision made by the team; real officers end the day at the last stop).
         Set return_to_start=True to switch the terminal condition.
"""


class RouteProblem:
    def __init__(self, env, start_node, required_distributors, return_to_start=False):
        missing = [d for d in required_distributors if d not in env.graph]
        if missing:
            raise ValueError(f"Unknown locations: {missing}")
        if start_node in required_distributors:
            raise ValueError("The start node cannot also be a required distributor")
        self.env = env
        self.start_node = start_node
        self.required = frozenset(required_distributors)
        self.return_to_start = return_to_start

    # ----------------------------------------------------------- proposal methods
    def initial_state(self):
        return (self.start_node, frozenset())

    def is_goal(self, state):
        node, visited = state
        if visited != self.required:
            return False
        return node == self.start_node if self.return_to_start else True

    def get_actions(self, state):
        return self.env.get_neighbors(state[0])

    def apply_action(self, state, action):
        _, visited = state
        if action in self.required:
            visited = visited | {action}
        return (action, visited)

    def step_cost(self, state, action):
        return self.env.get_edge_cost(state[0], action)

    def path_cost(self, path):
        return round(sum(self.env.get_edge_cost(a, b) for a, b in zip(path, path[1:])), 2)

    # ----------------------------------------------------------------- heuristic
    def heuristic(self, state):
        """Admissible lower bound on the remaining cost: straight-line distance to the
        farthest unvisited distributor (it must still be reached). 0 once all visited."""
        node, visited = state
        remaining = self.required - visited
        if not remaining:
            if self.return_to_start:
                return self.env.estimate_cost(node, self.start_node)
            return 0.0
        return max(self.env.estimate_cost(node, u) for u in remaining)
