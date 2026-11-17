"""AStarAgent and BFSAgent: both solve the same RouteProblem state space.

State = (current_node, frozenset(visited distributors)).
Both agents record the same statistics so the comparison is fair:
    nodes_expanded : states popped from the frontier and expanded
    runtime_ms     : wall-clock time of solve()
    path_cost      : weighted travel cost of the returned route
    path_edges     : number of roads used
"""

import heapq
import time
from collections import deque


class _BaseAgent:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.parents = {}          # state -> (previous_state, node moved to)
        self.explored = set()      # states already expanded
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.runtime_ms = 0.0
        self.path = None
        self.path_cost = None
        self.problem = None

    def reconstruct_path(self, state):
        """Follow parent links back to the start; return the list of locations."""
        nodes = [state[0]]
        while self.parents[state] is not None:
            state, _ = self.parents[state]
            nodes.append(state[0])
        return nodes[::-1]

    def _finish(self, goal_state, started):
        self.runtime_ms = (time.perf_counter() - started) * 1000
        if goal_state is None:
            return None
        self.path = self.reconstruct_path(goal_state)
        self.path_cost = self.problem.path_cost(self.path)
        return self.path

    def get_statistics(self):
        return {
            "found": self.path is not None,
            "path": self.path,
            "path_cost": self.path_cost,
            "path_edges": len(self.path) - 1 if self.path else None,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "runtime_ms": round(self.runtime_ms, 4),
        }


class AStarAgent(_BaseAgent):
    """A* search with f(n) = g(n) + h(n).

    Open list: heapq of (f, tie_breaker, g, state).
    h(n) is the problem's admissible (and consistent) straight-line heuristic.
    use_heuristic=False sets h = 0, which makes A* behave like uniform-cost search
    (the fallback described in the proposal, and our independent optimality check).
    """

    def __init__(self, use_heuristic=True):
        self.use_heuristic = use_heuristic
        super().__init__()

    def _h(self, state):
        return self.problem.heuristic(state) if self.use_heuristic else 0.0

    def solve(self, problem):
        self._reset()
        self.problem = problem
        started = time.perf_counter()

        start = problem.initial_state()
        self.parents[start] = None
        best_g = {start: 0.0}                      # cheapest known cost to each state
        counter = 0                                # tie-breaker: first in, first out
        open_list = [(self._h(start), counter, 0.0, start)]
        self.nodes_generated = 1

        while open_list:
            _, _, g, state = heapq.heappop(open_list)
            if state in self.explored:             # stale entry (a cheaper one was already used)
                continue
            if problem.is_goal(state):             # goal test on pop -> guarantees optimality
                return self._finish(state, started)

            self.explored.add(state)
            self.nodes_expanded += 1

            for action in problem.get_actions(state):
                nxt = problem.apply_action(state, action)
                new_g = g + problem.step_cost(state, action)
                if nxt in self.explored:
                    continue
                if new_g < best_g.get(nxt, float("inf")):
                    best_g[nxt] = new_g
                    self.parents[nxt] = (state, action)
                    counter += 1
                    heapq.heappush(open_list, (new_g + self._h(nxt), counter, new_g, nxt))
                    self.nodes_generated += 1

        return self._finish(None, started)         # no route exists


class BFSAgent(_BaseAgent):
    """Breadth-first search: finds the route with the FEWEST ROADS, ignoring edge weights.

    Its travel cost is measured afterwards on the weighted map (problem.path_cost),
    so it is a baseline, not a guarantee of the cheapest route.
    The goal test is done when a state is generated (standard for BFS).
    """

    def solve(self, problem):
        self._reset()
        self.problem = problem
        started = time.perf_counter()

        start = problem.initial_state()
        self.parents[start] = None
        self.nodes_generated = 1
        if problem.is_goal(start):
            return self._finish(start, started)

        frontier = deque([start])                  # FIFO queue
        seen = {start}                             # generated states (frontier + explored)

        while frontier:
            state = frontier.popleft()
            self.explored.add(state)
            self.nodes_expanded += 1

            for action in problem.get_actions(state):
                nxt = problem.apply_action(state, action)
                if nxt in seen:
                    continue
                seen.add(nxt)
                self.parents[nxt] = (state, action)
                self.nodes_generated += 1
                if problem.is_goal(nxt):
                    return self._finish(nxt, started)
                frontier.append(nxt)

        return self._finish(None, started)
