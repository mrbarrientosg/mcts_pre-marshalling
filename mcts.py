import numpy as np
import random
import copy
from model import MarshallingState, State

class Node:

    def __init__(self, state: State, action = None):
        self.visits: int = 0
        self.reward: int = 0
        self.reward2: int = 0
        self.parent: Node = None
        self.children: list = list()
        self.actions: list = state.get_actions()
        self.action = action

    def best_child(self):
        return max(self.children, key=lambda child: child.get_uct())

    def get_uct(self) -> float:
        c = 0.1
        w = self.reward
        n = self.visits
        sumsq = self.reward2

        if self.parent is None:
            t = self.visits
        else:
            t = self.parent.visits

        UTC = w/n + c * np.sqrt(np.log(t)/n)

        D = 32.0
        Modification = np.sqrt((sumsq - n * (w/n)**2 + D)/n)
        return UTC + Modification

    def update(self, reward):
        self.visits += 1
        self.reward += reward
        self.reward2 += reward**2

class MCTS:

    def __init__(self):
        self.best_state = None

    def _expand(self, node: Node, state: State):
        action = random.sample(node.actions, 1)[0]
        state.transition(action)
        node.actions.remove(action)

        new = Node(state, action)
        new.parent = node
        node.children.append(new)
        return new

    def _tree_policy(self, node: Node, state: State):
        current = node

        while not state.is_terminal():
            if current.actions:
                return self._expand(current, state)
            else:
                current = current.best_child()
                state.transition(current.action)

        return current

    def search(self, root_state: State):
        root = Node(root_state)
        current = None
        state = None

        for _ in range(100):
            state = copy.deepcopy(root_state)
            current = self._tree_policy(root, state)

            state.simulation()

            if self.best_state is None:
                self.best_state = copy.deepcopy(state)
            elif self.best_state.get_reward() > state.get_reward() and state.is_terminal():
                self.best_state = copy.deepcopy(state)

            while current is not None:
                current.update(state.get_reward())
                current = current.parent
