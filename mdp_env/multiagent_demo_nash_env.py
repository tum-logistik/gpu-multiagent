import torch
import numpy as np
from tensorforce import Agent, Environment, Runner

def torch_lib_check():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

class DemoNashCoopEnv(Environment):
    """
    Integer action space, [-2, +2]
    2 agents
    Reward = sum of a1 + a1 only if |a1 - a2| = 1
    NEQ,
    p 2  -> p' 1
    p 1  -> p' 2
    p 0  -> p' 1
    p -1 -> p' 0
    p -2 -> p' 0

    """

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='int', num_values=11)

    def actions(self):
        return dict(type='int', num_values=3)

    def num_actors(self):
        return 1  # Indicates that environment has multiple actors
    
    def reset(self, num_parallel = 1):
        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(num_parallel)

        # Single shared environment logic, plus per-actor perspective
        self._states = np.random.randint(0, high=10)
        self.second_actor = True

        states = np.stack([self._states, self._states], axis=0)

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), states

    def execute(self, actions):
        # Single shared environment logic, plus per-actor perspective
        if np.abs(actions[0] - actions[1]) == 1:
            reward = np.max(self._states) + actions[0] + actions[1]
        else:
            reward = np.max(self._states)
        
        self._states = reward
        states = np.stack([self._states, self._states], axis=0)
        terminal = np.stack([False, False], axis=0)
        reward = np.stack([reward, reward], axis=0)

        # Always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]
        return self._parallel_indices.copy(), states, terminal, reward
    
    def is_vectorizable(self):
        return True

