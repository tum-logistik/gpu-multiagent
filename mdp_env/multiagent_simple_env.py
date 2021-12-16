import torch
import numpy as np
from tensorforce import Agent, Environment, Runner

def torch_lib_check():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

class MultiactorEnvironment(Environment):
    """
    Example multi-actor environment, illustrating best-practice implementation pattern.
    State space: position in [0, 10].
    Action space: movement in {-1, 0, 1}.
    Random start in [3, 7].
    Actor 1 perspective as is, actor 2 perspective mirrored.
    Positive reward for being closer to 10.
    """

    def __init__(self):
        super().__init__()

    def states(self):
        return dict(type='int', num_values=11)

    def actions(self):
        return dict(type='int', num_values=3)

    def num_actors(self):
        return 2  # Indicates that environment has multiple actors

    def reset(self, num_parallel = 1):
        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(num_parallel)

        # Single shared environment logic, plus per-actor perspective
        self._states = 3 + np.random.randint(5)
        self.second_actor = True
        states = np.stack([self._states, 10 - self._states], axis=0)

        # Always for multi-actor environments: return per-actor values
        return self._parallel_indices.copy(), states

    def execute(self, actions):
        # Single shared environment logic, plus per-actor perspective
        if self.second_actor:
            self.second_actor = self.second_actor and not (np.random.random_sample() < 0.1)
            terminal = np.stack([False, False], axis=0)
            delta = (actions[0] - 1) - (actions[1] - 1)
            rand_val = np.clip(self._states + delta, a_min=0, a_max=10)
            states = np.stack([self._states, rand_val], axis=0)
        else:
            terminal = np.stack([False, False], axis=0)
            delta = (actions[0] - 1)
            self._states = np.clip(self._states + delta, a_min=0, a_max=10)
            states = np.stack([self._states, self._states], axis=0)
        reward = np.stack([self._states, 10 - self._states], axis=0)

        # Always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]
        return self._parallel_indices.copy(), states, terminal, reward
    
    def is_vectorizable(self):
        return True

