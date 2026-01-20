import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    n_actions: int = 5  # Number of discrete actions
    state_dim: int = 6  # Dimension of state space

class QLearningCalibrator:
    def __init__(self, config: Optional[QLearningConfig] = None):
        self.config = config if config is not None else QLearningConfig()
        self.q_table = {}
        self.exploration_rate = self.config.exploration_rate
        self.state_encoder = {}  # For discretizing continuous states
        self.state_counter = 0
    
    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete key for Q-table"""
        # Simple binning approach - in practice, you might want something more sophisticated
        # like tile coding or neural network approximation for high-dimensional spaces
        state_key = "_".join(f"{s:.2f}" for s in state)
        if state_key not in self.state_encoder:
            self.state_encoder[state_key] = self.state_counter
            self.state_counter += 1
        return state_key
    
    def get_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy"""
        state_key = self._get_state_key(state)
        
        # Initialize Q-values for this state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.config.n_actions)
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.config.n_actions)
        
        # Exploitation: best action from Q-table
        return np.argmax(self.q_table[state_key])
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        """Update Q-values using the Q-learning update rule"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values if not present
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.config.n_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.config.n_actions)
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.config.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.config.learning_rate * td_error
        
        # Decay exploration rate
        self.exploration_rate = max(
            self.config.min_exploration,
            self.exploration_rate * self.config.exploration_decay
        )
    
    def get_optimal_action(self, state: np.ndarray) -> int:
        """Get the optimal action for a given state (no exploration)"""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            return np.random.randint(0, self.config.n_actions)
        return np.argmax(self.q_table[state_key])