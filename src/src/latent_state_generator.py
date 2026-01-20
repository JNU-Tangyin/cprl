import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HSMMParams:
    n_states: int
    transition_matrix: np.ndarray  # shape: (n_states, n_states)
    initial_probs: np.ndarray      # shape: (n_states,)


class LatentStateGenerator:
    def __init__(self, n_states: int = 3, random_seed: int = 42):
        """
        Simple HSMM-like latent state generator (actually more like a persistent HMM).

        Args:
            n_states:   Number of latent states (e.g., regimes).
            random_seed: Random seed for reproducibility.
        """
        self.n_states = n_states
        self.rng = np.random.RandomState(random_seed)
        self.params: Optional[HSMMParams] = None
        self.current_state: Optional[int] = None

    def initialize_params(
        self,
        data: Optional[np.ndarray] = None,
        method: str = "random",
    ) -> None:
        """
        Initialize model parameters.

        Args:
            data:   Optional data array; reserved for future use
                    (e.g. estimating transition matrix from data).
            method: Initialization method, currently only 'random'.
        """
        if method == "random":
            # Random transition matrix with strong diagonal (state persistence)
            trans_mat = self.rng.dirichlet(
                np.ones(self.n_states) * 0.5, size=self.n_states
            )
            # Encourage staying in the same state
            np.fill_diagonal(trans_mat, trans_mat.diagonal() * 5.0)
            trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)

            # Uniform initial distribution
            initial_probs = np.ones(self.n_states) / float(self.n_states)

            self.params = HSMMParams(
                n_states=self.n_states,
                transition_matrix=trans_mat,
                initial_probs=initial_probs,
            )
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def sample_sequence(self, length: int) -> np.ndarray:
        """
        Sample a sequence of latent states of given length.

        Returns:
            states: np.ndarray of shape (length,)
        """
        if self.params is None:
            self.initialize_params()

        states = np.zeros(length, dtype=int)
        states[0] = self.rng.choice(
            self.n_states, p=self.params.initial_probs
        )

        for t in range(1, length):
            states[t] = self.rng.choice(
                self.n_states,
                p=self.params.transition_matrix[states[t - 1]],
            )

        # 记录最后一个 state，方便外部访问
        self.current_state = int(states[-1])
        return states

    def get_state_durations(self) -> np.ndarray:
        """
        Approximate expected duration for each state under geometric
        sojourn time implied by self-transition probability.

        E[duration_i] = 1 / (1 - P(i->i))
        """
        if self.params is None:
            raise ValueError("Model parameters not initialized")
        diag = np.diag(self.params.transition_matrix)
        if np.any(diag >= 1.0):
            raise ValueError("Self-transition probability must be < 1 for all states")
        return 1.0 / (1.0 - diag)

    def get_next_state_probs(self, current_state: int) -> np.ndarray:
        """
        Get probability distribution over next states given current_state.
        """
        if self.params is None:
            raise ValueError("Model parameters not initialized")
        return self.params.transition_matrix[current_state]

    def get_most_probable_next_state(self, current_state: int) -> int:
        """
        Get the argmax of next-state distribution.
        """
        return int(np.argmax(self.get_next_state_probs(current_state)))
