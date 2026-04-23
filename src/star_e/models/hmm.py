"""Hidden Markov Model for market regime detection."""

from typing import Literal, Optional

import mlflow
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from star_e.models.base import BaseRegimeDetector


class RegimeHMM(BaseRegimeDetector):
    """
    Hidden Markov Model for market regime detection.

    Identifies latent market states (Bull, Bear, Sideways) from observable
    features like returns and volatility using the Baum-Welch algorithm
    for training and Viterbi algorithm for decoding.

    Attributes:
        n_states: Number of hidden states (default: 3 for Bull/Bear/Sideways)
        model: Trained GaussianHMM model
        scaler: StandardScaler for feature normalization
        state_labels: Human-readable state names ordered by mean return
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: Literal["full", "diag", "spherical", "tied"] = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize RegimeHMM.

        Args:
            n_states: Number of hidden states
            covariance_type: Type of covariance matrix
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.state_labels = ["Bear", "Sideways", "Bull"]
        self._state_order: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, features: np.ndarray) -> "RegimeHMM":
        """
        Fit HMM to feature matrix using Baum-Welch algorithm.

        Args:
            features: (n_samples, n_features) array. First column should be
                      returns for proper state ordering.

        Returns:
            Self for method chaining
        """
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled)

        # Reorder states by mean return (ascending: Bear < Sideways < Bull)
        state_means = self.model.means_[:, 0]
        self._state_order = np.argsort(state_means)
        self._is_fitted = True

        # Log to MLflow if active
        if mlflow.active_run():
            mlflow.log_params(
                {
                    "hmm_n_states": self.n_states,
                    "hmm_covariance_type": self.covariance_type,
                    "hmm_n_iter": self.n_iter,
                    "hmm_converged": self.model.monitor_.converged,
                }
            )

        return self

    def decode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode most likely state sequence using Viterbi algorithm.

        Args:
            features: (n_samples, n_features) feature matrix

        Returns:
            Tuple of:
                - state_sequence: Array of state indices (0=Bear, 1=Sideways, 2=Bull)
                - state_probabilities: (n_samples, n_states) probability matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before decoding")

        X_scaled = self.scaler.transform(features)
        log_prob, states = self.model.decode(X_scaled, algorithm="viterbi")

        # Remap to ordered states (Bear < Sideways < Bull)
        ordered_states = np.array(
            [np.where(self._state_order == s)[0][0] for s in states]
        )

        # Get state probabilities
        state_probs = self.model.predict_proba(X_scaled)
        # Reorder probability columns
        state_probs = state_probs[:, self._state_order]

        return ordered_states, state_probs

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict most likely state for each observation."""
        states, _ = self.decode(features)
        return states

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get state transition probability matrix.

        Returns:
            (n_states, n_states) matrix where entry [i,j] is P(state_j | state_i)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Reorder according to Bear < Sideways < Bull
        trans = self.model.transmat_[self._state_order][:, self._state_order]
        return trans

    def expected_duration(self) -> dict[str, float]:
        """
        Calculate expected duration in each state.

        The expected duration in state i is 1 / (1 - P(stay in state i)).

        Returns:
            Dictionary mapping state labels to expected duration in periods
        """
        trans = self.get_transition_matrix()
        durations = {}
        for i, label in enumerate(self.state_labels[: self.n_states]):
            p_stay = trans[i, i]
            if p_stay >= 1:
                durations[label] = float("inf")
            else:
                durations[label] = 1 / (1 - p_stay)
        return durations

    def steady_state_probabilities(self) -> dict[str, float]:
        """
        Calculate steady-state (long-run) probabilities of each state.

        Returns:
            Dictionary mapping state labels to steady-state probabilities
        """
        trans = self.get_transition_matrix()

        # Solve pi * P = pi with sum(pi) = 1
        n = len(trans)
        A = np.vstack([trans.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1

        pi = np.linalg.lstsq(A, b, rcond=None)[0]

        return {
            label: float(pi[i]) for i, label in enumerate(self.state_labels[: self.n_states])
        }

    def select_n_states(
        self,
        features: np.ndarray,
        state_range: range = range(2, 6),
        criterion: Literal["aic", "bic"] = "bic",
    ) -> int:
        """
        Select optimal number of states using information criteria.

        Args:
            features: Feature matrix for model selection
            state_range: Range of state counts to evaluate
            criterion: Information criterion ("aic" or "bic")

        Returns:
            Optimal number of states
        """
        X_scaled = self.scaler.fit_transform(features)
        scores = {}

        for n in state_range:
            try:
                model = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                )
                model.fit(X_scaled)

                log_likelihood = model.score(X_scaled)
                n_features = X_scaled.shape[1]

                # Number of free parameters
                if self.covariance_type == "full":
                    n_cov_params = n * n_features * (n_features + 1) // 2
                elif self.covariance_type == "diag":
                    n_cov_params = n * n_features
                else:
                    n_cov_params = n

                n_params = (
                    n * n  # Transition matrix
                    + n * n_features  # Means
                    + n_cov_params  # Covariance
                    + n - 1  # Initial state
                )

                n_samples = X_scaled.shape[0]

                if criterion == "aic":
                    scores[n] = 2 * n_params - 2 * log_likelihood
                else:  # bic
                    scores[n] = n_params * np.log(n_samples) - 2 * log_likelihood

            except Exception as e:
                print(f"Warning: Could not fit HMM with {n} states: {e}")
                continue

        if not scores:
            raise ValueError("Could not fit HMM with any number of states")

        optimal = min(scores, key=scores.get)

        # Log to MLflow if active
        if mlflow.active_run():
            mlflow.log_metrics(
                {f"hmm_{criterion}_{n}_states": score for n, score in scores.items()}
            )
            mlflow.log_metric(f"hmm_optimal_states_{criterion}", optimal)

        return optimal

    def get_state_statistics(self) -> dict[str, dict]:
        """
        Get mean and covariance statistics for each state.

        Returns:
            Dictionary with state statistics
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        stats = {}
        for i, label in enumerate(self.state_labels[: self.n_states]):
            orig_idx = self._state_order[i]
            stats[label] = {
                "mean": self.model.means_[orig_idx].tolist(),
                "covariance": self.model.covars_[orig_idx].tolist(),
            }
        return stats
