"""
Gaussian Mixture Model for market regime detection.

GMM provides a complementary approach to HMM for identifying market states,
without assuming temporal dependencies between observations.
"""

import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Optional, Literal
import mlflow


class RegimeGMM:
    """
    Gaussian Mixture Model for market regime clustering.

    Identifies market regimes by clustering return/volatility features
    into distinct Gaussian components.

    Unlike HMM, GMM does not model temporal transitions but can be
    combined with HMM for ensemble regime detection.

    Attributes:
        n_components: Number of mixture components (regimes)
        covariance_type: Type of covariance parameters
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        max_iter: int = 200,
        n_init: int = 10,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.state_labels = ["Bear", "Sideways", "Bull"]
        self._state_order: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, features: np.ndarray) -> "RegimeGMM":
        """
        Fit GMM to feature matrix.

        Args:
            features: (n_samples, n_features) array containing
                     returns, volatility, and other indicators

        Returns:
            Self for chaining
        """
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled)

        state_means = self.model.means_[:, 0]
        self._state_order = np.argsort(state_means)
        self._fitted = True

        mlflow.log_params({
            "gmm_n_components": self.n_components,
            "gmm_covariance_type": self.covariance_type,
            "gmm_n_init": self.n_init,
        })
        mlflow.log_metrics({
            "gmm_aic": self.model.aic(X_scaled),
            "gmm_bic": self.model.bic(X_scaled),
            "gmm_converged": float(self.model.converged_),
        })

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for new data.

        Args:
            features: Feature matrix

        Returns:
            Array of regime labels (0=Bear, 1=Sideways, 2=Bull)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(features)
        raw_labels = self.model.predict(X_scaled)

        ordered_labels = np.array([
            np.where(self._state_order == label)[0][0]
            for label in raw_labels
        ])

        return ordered_labels

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            features: Feature matrix

        Returns:
            (n_samples, n_components) probability matrix
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(features)
        probs = self.model.predict_proba(X_scaled)

        return probs[:, self._state_order]

    def get_component_params(self) -> dict:
        """Get means and covariances for each component."""
        ordered_means = self.model.means_[self._state_order]
        ordered_covs = self.model.covariances_[self._state_order]
        ordered_weights = self.model.weights_[self._state_order]

        return {
            "means": ordered_means,
            "covariances": ordered_covs,
            "weights": ordered_weights,
            "labels": self.state_labels[:self.n_components],
        }

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Compute log-likelihood for each sample."""
        X_scaled = self.scaler.transform(features)
        return self.model.score_samples(X_scaled)

    def select_n_components(
        self,
        features: np.ndarray,
        component_range: range = range(2, 8),
        criterion: Literal["aic", "bic"] = "bic",
    ) -> int:
        """
        Select optimal number of components using information criterion.

        Args:
            features: Feature matrix
            component_range: Range of components to test
            criterion: "aic" or "bic"

        Returns:
            Optimal number of components
        """
        X_scaled = self.scaler.fit_transform(features)
        scores = {}

        for n in component_range:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                n_init=self.n_init,
                random_state=self.random_state,
            )
            gmm.fit(X_scaled)

            if criterion == "aic":
                scores[n] = gmm.aic(X_scaled)
            else:
                scores[n] = gmm.bic(X_scaled)

        optimal = min(scores, key=scores.get)

        mlflow.log_metrics({
            f"gmm_{criterion}_{n}_components": score
            for n, score in scores.items()
        })
        mlflow.log_metric(f"gmm_optimal_components_{criterion}", optimal)

        return optimal


class BayesianRegimeGMM:
    """
    Bayesian Gaussian Mixture for automatic component selection.

    Uses Dirichlet process prior to automatically determine
    the number of regimes from the data.
    """

    def __init__(
        self,
        n_components: int = 10,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        weight_concentration_prior: float = 0.1,
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.weight_concentration_prior = weight_concentration_prior
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self._fitted = False
        self._active_components: Optional[int] = None

    def fit(self, features: np.ndarray) -> "BayesianRegimeGMM":
        """Fit Bayesian GMM."""
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled)

        self._active_components = np.sum(self.model.weights_ > 0.01)
        self._fitted = True

        mlflow.log_params({
            "bgmm_max_components": self.n_components,
            "bgmm_weight_prior": self.weight_concentration_prior,
        })
        mlflow.log_metrics({
            "bgmm_active_components": self._active_components,
            "bgmm_converged": float(self.model.converged_),
        })

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels."""
        X_scaled = self.scaler.transform(features)
        return self.model.predict(X_scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict regime probabilities."""
        X_scaled = self.scaler.transform(features)
        return self.model.predict_proba(X_scaled)

    @property
    def active_components(self) -> int:
        """Number of active components (with significant weight)."""
        return self._active_components


class EnsembleRegimeDetector:
    """
    Ensemble of HMM and GMM for robust regime detection.

    Combines temporal (HMM) and clustering (GMM) approaches
    for more robust regime identification.
    """

    def __init__(
        self,
        n_states: int = 3,
        hmm_weight: float = 0.6,
        gmm_weight: float = 0.4,
    ):
        from star_e.models.hmm import RegimeHMM

        self.n_states = n_states
        self.hmm_weight = hmm_weight
        self.gmm_weight = gmm_weight

        self.hmm = RegimeHMM(n_states=n_states)
        self.gmm = RegimeGMM(n_components=n_states)
        self._fitted = False

    def fit(self, features: np.ndarray) -> "EnsembleRegimeDetector":
        """Fit both HMM and GMM."""
        self.hmm.fit(features)
        self.gmm.fit(features)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regimes using ensemble voting.

        Returns the regime with highest combined probability.
        """
        _, hmm_probs = self.hmm.decode(features)
        gmm_probs = self.gmm.predict_proba(features)

        combined_probs = (
            self.hmm_weight * hmm_probs +
            self.gmm_weight * gmm_probs
        )

        return np.argmax(combined_probs, axis=1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get ensemble probability estimates."""
        _, hmm_probs = self.hmm.decode(features)
        gmm_probs = self.gmm.predict_proba(features)

        return (
            self.hmm_weight * hmm_probs +
            self.gmm_weight * gmm_probs
        )

    def get_model_agreement(self, features: np.ndarray) -> float:
        """
        Calculate agreement rate between HMM and GMM predictions.

        High agreement suggests robust regime identification.
        """
        hmm_states, _ = self.hmm.decode(features)
        gmm_states = self.gmm.predict(features)

        agreement = np.mean(hmm_states == gmm_states)
        return agreement
