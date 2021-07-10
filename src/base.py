from typing import Optional
from abc import abstractmethod, ABC
import numpy as np


class BaseEstimator(ABC):

    def __init__(self):
        self.theta = None

    def fit(
            self, X: np.ndarray,
            y: np.ndarray,
            fit_method: Optional[str] = None,
            epochs: Optional[int] = None,
            learning_rate: Optional[float] = None
    ) -> None:
        self._validate_fit_data(X=X, y=y)
        self._fit_estimator(X=X, y=y, fit_method=fit_method, epochs=epochs, learning_rate=learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_predict_data(X=X)
        return self._predict(X=X)

    @abstractmethod
    def _fit_estimator(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_method: Optional[str] = None,
            epochs: Optional[int] = None,
            learning_rate: Optional[float] = None
    ) -> None:
        ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    def _validate_fit_data(X: np.ndarray, y: np.ndarray) -> None:
        assert X.ndim == 2, "features should be 2D-array. Use X.reshape(-1, 1) if only a single feature present."
        assert y.ndim == 1, "labels should be passed as a 1D-array. Use y.flatten()."
        assert X.shape[0] == y.shape[0], "X and y should have the same length."

    @staticmethod
    def _validate_predict_data(X: np.ndarray) -> None:
        assert X.ndim == 2, "features should be passed in as a 2D-array. " \
                            "Use X.reshape(-1, 1) if only a single feature present."
