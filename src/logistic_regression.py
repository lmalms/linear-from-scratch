from typing import Optional
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from base import BaseEstimator
from logistic_regression_data import *


class LogisticRegressionEstimator(BaseEstimator):

    def __init__(self, fit_intercept: bool):
        super().__init__()
        self.fit_intercept = fit_intercept

    def _fit_estimator(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_method: Optional[str] = None,
            epochs: Optional[int] = None,
            learning_rate: Optional[float] = None
    ) -> None:
        if self.fit_intercept:
            X = np.column_stack([np.ones(shape=(X.shape[0], 1)), X])
        n_samples, n_features = X.shape

        # Fit using gradient descent
        theta = [np.random.randn() for _ in range(n_features)]
        with tqdm.trange(epochs if epochs is not None else 5000) as t:
            for _ in t:
                # Calculate gradient for each sample
                grad = np.array([
                    [-X[i, j] * (y[i] - self.logistic(np.dot(X[i], theta))) for j in range(n_features)]
                    for i in range(n_samples)
                ])
                # Sum over all samples
                grad = np.sum(grad, axis=0)
                theta = theta - (learning_rate * grad if learning_rate is not None else 0.001 * grad)
                loss = self.negative_log_likelihood(X, y, theta=theta)
                t.set_description(f"loss: {loss:.3f} ; theta: {theta}")
        self.theta = theta

    @staticmethod
    def logistic(x: float) -> float:
        return 1. / (1. + np.exp(-x))

    def negative_log_likelihood(self, X: np.ndarray, y: np.ndarray, theta: Optional[np.ndarray] = None) -> float:
        """Loss function (negative log likelihood) for given X, y and theta."""
        self._validate_fit_data(X, y)
        n_samples, _ = X.shape
        if theta is None:
            theta = self.theta
        return np.sum(
            np.array([
                -np.log(self.logistic(np.dot(X[i], theta)))
                if y[i] == 1 else -np.log(1. - self.logistic(np.dot(X[i], theta)))
                for i in range(n_samples)
            ])
        )

    def _predict(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        if self.fit_intercept:
            X = np.column_stack([np.ones(shape=(n_samples, 1)), X])
        class_probabilities = np.array([self.logistic(np.dot(X[i, :], self.theta)) for i in range(n_samples)])
        class_memberships = (class_probabilities >= 0.5).astype(float)
        return np.column_stack([class_memberships.reshape(-1, 1), class_probabilities.reshape(-1, 1)])

    def _validate_fit_data(self, X: np.ndarray, y: np.ndarray) -> None:
        super()._validate_fit_data(X=X, y=y)
        assert all([int(i) in [0, 1] for i in y]), "y should only contain either ones or zeros."

    def _validate_predict_data(self, X: np.ndarray) -> None:
        super()._validate_predict_data(X=X)
        assert self.theta.shape[0] == (X.shape[1] + 1) if self.fit_intercept else X.shape[1], \
            "training and test sets have unequal number of features."


if __name__ == "__main__":
    # Data
    data = np.array([list(row) for row in tuples])
    X, y = data[:, :2], data[:, -1]
    n_samples, _ = X.shape
    # Scale data
    X = np.column_stack([
        ((X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])).reshape(-1, 1)
        if np.std(X[:, j]) > 0 else X[:, j].reshape(-1, 1)
        for j in range(X.shape[1])
    ])

    # Fit model
    model = LogisticRegressionEstimator(fit_intercept=True)
    model.fit(X=X, y=y, fit_method="gradient_descent", epochs=2500, learning_rate=0.002)
    y_hat = model.predict(X=X)[:, 1]
    theta_star = model.theta

    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
    # View the data and decision boundary
    ax[0, 0].scatter(
        [x[0] for x, y in zip(X, y) if y == 1],
        [x[1] for x, y in zip(X, y) if y == 1],
        label="paid",
        marker="o",
        alpha=0.5,
        c="tab:blue"
    )
    ax[0, 0].scatter(
        [x[0] for x, y in zip(X, y) if y == 0],
        [x[1] for x, y in zip(X, y) if y == 0],
        label="free",
        marker="+",
        alpha=0.5,
        c="tab:orange"
    )
    ax[0, 0].plot(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
        np.array([
            -(theta_star[0] + x_i*theta_star[1]) / theta_star[2]
            for x_i in np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        ]),
        c="tab:green",
        lw=2.5,
        ls=":"
    )
    ax[0, 0].set_xlabel("years experience (scaled)")
    ax[0, 0].set_ylabel("annual salary (scaled)")
    ax[0, 0].legend(loc=0, fontsize="small", borderaxespad=.3)
    ax[0, 0].set_title("paid vs. free accounts")

    ax[0, 1].scatter(y_hat, y, marker="+", alpha=0.5)
    ax[0, 1].set_xlabel("predicted")
    ax[0, 1].set_ylabel("actual")
    ax[0, 1].set_title("actual vs. predicted class probabilities")

    ax[1, 0].scatter(
        X[:, 0],
        [
            1./(1.+np.exp(-np.dot(np.array([1, X[i, 0]]), np.array([theta_star[0], theta_star[1]]))))
            for i in range(n_samples)
        ],
        alpha=0.5,
        label="$X_{1}$",
        c="tab:blue"
    )
    ax[1, 0].scatter(X[:, 0], y, marker="+", c="tab:green", alpha=0.25)
    ax[1, 0].scatter(
        X[:, 1],
        [
            1. / (1. + np.exp(-np.dot(np.array([1, X[i, 1]]), np.array([theta_star[0], theta_star[2]]))))
            for i in range(n_samples)
        ],
        alpha=0.5,
        label="$X_{2}$",
        c="tab:orange"
    )
    ax[1, 0].scatter(X[:, 1], y, marker="+", c="tab:green", alpha=0.25)
    ax[1, 0].set_title("features vs. class probabilities")
    ax[1, 0].set_xlabel("$X_{1}, X_{2}$")
    ax[1, 0].set_ylabel("class probabilities")
    ax[1, 0].legend(loc=1, fontsize="small", borderaxespad=.3)

    # Negative log likelihoods for different thetas.
    theta_range = np.linspace(0, 8, 100)
    ax[1, 1].plot(
        theta_range,
        [
            model.negative_log_likelihood(
                np.column_stack([np.ones(shape=(n_samples, 1)), X]),
                y,
                theta=np.array([theta_star[0], theta_j, theta_star[2]])
            )
            for theta_j in theta_range
        ],
        label="$\\theta_{1}$",
        lw=2.5,
        alpha=0.75
    )
    ax[1, 1].plot(
        -theta_range,
        [
            model.negative_log_likelihood(
                np.column_stack([np.ones(shape=(n_samples, 1)), X]),
                y,
                theta=np.array([theta_star[0], theta_star[1], theta_j])
            )
            for theta_j in -theta_range
        ],
        label="$\\theta_{2}$",
        alpha=0.75,
        lw=2.5
    )
    ax[1, 1].set_ylabel("negative log likelihood")
    ax[1, 1].set_xlabel("$\\theta_{1}, \\theta_{2}$")
    ax[1, 1].set_title("loss functions")
    ax[1, 1].legend(loc=0, fontsize="small", borderaxespad=.3)

    fig.tight_layout()
    plt.show()
