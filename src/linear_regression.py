from typing import Optional

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from base import BaseEstimator


class LinearRegressionEstimator(BaseEstimator):

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
            X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        n_samples, n_features = X.shape
        if (fit_method is None) or (fit_method == "exact"):
            self.theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        elif fit_method == "gradient_descent":
            # Initialise random theta
            self.theta = np.random.rand(n_features)
            for i in range(epochs if epochs is not None else 5000):
                # Calculate gradient
                grad = np.array([
                    (-2.0 / n_samples) * np.dot(X[:, j], (y - np.dot(X[:, j], self.theta[j])))
                    for j in range(n_features)
                ])
                # Update theta
                self.theta = self.theta - (learning_rate * grad if learning_rate is not None else 0.001 * grad)

        elif fit_method == "cvxpy":
            theta = cp.Variable(shape=(n_features,))
            loss = cp.sum_squares(y - X @ theta) / n_samples
            _ = cp.Problem(cp.Minimize(loss)).solve()
            self.theta = theta.value

    def _predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_predict_data(X=X)
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        return np.dot(X, self.theta)

    def _validate_predict_data(self, X: np.ndarray) -> None:
        super()._validate_predict_data(X=X)
        assert self.theta.shape[0] == (X.shape[1] + 1) if self.fit_intercept else X.shape[1], \
            "training and test sets have unequal number of features."


if __name__ == "__main__":

    # Simple linear regression
    np.random.seed(0)
    n_samples = 200
    n_features = 1

    true_theta = np.array([3.75, 2.4])
    error = np.random.rand(n_samples)
    X_train = np.random.rand(n_samples, n_features)

    # Feature Scaling
    X_train = np.concatenate([
        (X_train[:, j] - np.mean(X_train[:, j])) / np.std(X_train[:, j])
        for j in range(n_features)
    ]).reshape(n_samples, n_features)

    # Generate test data
    X_test = np.concatenate([
        np.linspace(X_train[:, j].min(), X_train[:, j].max(), n_samples)
        for j in range(n_features)
    ]).reshape(n_samples, n_features)

    y_train = np.dot(
        np.concatenate([np.ones_like(X_train), X_train], axis=1),
        true_theta
    ) + 5.5 * error

    # Initialise model
    lr_estimator = LinearRegressionEstimator(fit_intercept=True)

    # Fit using exact method
    lr_estimator.fit(X_train, y_train, fit_method="exact")
    y_hat_exact = lr_estimator.predict(X_test)

    # Fit using gradient descent
    lr_estimator.fit(X_train, y_train, fit_method="gradient_descent", epochs=2500, learning_rate=0.001)
    y_hat_grad_descent = lr_estimator.predict(X_test)

    # Fit using cvxpy
    lr_estimator.fit(X_train, y_train, fit_method="cvxpy")
    y_hat_cvxpy = lr_estimator.predict(X_test)

    fig1, ax1 = plt.subplots(1, 1)
    ax1.scatter(X_train, y_train, alpha=0.75, lw=2, label="training data")
    ax1.plot(X_test, y_hat_exact, color="tab:orange", lw=2.5, ls="-", label="exact")
    ax1.plot(X_test, y_hat_grad_descent, color="tab:green", lw=2.5, ls="--", label="gradient descent")
    ax1.plot(X_test, y_hat_cvxpy, color="tab:red", lw=2.5, ls="-.", label="cvxpy")
    ax1.legend(loc=0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Simple Linear Regression")

    fig1.tight_layout()

    # Multivariate Regression
    n_samples = 200
    n_features = 4

    true_theta = np.array([5.75, -8.4, 10.3, -7.3])
    X_train = np.random.rand(n_samples, n_features)

    # Feature Scaling
    X_train = np.concatenate([
        (X_train[:, j] - np.mean(X_train[:, j])) / np.std(X_train[:, j])
        for j in range(n_features)
    ]).reshape(n_samples, n_features)

    X_test = np.concatenate([
        np.linspace(X_train[:, j].min(), X_train[:, j].max(), n_samples)
        for j in range(n_features)
    ]).reshape(n_samples, n_features)

    y_train = np.dot(X_train, true_theta)

    # Initialise model
    lr_estimator = LinearRegressionEstimator(fit_intercept=False)

    color_map = {
        "exact": "tab:orange",
        "gradient descent": "tab:green",
        "cvxpy": "tab:red"
    }
    ls_map = {
        "exact": "-",
        "gradient descent": "--",
        "cvxpy": "-."
    }

    fig2, ax2 = plt.subplots(1, n_features, figsize=(12, 3.5), sharey="row")
    for j in range(n_features):
        ax2[j].scatter(X_train[:, j], y_train, lw=2.5, ls="-", label="training data", alpha=0.5)
        for fit_method in ["exact", "gradient descent", "cvxpy"]:
            lr_estimator.fit(X_train, y_train, fit_method=fit_method)
            ax2[j].plot(
                X_test[:, j], np.dot(X_test[:, j], lr_estimator.theta[j]),
                lw=2.5, ls=ls_map[fit_method], color=color_map[fit_method], label=fit_method
            )
            ax2[j].legend(loc=0, fontsize="small")
            ax2[j].set_title(f"feature {j+1}", fontsize="medium")
            ax2[j].set_xlabel("x")
    ax2[0].set_ylabel("y")
    fig2.suptitle("Multivariate Linear Regression")
    fig2.tight_layout()
    plt.show()


