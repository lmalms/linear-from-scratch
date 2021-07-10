from typing import Optional

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from base import BaseEstimator


class RidgeRegressionEstimator(BaseEstimator):

    def __init__(self, fit_intercept: bool, l2_penalty: float) -> None:
        super().__init__()
        self.fit_intercept = fit_intercept
        self.l2_penalty = l2_penalty

    def _fit_estimator(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_method: Optional[str] = None,
            epochs: Optional[int] = None,
            learning_rate: Optional[int] = None
    ) -> None:
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        n_samples, n_features = X.shape
        if (fit_method is None) or (fit_method == "gradient_descent"):
            self.theta = np.random.rand(n_features)
            for epoch in range(epochs if epochs is not None else 5000):
                if self.fit_intercept:
                    grad_0 = np.array([
                        (-2.0 / n_samples) * np.dot(X[:, 0], y - np.dot(X[:, 0], self.theta[0]))
                    ])
                    grad = np.concatenate(
                        [
                            grad_0,
                            np.array([-2. * (
                                    np.dot(X[:, j], y - np.dot(X[:, j], self.theta[j])) / n_samples
                                    - self.l2_penalty * self.theta[j]
                            ) for j in range(1, n_features)])
                        ]
                    )
                else:
                    grad = np.array([
                        -2. * (
                            np.dot(X[:, j], y - np.dot(X[:, j], self.theta[j])) / n_samples
                            - self.l2_penalty * self.theta[j]
                        )
                        for j in range(n_features)
                    ])

                self.theta = self.theta - (learning_rate * grad if learning_rate is not None else 0.001 * grad)

        else:  # fit using cvxpy
            theta = cp.Variable(shape=(n_features,))
            mse = cp.sum_squares(y - X@theta) / n_samples
            l2_reg = self.l2_penalty * cp.pnorm(theta[1:] if self.fit_intercept else theta, p=2) ** 2
            loss = mse + l2_reg
            _ = cp.Problem(cp.Minimize(loss)).solve()
            self.theta = theta.value

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        return np.dot(X, self.theta)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        n_samples, _ = X.shape
        return np.sum((y - np.dot(X, self.theta)) ** 2) / n_samples

    def _validate_predict_data(self, X: np.ndarray) -> None:
        super()._validate_predict_data(X=X)
        assert self.theta.shape[0] == (X.shape[1] + 1) if self.fit_intercept else X.shape[1], \
            "training and test sets have unequal number of features."


def compute_ridge_loss(
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        l2_penalty: float,
        includes_intercept: bool
) -> float:
    n_samples, n_features = X.shape
    errors = [y[i] - sum([X[i, j] * theta[j] for j in range(n_features)]) for i in range(n_samples)]
    mse = sum([error_i ** 2 for error_i in errors]) / n_samples
    reg = (
        l2_penalty * sum([theta[j] ** 2 for j in range(1, n_features)])
        if includes_intercept else l2_penalty * sum([theta[j] ** 2 for j in range(n_features)])
    )
    return mse + reg


if __name__ == "__main__":
    np.random.seed(0)

    # Generate some sample data
    n_samples, n_features = 500, 1

    X = np.random.randn(n_samples, n_features)
    theta_true = np.array([3.5, 2.75])
    error = np.random.randn(n_samples)
    y = np.dot(
        np.concatenate([np.ones_like(X), X], axis=1),
        theta_true
    ) + 0.5 * error

    fig, ax = plt.subplots(2, 2, figsize=(8, 7))

    ax[0, 0].scatter(X, y, alpha=0.5)
    for i, l2_penalty in enumerate(np.linspace(0., 10., 20)):
        model = RidgeRegressionEstimator(fit_intercept=True, l2_penalty=l2_penalty)
        model.fit(X=X, y=y, fit_method="gradient_descent", epochs=2500, learning_rate=0.001)
        y_hat = model.predict(
            X=np.linspace(X.min(), X.max(), n_samples).reshape(n_samples, n_features)
        )
        ax[0, 0].plot(
            np.linspace(X.min(), X.max(), n_samples),
            y_hat,
            label=f"$\lambda_{2}$={np.round(l2_penalty,1)}" if i % 2 == 0 else None,
            ls="--",
            lw=2.
        )
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("y")
    ax[0, 0].set_title("Ridge Regression")
    ax[0, 0].legend(ncol=2, loc=0, fontsize="small", borderaxespad=.3)

    # Loss vs. theta for different l2_penalties
    n_samples, n_features = 500, 1
    theta_true = np.array([3.5, 2.75])
    X = np.concatenate([np.ones(shape=(n_samples, n_features)), np.random.randn(n_samples, n_features)], axis=1)
    error = np.random.randn(n_samples)
    y = np.dot(X, theta_true) + 0.5 * error

    slopes = np.linspace(-2.25, 7.75, 100)
    for i, l2_penalty in enumerate(np.linspace(0., 2.5, 10)):
        ax[0, 1].plot(
            slopes,
            [
                compute_ridge_loss(theta=np.array([3.5, slope]), X=X, y=y, l2_penalty=l2_penalty,
                                   includes_intercept=True)
                for slope in slopes
            ],
            label=f"$\lambda_{2}$={np.round(l2_penalty, 1)}" if i % 2 == 0 else None,
            lw=2.5
        )
    ax[0, 1].legend(fontsize="small", borderaxespad=.3, loc=0, ncol=2)
    ax[0, 1].set_title(f"Loss as a function of $\\theta$ and $\lambda$")
    ax[0, 1].set_xlabel("$\\theta_{1}$")
    ax[0, 1].set_ylabel("loss")

    # Plot train vs test losses
    n_samples, n_features = 100, 20
    sigma = 2.5
    test_size = 0.5
    n_test = int(n_samples * test_size)
    X = np.random.randn(n_samples, n_features)
    theta_star = np.random.randn(n_features)
    y = np.dot(X, theta_star) + np.random.normal(0, sigma, size=n_samples)

    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

    l2_penalties = np.logspace(-2, 2, 50)
    theta_values = []
    train_mse = []
    test_mse = []
    for l2_penalty in l2_penalties:
        model = RidgeRegressionEstimator(fit_intercept=False, l2_penalty=l2_penalty)
        model.fit(X=X_train, y=y_train, fit_method="cvxpy")
        theta_values.append(model.theta)
        train_mse.append(model.score(X_train, y_train))
        test_mse.append(model.score(X_test, y_test))

    ax[1, 0].plot(l2_penalties, np.array(train_mse), label="train mse")
    ax[1, 0].plot(l2_penalties, np.array(test_mse), label="test mse")
    ax[1, 0].set_xscale("log")
    ax[1, 0].legend(loc=0, fontsize="small", borderaxespad=.3)
    ax[1, 0].set_xlabel("$\lambda_{2}$")
    ax[1, 0].set_ylabel("MSE")
    ax[1, 0].set_title("Train vs. Test MSE")

    # Plot regularization paths
    for i in range(n_features):
        ax[1, 1].plot(l2_penalties, np.array([theta[i] for theta in theta_values]))

    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xlabel("$\lambda_{2}$")
    ax[1, 1].set_ylabel("$\\theta$")
    ax[1, 1].set_title("Regularization Paths")

    fig.tight_layout()
    plt.show()

