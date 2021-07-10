from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from base import BaseEstimator


class LassoRegressionEstimator(BaseEstimator):

    def __init__(
            self,
            fit_intercept: bool,
            l1_penalty: float
    ) -> None:
        super().__init__()
        self.fit_intercept = fit_intercept
        self.l1_penalty = l1_penalty

    def _fit_estimator(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fit_method: Optional[str] = "cvxpy",
            epochs: Optional[int] = None,
            learning_rate: Optional[float] = None
    ) -> None:
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        n_samples, n_features = X.shape

        theta = cp.Variable(shape=(n_features,))
        mse = cp.sum_squares(y - X@theta) / n_samples
        l1_reg = self.l1_penalty * cp.pnorm(theta[1:] if self.fit_intercept else theta, p=1)
        loss = mse + l1_reg
        _ = cp.Problem(cp.Minimize(loss)).solve()
        self.theta = theta.value

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = np.concatenate([np.ones(shape=(len(X), 1)), X], axis=1)
        return np.dot(X, self.theta)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        n_samples, _ = X.shape
        return np.sum((y - np.dot(X, self.theta)) ** 2) / n_samples

    def _validate_predict_data(self, X: np.ndarray) -> None:
        super()._validate_predict_data(X=X)
        assert self.theta.shape[0] == (X.shape[1] + 1) if self.fit_intercept else X.shape[1], \
            "training and test sets have unequal number of features."


def compute_lasso_loss(
        X: np.ndarray,
        y: np.ndarray,
        theta: np.array,
        l1_penalty: float,
        includes_intercept: bool
) -> np.ndarray:
    n_samples, n_features = X.shape
    errors = [y[i] - sum([np.dot(X[i, j], theta[j]) for j in range(n_features)]) for i in range(n_samples)]
    mse = sum([error ** 2 for error in errors]) / n_samples
    l1_reg = (
            l1_penalty * np.sum(np.abs(theta[1:])) if includes_intercept else l1_penalty * np.sum(np.abs(theta))
    )
    return mse+l1_reg


if __name__ == "__main__":

    np.random.seed(0)

    # Generate some random data.
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

    for i, l1_penalty in enumerate(np.linspace(0., 10., 20)):
        model = LassoRegressionEstimator(fit_intercept=True, l1_penalty=l1_penalty)
        model.fit(X, y)
        ax[0, 0].plot(
            np.linspace(X.min(), X.max(), n_samples),
            model.predict(X=np.linspace(X.min(), X.max(), n_samples*n_features).reshape(n_samples, n_features)),
            label=f"$\lambda_{1}$={np.round(l1_penalty)}" if i % 2 == 0 else None,
            lw=2.5,
            ls="--"
        )

    ax[0, 0].legend(ncol=2, fontsize="small", borderaxespad=.3, loc=2)
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("y")
    ax[0, 0].set_title("Lasso Regression for different $\lambda_{1}$")

    # Compute loss functions for different lambda
    slopes = np.linspace(-2.25, 7.75, 100)
    for i, l1_penalty in enumerate(np.linspace(0., 10., 20)):
        ax[0, 1].plot(
            slopes,
            np.array([
                compute_lasso_loss(X=np.concatenate([np.ones_like(X), X], axis=1), y=y, theta=np.array([3.5, slope]),
                                   l1_penalty=l1_penalty, includes_intercept=True) for slope in slopes
            ]),
            label=f"$\lambda_{1}$={np.round(l1_penalty)}" if i % 2 == 0 else None,
            lw=2.5
        )

    ax[0, 1].legend(fontsize="small", borderaxespad=.3, loc=0, ncol=2)
    ax[0, 1].set_ylabel("loss")
    ax[0, 1].set_xlabel("$\\theta_{1}$")
    ax[0, 1].set_title(f"L1 Reg Loss for different $\lambda_{1}$")

    # Generate some sparse data
    n_samples, n_features = 100, 20
    sigma = 2.25
    sparsity = 0.8
    test_size = 0.5
    n_test = int(n_samples * test_size)

    theta_star = np.random.randn(n_features)
    zero_idx = [np.random.choice(np.arange(n_features), replace=False) for _ in range(int(sparsity * n_features))]
    for idx in zero_idx:
        theta_star[idx] = 0
    X = np.random.randn(n_samples, n_features)
    y = np.dot(X, theta_star) + np.random.normal(0, sigma, size=n_samples)
    X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[-n_test:]

    l1_penalties = np.logspace(-2, 2, 50)
    theta_values = []
    train_mse = []
    test_mse = []
    for l1_penalty in l1_penalties:
        model = LassoRegressionEstimator(fit_intercept=False, l1_penalty=l1_penalty)
        model.fit(X=X_train, y=y_train, fit_method="cvxpy")
        theta_values.append(model.theta)
        train_mse.append(model.score(X=X_train, y=y_train))
        test_mse.append(model.score(X=X_test, y=y_test))

    ax[1, 0].plot(l1_penalties, np.array(train_mse), label="train mse")
    ax[1, 0].plot(l1_penalties, np.array(test_mse), label="test mse")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("$\lambda_{1}$")
    ax[1, 0].set_ylabel("MSE")
    ax[1, 0].set_title("Train vs. Test MSE")
    ax[1, 0].legend()

    for i in range(n_features):
        ax[1, 1].plot(l1_penalties, np.array([theta_value[i] for theta_value in theta_values]))

    ax[1, 1].set_ylabel("$\\theta$")
    ax[1, 1].set_xlabel("$\lambda_{1}$")
    ax[1, 1].set_title("Regularization Paths")
    ax[1, 1].set_xscale("log")

    fig.tight_layout()
    plt.show()
