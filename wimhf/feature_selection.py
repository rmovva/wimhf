"""Feature selection utilities for WIMHF."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from skglm.datafits import Logistic as SkglmLogistic
from skglm.datafits import Quadratic
from skglm.estimators import GeneralizedLinearEstimator
from skglm.penalties import WeightedL1
from skglm.solvers import AndersonCD

def select_neurons_controlled_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    controls: Optional[np.ndarray] = None,
    classification: bool = False,
    max_iter: int = 50,
    max_samples: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
    standardize: bool = True,
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using weighted L1 where control variables are unpenalised.
    """
    rng = np.random.default_rng(seed)

    if max_samples is not None and activations.shape[0] > max_samples:
        idx = rng.choice(activations.shape[0], max_samples, replace=False)
        activations = activations[idx]
        target = target[idx]
        if controls is not None:
            controls = controls[idx]
        if verbose:
            print(f"[controlled-lasso] subsampled to {max_samples} rows")

    if controls is not None:
        X = np.column_stack([controls, activations])
        n_controls = controls.shape[1]
    else:
        X = activations
        n_controls = 0

    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    weights = np.ones(X_scaled.shape[1])
    weights[:n_controls] = 0.0
    datafit = SkglmLogistic() if classification else Quadratic()

    alpha_low, alpha_high = 1e-6, 1e4
    best_coef = None
    best_alpha = None

    if verbose:
        print(f"{'iter':>4} {'alpha':>12} {'#feat':>8} {'time(s)':>8}")
        print("-" * 40)

    for iteration in range(30):
        t0 = time.time()
        alpha = np.sqrt(alpha_low * alpha_high)
        model = GeneralizedLinearEstimator(
            datafit=datafit,
            penalty=WeightedL1(weights=weights, alpha=alpha),
            solver=AndersonCD(tol=1e-3, max_iter=max_iter),
        )
        model.fit(X_scaled, target)
        coef = model.coef_.flatten()
        activation_coef = coef[n_controls:]
        n_nonzero = int(np.sum(activation_coef != 0))

        if verbose:
            print(f"{iteration:4d} {alpha:12.2e} {n_nonzero:8d} {time.time()-t0:8.2f}")

        if n_nonzero == n_select:
            best_coef = coef
            best_alpha = alpha
            break
        if n_nonzero < n_select:
            alpha_high = alpha
        else:
            alpha_low = alpha
            best_coef = coef
            best_alpha = alpha

        if abs(alpha_high - alpha_low) < 1e-10:
            break

    if best_coef is None:
        raise ValueError("Failed to find a suitable alpha for controlled LASSO.")

    act_coef = best_coef[n_controls:]
    selected = np.where(act_coef != 0)[0]
    coefs = act_coef[selected]

    order = np.argsort(np.abs(coefs))[::-1]
    selected = selected[order][:n_select]
    coefs = coefs[order][:n_select]

    if verbose:
        print(f"[controlled-lasso] alpha={best_alpha:.2e}, selected={len(selected)}")

    return selected.tolist(), coefs.tolist()


def select_neurons_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    alpha: Optional[float] = None,
    max_iter: int = 1000,
    verbose: bool = False,
) -> Tuple[List[int], List[float]]:
    """Standard L1 feature selection without control variables."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)

    if alpha is None:
        alpha_low, alpha_high = 1e-6, 1e4
        coef = None

        if verbose:
            print(f"{'iter':>4} {'alpha':>12} {'#feat':>8} {'time(s)':>8}")
            print("-" * 40)

        for iteration in range(20):
            t0 = time.time()
            alpha = np.sqrt(alpha_low * alpha_high)
            if classification:
                model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=1 / alpha,
                    max_iter=max_iter,
                )
            else:
                model = Lasso(alpha=alpha, max_iter=max_iter)
            model.fit(X_scaled, target)
            coef = model.coef_.flatten()
            n_nonzero = int(np.sum(coef != 0))

            if verbose:
                print(f"{iteration:4d} {alpha:12.2e} {n_nonzero:8d} {time.time()-t0:8.2f}")

            if n_nonzero == n_select:
                break
            if n_nonzero < n_select:
                alpha_high = alpha
            else:
                alpha_low = alpha
        if coef is None:
            raise ValueError("LASSO search failed.")
    else:
        if verbose:
            print(f"[lasso] using provided alpha={alpha:.2e}")
        if classification:
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1 / alpha,
                max_iter=max_iter,
            )
        else:
            model = Lasso(alpha=alpha, max_iter=max_iter)
        model.fit(X_scaled, target)
        coef = model.coef_.flatten()

    order = np.argsort(-np.abs(coef))[:n_select]
    return order.tolist(), coef[order].tolist()


def select_neurons_controlled_ols(
    activations: np.ndarray,
    target: np.ndarray,
    controls: np.ndarray,
    n_select: int,
    classification: bool = False,
    show_progress: bool = False,
    standardize: bool = True,
) -> Tuple[List[int], List[float]]:
    """Select neurons via univariate regression controlling for additional covariates."""
    assert len(activations) == len(target) == len(controls), "Mismatched input sizes."

    if standardize:
        scaler_act = StandardScaler()
        scaler_ctrl = StandardScaler()
        activations = scaler_act.fit_transform(activations)
        controls = scaler_ctrl.fit_transform(controls)

    coefs: List[float] = []
    iterator = range(activations.shape[1])
    if show_progress:
        iterator = tqdm(iterator, desc="Controlled OLS coefficients")

    for i in iterator:
        X = np.column_stack([controls, activations[:, i]])
        X = sm.add_constant(X)
        if classification:
            model = sm.Logit(target, X).fit(disp=0)
        else:
            model = sm.OLS(target, X).fit()
        coefs.append(model.params[-1])

    coefs = np.array(coefs)
    idx = np.argsort(-np.abs(coefs))[:n_select]
    return idx.tolist(), coefs[idx].tolist()
