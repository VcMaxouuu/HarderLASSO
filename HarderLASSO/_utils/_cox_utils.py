"""Cox regression utilities for survival analysis.

This module provides utility functions for Cox proportional hazards regression,
including concordance index computation, survival analysis, and partial
log-likelihood calculation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from numpy import ndarray


def survival_analysis_training_info(
    times: ndarray,
    events: ndarray,
    predictions: ndarray
) -> pd.DataFrame:
    """
    Compute survival analysis information from training data.

    This function computes the cumulative baseline hazard function,
    baseline survival function, and individual survival functions
    for each sample in the training data.

    Parameters
    ----------
    times : ndarray
        Survival times of shape (n_samples,).
    events : ndarray
        Event indicators of shape (n_samples,) where 1 indicates
        event and 0 indicates censoring.
    predictions : ndarray
        Risk scores from the model of shape (n_samples,).

    Returns
    -------
    pd.DataFrame
        DataFrame containing durations, baseline hazard and survival functions,
        and individual survival probabilities.

    Notes
    -----
    Uses Efron approximation for handling tied event times.
    """
    times = np.asarray(times)
    events = np.asarray(events)
    predictions = np.asarray(predictions)

    exp_predictions = np.exp(predictions)

    # Group by durations to handle ties
    df = pd.DataFrame({
        "durations": times,
        "exp_pred": exp_predictions,
        "uncensored_exp_pred": exp_predictions * events,
        "events": events
    })

    df = df.groupby("durations", as_index=False).agg({
        "exp_pred": "sum",
        "uncensored_exp_pred": "sum",
        "events": "sum"
    })

    df = df.sort_values("durations", ascending=False).reset_index(drop=True)

    baseline_cumulative_hazard = np.zeros(len(df))
    for i in range(len(df)):
        d_i = int(df.loc[i, "events"])  # Number of events at this time
        if d_i == 0:
            baseline_cumulative_hazard[i] = 0
            continue

        # Efron approximation for tied events
        risk_set = df.loc[:i, "exp_pred"].sum()  # Sum of exp(predictions) in the risk set
        tied_exp_pred = df.loc[i, "uncensored_exp_pred"]  # Sum of exp(predictions) for tied events

        # Compute the hazard increment
        hazard_increment = 0
        for k in range(d_i):
            hazard_increment += 1 / (risk_set - (k / d_i) * tied_exp_pred)

        baseline_cumulative_hazard[i] = hazard_increment

    df["baseline_cumulative_hazard"] = np.flip(np.flip(baseline_cumulative_hazard).cumsum())
    df["baseline_survival"] = np.exp(-df["baseline_cumulative_hazard"])

    # Compute individual survival probabilities
    baseline_values = df["baseline_survival"].values
    survival_probabilities = baseline_values[:, np.newaxis] ** exp_predictions

    survival_probabilities_df = pd.DataFrame(
        survival_probabilities,
        columns=[f"individual_{i+1}" for i in range(survival_probabilities.shape[1])]
    )

    result_df = pd.concat([df, survival_probabilities_df], axis=1).reset_index(drop=True)
    return result_df[::-1].reset_index(drop=True)


def concordance_index(
    times: ndarray,
    events: ndarray,
    predictions: ndarray
) -> float:
    """
    Compute Harrell's concordance index (C-index) for survival data.

    The concordance index measures the fraction of all pairs of subjects
    whose predicted survival times are correctly ordered among all subjects
    that can actually be ordered.

    Parameters
    ----------
    times : ndarray
        Survival times of shape (n_samples,).
    events : ndarray
        Event indicators of shape (n_samples,) where 1 indicates
        event and 0 indicates censoring.
    predictions : ndarray
        Risk scores from the model of shape (n_samples,).
        Higher values indicate higher risk (shorter survival).

    Returns
    -------
    float
        Concordance index value between 0 and 1, where 1.0 indicates
        perfect concordance, 0.5 indicates random predictions, and
        0.0 indicates perfect anti-concordance.

    Notes
    -----
    Only pairs where the comparison is meaningful are considered.
    Ties in predictions are handled by assigning a score of 0.5.
    """
    # Find indices of events (not censored)
    event_idx = np.where(events == 1)[0]

    # Get times and predictions for events
    times_i = times[event_idx][:, None]  # Shape: (n_events, 1)
    times_j = times[None, :]  # Shape: (1, n_samples)

    pred_i = predictions[event_idx][:, None]  # Shape: (n_events, 1)
    pred_j = predictions[None, :]  # Shape: (1, n_samples)

    events_j = events[None, :]  # Shape: (1, n_samples)

    # Determine comparable pairs
    # Case 1: Event i occurs before observation j
    check1 = (times_i < times_j)
    # Case 2: Same observation time but j is censored
    check2 = ((times_i == times_j) & (events_j == 0))
    comparable = check1 | check2

    # Remove self-comparisons
    comparable[np.arange(len(event_idx)), event_idx] = False

    n_comparable = comparable.sum()
    if n_comparable == 0:
        return 0.0

    # Compute concordance
    concordant = np.where(pred_i > pred_j, 1.0, 0.0)
    ties = np.where(pred_i == pred_j, 0.5, 0.0)
    score = (concordant + ties)[comparable].sum()

    return score / n_comparable


def cox_partial_log_likelihood(
    times: ndarray,
    events: ndarray,
    predictions: ndarray
) -> float:
    """
    Compute negative partial log-likelihood for Cox regression.

    The partial log-likelihood is the standard optimization objective
    for Cox proportional hazards regression. This function computes
    the negative partial log-likelihood using the Efron approximation
    for handling tied event times.

    Parameters
    ----------
    times : ndarray
        Survival times of shape (n_samples,).
    events : ndarray
        Event indicators of shape (n_samples,) where 1 indicates
        event and 0 indicates censoring.
    predictions : ndarray
        Linear predictions from the model of shape (n_samples,).
        These are the log-hazard ratios.

    Returns
    -------
    float
        Negative partial log-likelihood value (higher is worse).

    Notes
    -----
    Uses Efron approximation for handling tied event times, which
    is more accurate than the Breslow approximation when there are
    many ties.
    """
    # Sort by descending survival times
    order = np.argsort(times)[::-1]
    t_s = times[order]
    p_s = predictions[order]
    e_s = events[order].astype(float)

    exp_p = np.exp(p_s)
    exp_pe = exp_p * e_s

    # Group by unique times
    unique_t = np.unique(t_s)[::-1]
    G = len(unique_t)

    group_uncens = np.zeros(G)
    group_exp = np.zeros(G)
    group_expe = np.zeros(G)
    group_events = np.zeros(G, dtype=int)

    for i, ut in enumerate(unique_t):
        mask = (t_s == ut)
        group_uncens[i] = np.sum(p_s[mask] * e_s[mask])
        group_exp[i] = np.sum(exp_p[mask])
        group_expe[i] = np.sum(exp_pe[mask])
        group_events[i] = int(e_s[mask].sum())

    cum_exp = np.cumsum(group_exp)

    # Compute log term using Efron approximation
    log_term = np.zeros(G)
    for i in range(G):
        d_i = group_events[i]
        if d_i == 0:
            continue
        R_i = cum_exp[i]  # Risk set size
        T_i = group_expe[i]  # Tied events contribution
        js = np.arange(d_i)
        log_term[i] = np.sum(np.log(R_i - (js / d_i) * T_i))

    loss_contrib = group_uncens - log_term
    loss = -np.sum(loss_contrib)
    return loss


def compute_survival_function(
    times: ndarray,
    baseline_hazard: ndarray,
    risk_score: float
) -> Tuple[ndarray, ndarray]:
    """
    Compute individual survival function from baseline hazard and risk score.

    Parameters
    ----------
    times : ndarray
        Time points at which to evaluate survival function.
    baseline_hazard : ndarray
        Baseline cumulative hazard at each time point.
    risk_score : float
        Individual risk score (exp(linear_predictor)).

    Returns
    -------
    tuple of ndarray
        Tuple of (times, survival_probabilities).
    """
    survival_probs = np.exp(-baseline_hazard * risk_score)
    return times, survival_probs


def compute_median_survival_time(
    times: ndarray,
    survival_probs: ndarray
) -> Optional[float]:
    """
    Compute median survival time from survival function.

    Parameters
    ----------
    times : ndarray
        Time points.
    survival_probs : ndarray
        Survival probabilities at each time point.

    Returns
    -------
    float or None
        Median survival time, or None if not estimable.
    """
    if (survival_probs <= 0.5).any():
        idx = np.where(survival_probs <= 0.5)[0][0]
        return times[idx]
    return None
