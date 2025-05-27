import numpy as np
import pandas as pd

def survival_analysis_training_info(times, events, predictions):
    """
    Compute the following information using the given training data :
        - cumulative baseline hazard function
        - baseline survival function
        - survival function for each individual in the training data
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

    baseline_values = df["baseline_survival"].values
    survival_probabilities = baseline_values[:, np.newaxis] ** exp_predictions

    survival_probabilities_df = pd.DataFrame(
        survival_probabilities,
        columns=[f"individual_{i+1}" for i in range(survival_probabilities.shape[1])]
    )

    result_df = pd.concat([df, survival_probabilities_df], axis=1).reset_index(drop=True)
    return result_df[::-1].reset_index(drop=True)


def concordance_index(times, events, predictions):
    event_idx = np.where(events == 1)[0]

    times_i = times[event_idx][:, None]
    times_j = times[None, :]

    pred_i = predictions[event_idx][:, None]
    pred_j = predictions[None, :]

    events_j = events[None, :]

    check1 = (times_i < times_j)
    check2 = ((times_i == times_j) & (events_j == 0))
    comparable = check1 | check2

    comparable[np.arange(len(event_idx)), event_idx] = False

    n_comparable = comparable.sum()

    concordant = np.where(pred_i > pred_j, 1.0, 0.0)
    ties = np.where(pred_i == pred_j, 0.5, 0.0)
    score = (concordant + ties)[comparable].sum()
    return score / n_comparable if n_comparable > 0 else 0.0

def cox_partial_log_likelihood(times, events, predictions):
    """Negative partial log-likelihood with ties events handled by Efron approximation."""

    order = np.argsort(times)[::-1]
    t_s = times[order]
    p_s = predictions[order]
    e_s = events[order].astype(float)

    exp_p  = np.exp(p_s)
    exp_pe = exp_p * e_s

    unique_t = np.unique(t_s)[::-1]
    G = len(unique_t)

    group_uncens = np.zeros(G)
    group_exp    = np.zeros(G)
    group_expe   = np.zeros(G)
    group_events = np.zeros(G, dtype=int)

    for i, ut in enumerate(unique_t):
        mask = (t_s == ut)
        group_uncens[i] = np.sum(p_s[mask] * e_s[mask])
        group_exp[i]    = np.sum(exp_p[mask])
        group_expe[i]   = np.sum(exp_pe[mask])
        group_events[i] = int(e_s[mask].sum())

    cum_exp = np.cumsum(group_exp)

    # calcul du terme log (Efron)
    log_term = np.zeros(G)
    for i in range(G):
        d_i = group_events[i]
        if d_i == 0:
            continue
        R_i = cum_exp[i]
        T_i = group_expe[i]
        js  = np.arange(d_i)
        log_term[i] = np.sum(np.log(R_i - (js / d_i) * T_i))

    loss_contrib = group_uncens - log_term
    loss = -np.sum(loss_contrib)
    return loss
