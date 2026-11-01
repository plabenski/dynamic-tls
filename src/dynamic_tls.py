"""Dynamic thermal load sensitivity model of relative PSII efficiency (Fv/Fm).

Units: temperature and intercept/acclimation shifts in °C; dynamic time in minutes;
reference acclimation tau values in days.
Thermal load and relative Fv/Fm are dimensionless. Leaf trajectories are propagated
independently before tree-level averaging. 

"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit, least_squares
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# ------------------------------------------- #
# 1. Thermal load and empirical dose-response #
# ------------------------------------------- #

def thermal_load_rate(T, a, b):
    """Return 1 / time_to_T50(T), the normalised thermal-load rate (min⁻¹). T is °C; a/b use minute-based TDT fits."""
    return 1.0 / np.exp((np.asarray(T) - b) / a)


def fvfm_from_dose(D, lam, k):
    """Map non-negative thermal load to relative Fv/Fm using exp(-(D/lam)**k)."""
    D = np.maximum(np.asarray(D), 0.0)
    return np.exp(-((D / lam) ** k))


def dose_from_fvfm(F, lam, k):
    """Return the inverse-Weibull dose coordinate; clip F to [1e-8, 1] for numerical stability."""
    F = np.clip(np.asarray(F), 1e-8, 1.0)
    return lam * (-np.log(F)) ** (1.0 / k)


def fit_species_mean_dose_response(df_response, species):
    """
    Fit F(D)=exp(-(D/lam)^k) to species-mean relative Fv/Fm.

    Required columns: species, dose, response_rel. 
    """
    x = (
        df_response.loc[
            df_response["species"] == species
        ]
        .copy()
        .sort_values("dose")
    )

    popt, pcov = curve_fit(
        lambda D, lam, k: fvfm_from_dose(D, lam, k),
        x["dose"].values,
        x["response_rel"].values,
        p0=[1.0, 1.0],
        bounds=(0, np.inf),
    )

    y_pred = fvfm_from_dose(
        x["dose"].values,
        popt[0],
        popt[1]
    )

    r2 = r2_score(
        x["response_rel"].values,
        y_pred
    )

    return {"lam": popt[0], "k": popt[1], "cov": pcov, "data": x , "r2": r2}


# ------------------------ #
# 2. Reference acclimation #
# ------------------------ #

def acclimation_target(T, A_max, T_acclim, beta):
    """Equilibrium upward shift in b [deg C] at temperature T."""
    heat_signal = max(float(T) - T_acclim, 0.0)
    return A_max * (1.0 - np.exp(-beta * heat_signal))


def simulate_acclimation_reference(
    temp,
    A_max,
    T_acclim,
    tau_up_days,
    tau_down_days,
    beta,
    A0=0.0,
):
    """Simulate acclimation for an irregular datetime-indexed temperature Series."""
    temp = temp.sort_index()
    A = A0
    out = []

    for i, T in enumerate(temp.values):
        if i == 0 or np.isnan(T):
            out.append(A)
            continue

        dt_days = (temp.index[i] - temp.index[i - 1]).total_seconds() / 86400.0
        A_eq = acclimation_target(T, A_max, T_acclim, beta)
        tau = tau_up_days if A_eq > A else tau_down_days
        A = A_eq + (A - A_eq) * np.exp(-dt_days / tau)
        out.append(A)

    return np.asarray(out)


def get_observed_acclimation(df_diff, treatment="Heat"):
    """Observed A states at HW1, PostHW1, HW2, PostHW2 from the acclimation experiment."""
    x = df_diff.set_index("Treatment")

    A_HW1 = x.loc[treatment, "HW1-Pre"]
    A_post1 = A_HW1 + x.loc[treatment, "PostHW1-HW1"]
    A_HW2 = A_post1 + x.loc[treatment, "HW2-PostHW1"]
    A_post2 = A_HW2 + x.loc[treatment, "PostHW2-HW2"]

    return np.array([A_HW1, A_post1, A_HW2, A_post2])


def fit_reference_acclimation(
    df_temp_acc,
    df_diff,
    obs_dates,
    treatment="Heat",
    temp_col="Tleaf",
    beta=0.3,
):
    """
    Fit A_max, T_acclim, tau_up and tau_down to the dedicated acclimation experiment.

    beta is fixed to reduce parameter non-identifiability.
    tau values are returned in days; the fitted A Series retains the temperature timestamps.
    """
    temp = df_temp_acc[temp_col].dropna().sort_index()
    observed = get_observed_acclimation(df_diff, treatment)

    def residuals(p):
        A_max, T_acclim, tau_up, tau_down = p
        A = simulate_acclimation_reference(
            temp, A_max, T_acclim, tau_up, tau_down, beta
        )
        predicted = np.array([
            A[temp.index.get_indexer([d], method="nearest")[0]]
            for d in obs_dates
        ])
        return predicted - observed

    fit = least_squares(
        residuals,
        x0=[2.0, 35.0, 2.0, 7.0],
        bounds=([0.05, 35.0, 0.05, 0.1],
                [4.0, 40.0, 10.0, 20.0]),
    )

    fitted_acclimation = simulate_acclimation_reference(temp, fit.x[0], fit.x[1],
                                                        fit.x[2], fit.x[3], beta)

    return {
        "A_max_ref": fit.x[0],
        "T_acclim_ref": fit.x[1],
        "tau_up_days": fit.x[2],
        "tau_down_days": fit.x[3],
        "beta": beta,
        "observed": observed,
        "fit": fit,
    }, pd.Series(fitted_acclimation, index=temp.index, name="A")


# -------------------------------------------- #
# 3. Recovery intervals, capacity, and fitting #
# -------------------------------------------- #

def repair_rate_schoolfield(T, r_ref, *, T_ref=25.0, Th=45.0, Ea=0.8e5, Eh=3.8e5, R=8.314):
    """Return the Schoolfield repair rate (min⁻¹); temperatures are in °C.

    Ea/Eh are activation/deactivation energies (J mol⁻¹); R is in J mol⁻¹ K⁻¹.
    Th controls high-temperature deactivation. T_ref is the Arrhenius reference.
    r_ref is a scaling parameter, so deactivation makes the actual rate at T_ref slightly smaller than r_ref.
    """
    Tk = np.asarray(T) + 273.15
    Trefk, Thk = T_ref + 273.15, Th + 273.15
    numerator = r_ref * np.exp((-Ea / R) * (1.0 / Tk - 1.0 / Trefk))
    denominator = 1.0 + np.exp((Eh / R) * (1.0 / Thk - 1.0 / Tk))
    return numerator / denominator


def create_recovery_dataframe(df_fvfm):
    """
    Create recovery-only intervals:
    evening -> next morning, and last evening -> later recovery middays.
    """
    fv = df_fvfm.reset_index().sort_values(["Tree_ID", "Datetime"])
    pairs = []

    for tree_id, x in fv.groupby("Tree_ID"):
        for _, eve in x[x["Campaign"] == "evening"].iterrows():
            mornings = x[
                (x["Campaign"] == "morning") &
                (x["Datetime"] > eve["Datetime"])
            ]
            if len(mornings):
                morn = mornings.iloc[0]
                pairs.append([
                    tree_id, eve["Datetime"], morn["Datetime"],
                    eve["Fvfm_rel_mean"], morn["Fvfm_rel_mean"], "Overnight"
                ])

        evenings = x[x["Campaign"] == "evening"]
        if len(evenings):
            last_eve = evenings.iloc[-1]
            recovery = x[
                (x["Campaign"] == "midday") &
                (x["Datetime"] > last_eve["Datetime"])
            ]
            for _, rec in recovery.iterrows():
                pairs.append([
                    tree_id, last_eve["Datetime"], rec["Datetime"],
                    last_eve["Fvfm_rel_mean"], rec["Fvfm_rel_mean"], "Recovery"
                ])

    out = pd.DataFrame(
        pairs,
        columns=["Tree_ID", "t0", "t1", "fvfm_start", "Fvfm_obs", "period"],
    )

    # Filtering is performed by fit_recovery so excluded intervals remain available for diagnostics.
    return out


def _temperature_intervals(row, df_temp, temp_col="Temperature"):
    """Return one temperature array per leaf (or one per tree) for (t0, t1].

    Inputs use a datetime/Tree_ID MultiIndex, optionally also Leaf_ID.
    Observed Fv/Fm at t0 is the initial state. 
    """
    if row["Tree_ID"] not in df_temp.index.get_level_values("Tree_ID"):
        return []
    tree = df_temp.xs(row["Tree_ID"], level="Tree_ID").sort_index()
    series = ([tree.xs(leaf, level="Leaf_ID")[temp_col]
               for leaf in tree.index.get_level_values("Leaf_ID").unique()]
              if "Leaf_ID" in tree.index.names else [tree[temp_col]])
    trajectories = []
    for temperatures in series:
        selected = temperatures.loc[(temperatures.index > row["t0"]) &
                                    (temperatures.index <= row["t1"])].dropna()
        if not selected.empty:
            trajectories.append(selected.to_numpy())
    return trajectories


def predict_recovery_interval(
    row, df_temp, r_ref, Th, model, lam, k, dt_min, temp_col="Temperature",
    I_scale=None, p_repair=6.0, *, T_ref=25.0, Ea=0.8e5, Eh=3.8e5,
):
    """Predict final tree-mean Fv/Fm for one observed recovery interval.

    Each measured leaf starts at the same observed tree-mean fvfm_start and is
    propagated independently; final leaf predictions are averaged. 
    With the capacity modifier, recovery_step() recalculates capacity from impairment at each timestep.
    """
    if model not in {"fvfm", "dose"}:
        raise ValueError("model must be 'fvfm' or 'dose'")
    trajectories = _temperature_intervals(row, df_temp, temp_col)
    if not trajectories:
        return np.nan
    params = {
        "use_recovery": True, "r_ref": r_ref, "Th": Th, "T_ref": T_ref, "Ea": Ea, "Eh": Eh,
        "recovery_model": model, "lam": lam, "k": k,
        "use_repair_capacity": I_scale is not None, "I_scale": I_scale, "p_repair": p_repair,
    }
    predictions = []
    for temperatures in trajectories:
        F = float(row["fvfm_start"])
        if I_scale is None:
            integrated_rate = np.sum(repair_rate_schoolfield(
                temperatures, r_ref=r_ref, Th=Th, T_ref=T_ref, Ea=Ea, Eh=Eh,
            ) * dt_min)
            if model == "fvfm":
                F = 1.0 - (1.0 - F) * np.exp(-integrated_rate)
            else:
                dose = dose_from_fvfm(F, lam, k) * np.exp(-integrated_rate)
                F = float(fvfm_from_dose(dose, lam, k))
        else:
            for temperature in temperatures:
                F, _, _, _ = recovery_step(F, temperature, params, dt_min)
        predictions.append(F)
    return float(np.mean(predictions))


def fit_recovery(
    df_fvfm,
    df_temp,
    lam,
    k,
    Th,
    model="fvfm",
    dt_min=5,
    min_fvfm=None,
    fit_repair_capacity=False,
    temp_col="Temperature",
    p_repair=6.0,
    *, T_ref=25.0, Ea=0.8e5, Eh=3.8e5,
):
    """Fit the repair-rate scale and optionally impairment-dependent repair capacity.

    Uses evening-to-morning and post-heatwave recovery observations. min_fvfm
    filters both interval endpoints for fitting; all intervals remain in returned
    diagnostics.
    """
    if model not in {"fvfm", "dose"}:
        raise ValueError("model must be 'fvfm' or 'dose'")
    if dt_min <= 0 or p_repair <= 0:
        raise ValueError("dt_min and p_repair must be positive.")
    rec_all = create_recovery_dataframe(df_fvfm)

    if min_fvfm is not None:
        fit_mask = (
            (rec_all["fvfm_start"] >= min_fvfm) &
            (rec_all["Fvfm_obs"] >= min_fvfm)
        )

        rec = rec_all.loc[fit_mask].copy()

    else:
        fit_mask = pd.Series(True, index=rec_all.index)
        rec = rec_all.copy()

    finite_endpoints = np.isfinite(rec_all[["fvfm_start", "Fvfm_obs"]]).all(axis=1)
    fit_mask = fit_mask & finite_endpoints
    rec = rec_all.loc[fit_mask].copy()
    rec_all["used_for_fit"] = fit_mask
    if rec.empty:
        raise ValueError("No recovery intervals remain after filtering.")
    available = rec.apply(lambda row: bool(_temperature_intervals(row, df_temp, temp_col)), axis=1)
    if not available.any():
        raise ValueError("No temperature samples are available for the recovery fit.")
    rec_all["used_for_fit"] = fit_mask & available.reindex(rec_all.index, fill_value=False)
    rec = rec.loc[available].copy()

    def predictions(data, r_ref, I_scale):
        return data.apply(
            lambda row: predict_recovery_interval(
                row=row, df_temp=df_temp, r_ref=r_ref, Th=Th, model=model, lam=lam, k=k,
                dt_min=dt_min, temp_col=temp_col, I_scale=I_scale, p_repair=p_repair,
                T_ref=T_ref, Ea=Ea, Eh=Eh,
            ),
            axis=1,
        )

    # Fit recovery parameters
    if fit_repair_capacity:

        def residuals(x):
            r_ref, I_scale = x

            pred = predictions(rec, r_ref=r_ref, I_scale=I_scale)
            valid = pred.notna()

            return (pred.loc[valid].values - rec.loc[valid, "Fvfm_obs"].values)

        fit = least_squares(
            residuals,
            x0=[0.001, 1.0],
            bounds=([1e-7, 0.01],[0.02, 20.0])
        )

        r_ref, I_scale = fit.x

    else:

        def residuals(x):
            pred = predictions(rec, r_ref=x[0], I_scale=None)
            valid = pred.notna()

            return (pred.loc[valid].values - rec.loc[valid, "Fvfm_obs"].values)

        fit = least_squares(
            residuals,
            x0=[0.001],
            bounds=([1e-7], [0.02]),
        )

        r_ref = fit.x[0]
        I_scale = None

    # Predict all recovery periods
    rec_all["Fvfm_pred"] = predictions(rec_all, r_ref=r_ref, I_scale=I_scale)
    rec_all["residual"] = (rec_all["Fvfm_obs"] - rec_all["Fvfm_pred"])
    rec_all["impairment_start"] = (1.0 - rec_all["fvfm_start"])

    # RMSE
    valid_all = rec_all[["Fvfm_obs", "Fvfm_pred"]].dropna()
    valid_fit = rec_all.loc[rec_all["used_for_fit"], ["Fvfm_obs", "Fvfm_pred"]].dropna()

    rmse_all = root_mean_squared_error(
        valid_all["Fvfm_obs"],
        valid_all["Fvfm_pred"],
    )

    rmse_fit = root_mean_squared_error(
        valid_fit["Fvfm_obs"],
        valid_fit["Fvfm_pred"],
    )

    return {
        "model": model,
        "Th": Th, "T_ref": T_ref, "Ea": Ea, "Eh": Eh,
        "dt_min": dt_min, "temp_col": temp_col,
        "r_ref": r_ref,
        "I_scale": I_scale,
        "p_repair": p_repair,
        "rmse": rmse_fit,
        "rmse_fit": rmse_fit,
        "rmse_all": rmse_all,
        "data": rec_all,
        "fit": fit,
        "min_fvfm": min_fvfm,
        "fit_repair_capacity": fit_repair_capacity,
    }


def repair_capacity_factor(impairment, I_scale, p=6.0):
    """Return 1 / (1 + (impairment / I_scale)**p), a dimensionless repair capacity.

    Impairment is 1 - relative Fv/Fm. Capacity is 0.5 at I_scale.
    """
    if I_scale is None or I_scale <= 0 or p <= 0:
        raise ValueError("I_scale and the capacity exponent must be positive.")
    return 1.0 / (1.0 + (np.maximum(np.asarray(impairment), 0.0) / I_scale) ** p)


# --------------------------------------- #
# 4. Dynamic state updates and simulation #
# --------------------------------------- #

def update_acclimation(A, T, params, dt_min):
    """Advance the intercept shift A (°C) towards its target; tau_up/tau_down and dt_min are in minutes."""
    if not params.get("use_acclimation", False):
        return 0.0

    A_eq = acclimation_target(
        T, params["A_max"], params["T_acclim"], params["beta"]
    )
    tau = params["tau_up"] if A_eq > A else params["tau_down"]
    return A_eq + (A - A_eq) * np.exp(-dt_min / tau)


def recovery_step(F, T, params, dt_min):
    """Advance recovery only; return F_new, base rate, capacity, and effective rate.

    The fvfm formulation acts on impairment; the dose formulation acts on the
    inverse-Weibull thermal dose.
    """
    if not params.get("use_recovery", False):
        return F, 0.0, 1.0, 0.0

    r_base = float(repair_rate_schoolfield(
        T, params["r_ref"],
        T_ref=params.get("T_ref", 25.0),
        Th=params.get("Th", 45.0),
        Ea=params.get("Ea", 0.8e5),
        Eh=params.get("Eh", 3.8e5),
    ))

    if params.get("use_repair_capacity", False):
        capacity = float(repair_capacity_factor(
            1.0 - F, I_scale=params.get("I_scale"), p=params.get("p_repair", 6.0),
        ))
    else:
        capacity = 1.0

    r_eff = r_base * capacity

    if params.get("recovery_model", "fvfm") == "fvfm":
        I = 1.0 - F
        I_new = I * np.exp(-r_eff * dt_min)
        F_new = 1.0 - I_new
        return float(np.clip(F_new, 0.0, 1.0)), r_base, capacity, r_eff

    if params.get("recovery_model", "fvfm") != "dose":
        raise ValueError("recovery_model must be 'fvfm' or 'dose'")
    D = dose_from_fvfm(F, params["lam"], params["k"])
    D *= np.exp(-r_eff * dt_min)
    F_new = float(fvfm_from_dose(D, params["lam"], params["k"]))
    return F_new, r_base, capacity, r_eff


def step_state(T, state, params, dt_min):
    """Advance damage, acclimation and recovery by one timestep."""
    F = state["F"]
    A = state["A"]
    D_cum = state["D_cum"]

    if np.isnan(T):
        return {
            **state,
            "b_dyn": params["b0"] + A,
            "D_equiv": float(dose_from_fvfm(F, params["lam"], params["k"])),
            "dD": 0.0,
            "r_repair_base": 0.0,
            "repair_capacity": 1.0,
            "r_repair": 0.0,
            "F_after_heat": F,
        }

    A_new = update_acclimation(A, T, params, dt_min)
    b_dyn = params["b0"] + A_new

    D_equiv = float(dose_from_fvfm(F, params["lam"], params["k"]))
    dD = float(thermal_load_rate(T, params["a"], b_dyn) * dt_min)
    D_after_heat = D_equiv + dD
    F_after_heat = float(fvfm_from_dose(D_after_heat, params["lam"], params["k"]))
    new_impairment = max(F - F_after_heat, 0.0)

    # All variants use the same recovery update after the heat increment.
    F_new, r_base, capacity, r_eff = recovery_step(F_after_heat, T, params, dt_min)

    return {
        "F": float(np.clip(F_new, 0.0, 1.0)),
        "A": A_new,
        "D_cum": D_cum + dD,
        "b_dyn": b_dyn,
        "D_equiv": D_equiv,
        "dD": dD,
        "r_repair_base": r_base,
        "repair_capacity": capacity,
        "r_repair": r_eff,
        "F_after_heat": F_after_heat,
    }


def _simulate_temperature_series(
    sub,
    params,
    dt_min=5,
    temp_col="Temperature",
    F0=1.0,
):
    """Simulate one temperature trajectory."""
    state = {
        "F": F0,
        "A": 0.0,
        "D_cum": 0.0,
    }

    rows = []

    for T in sub[temp_col].values:
        state = step_state(T, state, params, dt_min)
        rows.append(state.copy())

    states = pd.DataFrame(rows, index=sub.index)
    out = pd.concat([sub, states], axis=1)
    out["Fvfm_rel_pred"] = out["F"]

    return out


def simulate_leaf(
    df_temp_leaf,
    tree_id,
    leaf_id,
    params,
    dt_min=5,
    temp_col="Temperature",
    F0=1.0,
):
    """
    Simulate one measured leaf.

    df_temp_leaf must have MultiIndex:
        [datetime, Tree_ID, Leaf_ID]
    """
    sub = (
        df_temp_leaf
        .xs((tree_id, leaf_id), level=("Tree_ID", "Leaf_ID"))
        .sort_index()
        .copy()
    )

    out = _simulate_temperature_series(
        sub=sub,
        params=params,
        dt_min=dt_min,
        temp_col=temp_col,
        F0=F0,
    )

    out["Tree_ID"] = tree_id
    out["Leaf_ID"] = leaf_id

    return out


def simulate_tree_leaves(
    df_temp_leaf,
    tree_id,
    params,
    dt_min=5,
    temp_col="Temperature",
    F0=1.0,
):
    """
    Simulate all measured leaves of one tree independently.

    The nonlinear thermal model is applied to each measured leaf temperature
    trajectory first. Predictions are then aggregated to tree level.

    Tree-level output contains:
        fvfm_rel_pred      mean predicted relative Fv/Fm
        Fvfm_pred_min      minimum among measured leaves
        Fvfm_pred_max      maximum among measured leaves
        Fvfm_pred_sd       SD among measured leaves

    Other dynamic model states are averaged across leaves for diagnostics.

    Notes
    -----
    Acclimation is currently propagated independently along each measured leaf
    temperature trajectory. This is the minimal extension of the existing
    model. A shared tree-level acclimation state driven by mean
    leaf temperature can be implemented separately later.
    """
    tree = df_temp_leaf.xs(tree_id, level="Tree_ID")
    leaf_ids = tree.index.get_level_values("Leaf_ID").unique()

    leaf_sims = [
        simulate_leaf(
            df_temp_leaf=df_temp_leaf,
            tree_id=tree_id,
            leaf_id=leaf_id,
            params=params,
            dt_min=dt_min,
            temp_col=temp_col,
            F0=F0,
        )
        for leaf_id in leaf_ids
    ]

    df_leaf = pd.concat(leaf_sims).reset_index()

    # Mean dynamic states across leaves.
    exclude = {
        "Tree_ID", "Leaf_ID", temp_col, "Fvfm_rel_pred", "F"
    }
    state_cols = [
        c for c in df_leaf.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    agg_dict = {
        "Fvfm_rel_pred": ["mean", "min", "max", "std"],
        temp_col: "mean",
    }

    for col in state_cols:
        agg_dict[col] = "mean"

    tree_summary = (
        df_leaf
        .groupby("Datetime")
        .agg(agg_dict)
    )

    # Flatten MultiIndex columns produced by aggregation.
    flat_cols = []
    for col in tree_summary.columns:
        if isinstance(col, tuple):
            base, stat = col
            if base == "Fvfm_rel_pred":
                mapping = {
                    "mean": "Fvfm_rel_pred",
                    "min": "Fvfm_pred_min",
                    "max": "Fvfm_pred_max",
                    "std": "Fvfm_pred_sd",
                }
                flat_cols.append(mapping[stat])
            else:
                flat_cols.append(base if stat == "mean" else f"{base}_{stat}")
        else:
            flat_cols.append(col)

    tree_summary.columns = flat_cols
    tree_summary["F"] = tree_summary["Fvfm_rel_pred"]
    tree_summary["Tree_ID"] = tree_id

    return tree_summary


def simulate_tree(
    df_temp,
    df_t50,
    tree_id,
    params,
    dt_min=5,
    temp_col="Temperature",
    F0=1.0,
):
    """
    Simulate one tree.

    Automatically detects whether df_temp contains tree-level or leaf-level
    temperatures. With leaf-level input, each leaf is simulated independently
    and predictions are aggregated only afterwards.
    """
    # Isolate each tree's baseline intercept from the shared variant dictionary.
    params = dict(params)
    if params.get("use_individual_T50", False):
        params["b0"] = df_t50.loc[(df_t50["Tree_ID"]==tree_id), "b0"].values[0]

    if "Leaf_ID" in df_temp.index.names:
        return simulate_tree_leaves(
            df_temp_leaf=df_temp,
            tree_id=tree_id,
            params=params,
            dt_min=dt_min,
            temp_col=temp_col,
            F0=F0,
                )

    sub = df_temp.xs(tree_id, level="Tree_ID").sort_index().copy()

    out = _simulate_temperature_series(
        sub=sub,
        params=params,
        dt_min=dt_min,
        temp_col=temp_col,
        F0=F0,
    )

    out["Tree_ID"] = tree_id
    return out


def simulate_variant(
    df_temp,
    df_t50,
    tree_ids,
    params,
    dt_min=5,
    temp_col="Temperature",
):
    """
    Simulate a model variant for all trees.

    Works with both tree-level and leaf-level temperature data because
    simulate_tree() performs automatic input detection.
    """
    return pd.concat([
        simulate_tree(df_temp, df_t50, tree, params, dt_min, temp_col)
        for tree in tree_ids
    ]).reset_index()


def make_variant_params(
    base_params,
    variant,
    acclimation_params=None,
    recovery_fit=None,
):
    """Build an independent variant dictionary, transferring fitted recovery/acclimation.

    Public names are retained for compatibility with the run script. Recovery
    thermal settings must agree with base_params; mismatches require refitting,
    rather than silently using a different temperature response in simulation.
    """
    supported = {"damage", "damage_repair", "damage_repair_acclimation"}
    if variant not in supported:
        raise ValueError(f"Unsupported model variant: {variant}")
    p = dict(base_params)
    p.update({
        "use_recovery": False,
        "use_acclimation": False,
        "use_repair_capacity": p.get("use_repair_capacity", False),
    })

    if variant in {"damage_repair", "damage_repair_acclimation"}:
        if recovery_fit is None:
            raise ValueError("recovery_fit is required for this variant")
        for key in ("Th", "T_ref", "Ea", "Eh"):
            if key in recovery_fit:
                if key in p and not np.isclose(p[key], recovery_fit[key]):
                    raise ValueError(f"{key} differs between recovery fitting and simulation; refit recovery.")
                p[key] = recovery_fit[key]
        p.update({
            "use_recovery": True, "r_ref": recovery_fit["r_ref"],
            "recovery_model": recovery_fit["model"],
            "I_scale": recovery_fit.get("I_scale"),
            "p_repair": recovery_fit.get("p_repair", p.get("p_repair", 6.0)),
            "use_repair_capacity": recovery_fit.get("fit_repair_capacity", p.get("use_repair_capacity", False)),
        })

    if variant == "damage_repair_acclimation":
        if acclimation_params is None:
            raise ValueError("acclimation_params is required for this variant")
        p.update({
            "use_acclimation": True,
            "A_max": acclimation_params["A_max"],
            "T_acclim": acclimation_params["T_acclim"],
            "beta": acclimation_params["beta"],
            "tau_up": acclimation_params["tau_up"],
            "tau_down": acclimation_params["tau_down"],
        })

    return p


# ------------------------------------------------- #
# 5. Experiment-specific acclimation and evaluation #
# ------------------------------------------------- #

def _merge_sim_obs(sim, obs):
    """Match one tree's observations to the nearest simulated timestamps (no maximum time-gap filter)."""
    return pd.merge_asof(
        obs.sort_index(),
        sim[["Fvfm_rel_pred"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
    )


def fit_experiment_acclimation(
    df_temp,
    df_fvfm,
    df_t50,
    tree_ids,
    base_params,
    acclimation_reference,
    recovery_fit,
    dt_min=5,
    temp_col="Temperature",
    regularize=False, sigma_A=3.0, sigma_T=6.0, A_max_upper=4.0,
):
    """Refit A_max and T_acclim; transfer beta and acclimation/deacclimation times.

    Reference tau values are converted from days to minutes once. The supplied
    severity-weighted objective (weight = 1 + 5 * impairment) is retained.
    regularize adds reference-centred residuals scaled by sigma_A/sigma_T;
    A_max_upper sets the fitted amplitude bound.
    """
    if A_max_upper <= 0 or (regularize and (sigma_A <= 0 or sigma_T <= 0)):
        raise ValueError("Amplitude bound and active regularisation scales must be positive.")
    A_ref = acclimation_reference["A_max_ref"]
    T_ref = acclimation_reference["T_acclim_ref"]

    transferred = {
        "beta": acclimation_reference["beta"],
        "tau_up": acclimation_reference["tau_up_days"] * 24 * 60,
        "tau_down": acclimation_reference["tau_down_days"] * 24 * 60,
    }

    def residuals(x):
        acc = {
            "A_max": x[0],
            "T_acclim": x[1],
            **transferred,
        }
        params = make_variant_params(
            base_params,
            "damage_repair_acclimation",
            acclimation_params=acc,
            recovery_fit=recovery_fit,
        )

        errs = []
        for tree in tree_ids:
            sim = simulate_tree(df_temp, df_t50, tree, params, dt_min, temp_col)
            obs = df_fvfm.xs(tree, level="Tree_ID")
            merged = _merge_sim_obs(sim, obs)
            err = (merged["Fvfm_rel_pred"] - merged["Fvfm_rel_mean"]).dropna()
            obs = merged.loc[err.index, "Fvfm_rel_mean"]

            # Give stronger weight to severely impaired observations
            alpha = 2.0
            weights = 1.0 + alpha * (1.0 - obs)

            errs.extend(
                np.sqrt(weights) * err.values
            )

        if not errs:
            raise ValueError("No matched observations are available for acclimation fitting.")
        if regularize:
            errs.extend([(x[0] - A_ref) / sigma_A, (x[1] - T_ref) / sigma_T])
        return np.asarray(errs)

    fit = least_squares(
        residuals,
        x0=[min(1.4, A_max_upper), np.clip(T_ref, 30.0, 40.0)],
        bounds=([0.0, 30.0], [A_max_upper, 40.0]),
    )

    return {
        "A_max": fit.x[0],
        "T_acclim": fit.x[1],
        **transferred,
        "fit": fit,
        "regularize": regularize, "sigma_A": sigma_A, "sigma_T": sigma_T, "A_max_upper": A_max_upper,
    }


def match_simulation_to_observations(df_sim, df_fvfm):
    """Match observations to nearest simulated times within each tree, without a gap limit."""
    sim = df_sim.copy()
    obs = df_fvfm.reset_index().copy()
    sim["Tree_ID"] = sim["Tree_ID"].astype(int)
    obs["Tree_ID"] = obs["Tree_ID"].astype(int)

    return pd.merge_asof(
        obs.sort_values(["Datetime", "Tree_ID"]),
        sim.sort_values(["Datetime", "Tree_ID"]),
        on="Datetime",
        by="Tree_ID",
        direction="nearest",
    )


def evaluate_predictions(df_combined):
    """Return RMSE, prediction R², predicted-minus-observed bias, and the valid pair count."""
    x = df_combined[["Fvfm_rel_mean", "Fvfm_rel_pred"]].dropna()
    return {
        "RMSE": root_mean_squared_error(x["Fvfm_rel_mean"], x["Fvfm_rel_pred"]),
        "R2": r2_score(x["Fvfm_rel_mean"], x["Fvfm_rel_pred"]),
        "Bias": np.mean(x["Fvfm_rel_pred"] - x["Fvfm_rel_mean"]),
        "n": len(x),
    }


# --------------------------------------- #
# 6. Model-component and prediction plots #
# --------------------------------------- #
def plot_intercept_acclimation(simulations, variant, subplot_no=""):
    """Plot changes in intercept b of the TDT relationship"""

    sim = simulations[variant]
    var = "b_dyn"
    var_name = "Intercept b [°C]"

    fig, ax = plt.subplots(figsize=(9, 3), dpi=300)

    tree_ids = sorted(sim["Tree_ID"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for tree_id in tree_ids:

        x = (
            sim[sim["Tree_ID"] == tree_id]
            .sort_values("Datetime")
        )

        color = colors[(int(tree_id) - 1) % 10]

        ax.plot(
            x["Datetime"],
            x[var],
            color=color,
            lw=1.0,
            label=f"Tree {tree_id}",
            zorder=3,
        )

    ax.set_ylabel(var_name)
    ax.set_xlabel("")
    ax.set_title("") #(name.replace("_", " + ").title())

    ax.axvspan(pd.Timestamp("2026-07-13 08:15:00"), pd.Timestamp("2026-07-18 00:00:00"), color='0.85', alpha=0.5, zorder=-10)
    ax.set_xlim(pd.Timestamp("2026-07-13 00:00:00"), pd.Timestamp("2026-07-22 00:00:00"))

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=1)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%d-%m") #m
    )

    ax.yaxis.set_major_locator(
        mticker.MultipleLocator(1)
    )

    ax.grid(alpha=0.2)

    ax.legend(
        title="",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    ax.legend().set_visible(False)
    # ax.text(0.05, 1.12, "Heatwave",
    #     transform=ax.transAxes,
    #     ha="left", va="top")

    # ax.text(0.57, 1.12, "Recovery",
    #     transform=ax.transAxes,
    #     ha="left", va="top")

    ax.text(
        -0.08, 1.12, subplot_no,
        transform=ax.transAxes,
        ha="left", va="top",
        fontweight="bold")

    #ax.set_ylim(52.5, 56.5)

    fig.tight_layout()
    plt.show()

def plot_simulated_var(simulations, var, var_name, subplot_no=""):
    """Plot a named simulated state for each variant; var_name supplies its axis label and units."""

    for name, sim in simulations.items():

        fig, ax = plt.subplots(figsize=(9, 3), dpi=300)

        tree_ids = sorted(sim["Tree_ID"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for tree_id in tree_ids:

            x = (
                sim[sim["Tree_ID"] == tree_id]
                .sort_values("Datetime")
            )

            color = colors[(int(tree_id) - 1) % 10]

            ax.plot(
                x["Datetime"],
                x[var],
                color=color,
                lw=1.0,
                label=f"Tree {tree_id}",
                zorder=3,
            )

        ax.set_ylabel(var_name)
        ax.set_xlabel("")
        ax.set_title("")

        ax.axvspan(pd.Timestamp("2026-07-13 08:15:00"), pd.Timestamp("2026-07-18 00:00:00"), color='0.85', alpha=0.5, zorder=-10)
        ax.set_xlim(pd.Timestamp("2026-07-13 00:00:00"), pd.Timestamp("2026-07-22 00:00:00"))

        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=1)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%m") #m
        )

        ax.yaxis.set_major_locator(
            mticker.MultipleLocator(1)
        )

        ax.grid(alpha=0.2)

        ax.legend(
            title="",
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
        )
        ax.legend().set_visible(False)

        ax.text(
            -0.08, 1.12, subplot_no,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold")

        fig.tight_layout()
        plt.show()


def plot_variant_fvfm(simulations, df_fvfm, subplot_no=""):
    """Plot each variant's tree means with ±1 SD across predicted/measured leaves."""
    obs = df_fvfm.reset_index()

    for name, sim in simulations.items():

        fig, ax = plt.subplots(figsize=(9, 4), dpi=300)

        tree_ids = sorted(sim["Tree_ID"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # First draw uncertainty bands so they remain behind everything
        for tree_id in tree_ids:

            x = (
                sim[sim["Tree_ID"] == tree_id]
                .sort_values("Datetime")
            )

            color = colors[(int(tree_id) - 1) % 10]

            if {
                "Fvfm_pred_min",
                "Fvfm_pred_max"
            }.issubset(x.columns):

                ax.fill_between(
                    x["Datetime"],
                    x["Fvfm_rel_pred"]-x["Fvfm_pred_sd"], # x["Fvfm_pred_min"],
                    x["Fvfm_rel_pred"]+x["Fvfm_pred_sd"], # x["Fvfm_pred_max"],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    zorder=1,
                )

                # # Thin boundaries help when the band is very narrow
                #     x["Datetime"],
                #     x["Fvfm_rel_pred"]-x["Fvfm_pred_sd"],#x["Fvfm_pred_min"],
                #     zorder=2,

                #     x["Datetime"],
                #     x["Fvfm_rel_pred"]+x["Fvfm_pred_sd"], #x["Fvfm_pred_max"],
                #     zorder=2,

        # Predicted means
        tree_markers = {
            1: "o",
            2: "s",
            3: "^",
            4: "D",
            5: "v",
            6: "P",
            7: "X",
            8: "<",
            9: ">",
            10: "*",
        }

        for tree_id in tree_ids:

            x = (
                sim[sim["Tree_ID"] == tree_id]
                .sort_values("Datetime")
            )

            color = colors[(int(tree_id) - 1) % 10]

            ax.plot(
                x["Datetime"],
                x["Fvfm_rel_pred"],
                color=color,
                lw=1.0,
                label=f"Tree {tree_id}",
                zorder=3,
            )

        # Observed tree means
        for tree_id in tree_ids:

            x = obs[obs["Tree_ID"] == tree_id]

            color = colors[(int(tree_id) - 1) % 10]

            #     x["Datetime"],
            #     x["Fvfm_rel_mean"],
            #     zorder=7,
            #     marker=tree_markers[int(tree_id)],
            #     edgecolor="white",
            #     linewidth=0.6

            ax.errorbar(
                x["Datetime"],
                x["Fvfm_rel_mean"],
                yerr=x["Fvfm_rel_sd"],
                marker=tree_markers[int(tree_id)],
                lw=0,
                elinewidth=0.5,
                capsize=2,
                markersize=7,
                #markeredgecolor="white",
                #markeredgewidth=0.6,
                zorder=10)

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Relative $F_{v}/F_{m}$ [-]")
        ax.set_xlabel("")
        ax.set_title("") #(name.replace("_", " + ").title())

        ax.axvspan(pd.Timestamp("2026-07-13 08:15:00"), pd.Timestamp("2026-07-18 00:00:00"), color='0.85', alpha=0.5, zorder=-10)
        ax.set_xlim(pd.Timestamp("2026-07-13 00:00:00"), pd.Timestamp("2026-07-22 00:00:00"))

        ax.xaxis.set_major_locator(
            mdates.DayLocator(interval=1)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d-%m") #m
        )

        ax.grid(alpha=0.2)

        legend_handles = [
            Line2D(
                [0], [0],
                color=colors[i],
                lw=1,
                marker=tree_markers[i+1],
                markersize=5,
                markerfacecolor=colors[i],
                markeredgecolor=colors[i],
                label=str(i+1)
            )
            for i in range(0, 10)
        ]

        leg = ax.legend(handles=legend_handles, title="Tree ID", loc="lower left", ncols=2, columnspacing=1.3, title_fontsize=11,
                handleheight=0.1, borderaxespad=0.3, handletextpad=0.5, handlelength=1.5, labelspacing=0.3)

        ax.text(0.05, 1.08, "Heatwave",
            transform=ax.transAxes,
            ha="left", va="top")

        ax.text(0.57, 1.08, "Recovery",
            transform=ax.transAxes,
            ha="left", va="top")

        ax.text(
            -0.085, 1.08, subplot_no,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold")

        fig.tight_layout()
        plt.show()


def plot_observed_vs_predicted_leaf_range(
    simulations,
    df_fvfm_leaf,
    obs_col="Fvfm_rel",
    subplot_no=""
):
    """
    Observed vs predicted relative Fv/Fm with leaf-level variability.

    Horizontal extent:
        ±1 SD of predictions from individual leaf-temperature simulations

    Vertical extent:
        ±1 SD of observed Fv/Fm among measured leaves

    Parameters
    ----------
    simulations : dict[str, DataFrame]
        Tree-level aggregated simulations containing:
            datetime
            Tree_ID
            Fvfm_rel_pred
            Fvfm_pred_min
            Fvfm_pred_max

    df_fvfm_leaf : DataFrame
        Leaf-level Fv/Fm measurements containing:
            datetime
            Tree_ID
            Leaf_ID
            obs_col

    obs_col : str
        Column containing relative Fv/Fm of individual leaves.
    """

    obs = df_fvfm_leaf.reset_index()
    # Aggregate leaf-level physiological measurements
    obs_summary = (
        obs
        .groupby(["Datetime", "Tree_ID"])[obs_col]
        .agg(
            Fvfm_obs_mean="mean",
            Fvfm_obs_min="min",
            Fvfm_obs_max="max",
            Fvfm_obs_sd="std",
            n_leaf="count",
        )
        .reset_index()
    )

    for name, sim in simulations.items():
        # Match observations to nearest simulation timestep
        sim = sim[
            [
                "Datetime",
                "Tree_ID",
                "Fvfm_rel_pred",
                "Fvfm_pred_min",
                "Fvfm_pred_max",
                "Fvfm_pred_sd"
            ]
        ].copy()

        sim["Tree_ID"] = sim["Tree_ID"].astype(int)
        obs_summary["Tree_ID"] = obs_summary["Tree_ID"].astype(int)

        sim = sim.sort_values(["Datetime", "Tree_ID"])
        obs_summary = obs_summary.sort_values(
            ["Datetime", "Tree_ID"]
        )

        df = pd.merge_asof(
            obs_summary,
            sim,
            on="Datetime",
            by="Tree_ID",
            direction="nearest",
        )

        df = df.dropna(
            subset=[
                "Fvfm_obs_mean",
                "Fvfm_rel_pred",
                "Fvfm_pred_min",
                "Fvfm_pred_max",
                "Fvfm_pred_sd"
            ]
        )

        # The supplied figure uses symmetric ±1 SD bars, not the min/max ranges.

        xerr = np.vstack([
            df["Fvfm_pred_sd"],
            df["Fvfm_pred_sd"],
        ])

        yerr = np.vstack([
            df["Fvfm_obs_sd"],
            df["Fvfm_obs_sd"],
        ])

        xerr = np.clip(xerr, 0, 100)
        yerr = np.clip(yerr, 0, 100)

        # Plot
        tree_markers = {
            1: "o",
            2: "s",
            3: "^",
            4: "D",
            5: "v",
            6: "P",
            7: "X",
            8: "<",
            9: ">",
            10: "*",
        }
        fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=300)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, row in df.reset_index(drop=True).iterrows():

            color = colors[(int(row["Tree_ID"]) - 1) % 10]

            ax.errorbar(
                row["Fvfm_rel_pred"],
                row["Fvfm_obs_mean"],
                xerr=np.array([
                    [xerr[0, i]],
                    [xerr[1, i]]
                ]),
                yerr=np.array([
                    [yerr[0, i]],
                    [yerr[1, i]]
                ]),
                fmt=tree_markers[int(row["Tree_ID"])],
                color=color,
                ecolor=color,
                alpha=0.8,

                elinewidth=0.5,
                capsize=2,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )

        lim = [
            min(
                df["Fvfm_pred_min"].min(),
                df["Fvfm_obs_min"].min()
            ),
            max(
                df["Fvfm_pred_max"].max(),
                df["Fvfm_obs_max"].max()
            ),
        ]

        ax.plot(
            lim,
            lim,
            ls="--",
            color="black",
            lw=1.2,
            label="1:1"
        )



        # Regression using the MEANS only
        X = df[["Fvfm_rel_pred"]]
        y = df["Fvfm_obs_mean"]

        reg = LinearRegression().fit(X, y)

        xx = np.linspace(
            df["Fvfm_rel_pred"].min(),
            df["Fvfm_rel_pred"].max(),
            100
        )

        yy = reg.predict(pd.DataFrame({"Fvfm_rel_pred": xx}))

        ax.plot(
            xx,
            yy,
            color="black",
            lw=1,
            zorder=10
        )
        pr = pearsonr(y, df["Fvfm_rel_pred"])
        rmse = root_mean_squared_error(
            y,
            df["Fvfm_rel_pred"]
        )

        r2 = r2_score(
            y,
            df["Fvfm_rel_pred"]
        )

        bias = np.mean(
            df["Fvfm_rel_pred"]
            - y
        )

        ax.text(
            0.02,
            1.075,
            f"R² = {r2:.2f}   "
            f"r = {pr[0]:.2f}   "
            f"RMSE = {rmse:.2f}   "
            f"Bias = {bias:.2f}",
            bbox=dict(facecolor='0.85', edgecolor="darkgrey", alpha=0.4, boxstyle='round'),
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            ha="left"
        )

        if name!="damage_repair_acclimation":
            ax.text(
                0.03,
                0.97,
                f"y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}",
                transform=ax.transAxes,
                fontsize=11,
                va="top",
                ha="left"
            )
        else:
            ax.text(
                0.65,
                0.07,
                f"y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}",
                transform=ax.transAxes,
                fontsize=11,
                va="top",
                ha="left"
            )

        ax.set_xlabel("Predicted relative $F_{v}/F_{m}$ [-]")
        ax.set_ylabel("Observed relative $F_{v}/F_{m}$ [-]")

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

        ax.grid(alpha=0.2)
        ax.text(
            -0.18, 1.08, subplot_no,
            transform=ax.transAxes,
            ha="left", va="top",
            fontweight="bold")

        fig.tight_layout()
        plt.show()


def plot_tree_panels(
    simulations,
    df_fvfm_leaf,
    obs_col="Fvfm_rel",
):
    """Plot one 2 x 5 tree-panel figure per model variant.

    Lines are mean predictions, shading is the predicted leaf min/max range,
    open points are individual measured leaves, and filled points are observed
    tree means.
    """

    for name, sim in simulations.items():
        obs = df_fvfm_leaf.reset_index().copy()

        tree_ids = sorted(sim["Tree_ID"].unique())

        fig, axes = plt.subplots(
            2,
            5,
            figsize=(12, 5),
            sharex=True,
            sharey=True,
            dpi=300,
        )

        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for ax, tree_id in zip(axes, tree_ids):

            color = colors[(int(tree_id) - 1) % 10]

            s = (
                sim[sim["Tree_ID"] == tree_id]
                .sort_values("Datetime")
            )

            o = (
                obs[obs["Tree_ID"] == tree_id]
                .sort_values("Datetime")
            )

            # Prediction range
            ax.fill_between(
                s["Datetime"],
                s["Fvfm_pred_min"],
                s["Fvfm_pred_max"],
                color=color,
                alpha=0.18,
                linewidth=0,
                label="Predicted leaf range",
            )

            # Prediction mean
            ax.plot(
                s["Datetime"],
                s["Fvfm_rel_pred"],
                color=color,
                lw=1.8,
                label="Predicted mean",
            )

            # Individual observed leaves
            ax.scatter(
                o["Datetime"],
                o[obs_col],
                facecolors="none",
                edgecolors=color,
                alpha=0.65,
                s=24,
                linewidth=0.8,
                label="Observed leaves",
                zorder=4,
            )

            # Mean observed Fv/Fm at each Campaign
            obs_mean = (
                o.groupby("Datetime")[obs_col]
                .mean()
                .reset_index()
            )

            ax.scatter(
                obs_mean["Datetime"],
                obs_mean[obs_col],
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=45,
                label="Observed mean",
                zorder=5,
            )

            ax.set_title(f"Tree {tree_id}")

            ax.set_ylim(-0.05, 1.05)

            ax.grid(alpha=0.2)

            ax.xaxis.set_major_locator(
                mdates.DayLocator(interval=2)
            )

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%d-%#m")
            )
            ax.set_xlim(pd.Timestamp("2026-07-13 00:00:00"), pd.Timestamp("2026-07-22 00:00:00"))

        # Axis labels only on outer panels
        for ax in axes[5:]:
            ax.set_xlabel("")

        axes[0].set_ylabel("Relative $F_{v}/F_{m}$ [-]")
        axes[5].set_ylabel("Relative $F_{v}/F_{m}$ [-]")

        # One common legend
        handles, labels = axes[0].get_legend_handles_labels()

        # remove duplicate labels
        unique = dict(zip(labels, handles))

        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=4,
            frameon=False,
            fontsize=12
        )

        fig.suptitle(
            name.replace("_", " + ").title(),
             y=1.03,
            fontsize=14,
        )

        fig.tight_layout(
            rect=[0, 0, 1, 0.94]
        )

        plt.show()


def plot_observed_vs_predicted_mean(simulations, df_fvfm):
    """Compare predicted and observed means by date/Campaign for each variant; 
    observations are averaged across trees."""

    for name, sim in simulations.items():

        df_combined = match_simulation_to_observations(sim, df_fvfm)

        x = "Fvfm_rel_pred"
        y = "Fvfm_rel_mean"
        df_combined["Date"] = pd.to_datetime(df_combined["Date"], format="%d.%m.%Y")
        df_mean = df_combined.groupby(["Date", "Campaign"]).mean(numeric_only=True).reset_index()
        fig, ax = plt.subplots(figsize=(2.7, 2.5), dpi=500)
        sns.scatterplot(data=df_mean, x=x, y=y,
                        s=60, alpha=0.65, ax=ax, color="darkgrey")

        # Fit linear regression
        subset = df_mean[[x, y]].dropna(axis=0)
        x_arr = subset[x].to_numpy().reshape(-1,1)
        y_arr = subset[y].to_numpy().reshape(-1,1)
        reg = LinearRegression().fit(x_arr, y_arr)
        a, b = reg.coef_[0][0], reg.intercept_[0]
        lims = [0.48, 1.02]

        x_vals = np.arange(lims[0], lims[1], step=0.1).reshape(-1,1)
        y_pred = reg.predict(x_vals)

        # Calculate metrics
        r2 = r2_score(subset[y], subset[x])
        rmse = root_mean_squared_error(subset[x], subset[y])
        bias = np.mean(subset[x] - subset[y])
        pr = pearsonr(subset[y], subset[x])

        ax.plot(lims, lims, color="black", linestyle="--", lw=1, zorder=-10)
        ax.plot(x_vals, y_pred, color="black", linestyle="-", lw=1)
        ax.text(
            0.04,
            0.96,
            f"R² = {r2:.2f}",
            bbox=dict(facecolor='0.85', edgecolor="darkgrey", alpha=0.4, boxstyle='round'),
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="left"
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.set_ylim(0.45, 1.05)
        ax.set_xlim(0.45, 1.05)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))

        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()


def plot_recovery_rate(recovery_fit, Th):
    """Plot the fitted base repair rate and capacity curve using the fit's parameter settings."""
    fig, ax = plt.subplots(2,1, figsize=(4.5, 4.2), dpi=300)

    # First plot
    r_ref = recovery_fit["r_ref"]
    T = np.linspace(5, 50, 300)
    if "Th" in recovery_fit and not np.isclose(Th, recovery_fit["Th"]):
        raise ValueError("Plot Th differs from the fitted Th; use the same run-script TH.")
    r = repair_rate_schoolfield(
        T, r_ref=r_ref, Th=Th, T_ref=recovery_fit.get("T_ref", 25.0),
        Ea=recovery_fit.get("Ea", 0.8e5), Eh=recovery_fit.get("Eh", 3.8e5),
    )
    ax[0].plot(T, 100 * r, c="black", lw=1)
    ax[0].set_xlabel("Temperature [°C]")
    ax[0].set_ylabel("Repair rate\n[% min$^{-1}$]")
    ax[0].grid(alpha=0.25)
    ax[0].text(-0.3, 1.15, "(a)",
        transform=ax[0].transAxes,
        ha="left", va="top",
        fontweight="bold")

    ## Second plot
    I_half = recovery_fit["I_scale"]
    if I_half is not None:
        impairments = np.arange(0, 1, 0.01)
        repair_caps = repair_capacity_factor(impairments, I_half, p=recovery_fit.get("p_repair", 6.0))

        ax[1].plot(impairments, repair_caps, lw=1, color="black")
        ax[1].axvline(I_half, ls="--", lw=1, c="darkgrey")
        ax[1].text(I_half+0.03, 0.65, "$I_{50}$", c="darkgrey")

        ax[1].set_xlabel("Impairment (1 - relative $F_{v}/F_{m}$) [-]" )
        ax[1].set_ylabel("Repair\ncapacity [-]")

        ax[1].set_ylim(-0.05, 1.05)
        ax[1].grid(alpha=0.3)
        ax[1].text(-0.3, 1.15, "(b)",
            transform=ax[1].transAxes,
            ha="left", va="top",
            fontweight="bold")

        fig.tight_layout()
        plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_acclimation_target(A_max, T_acclim, beta):
    """Plot the equilibrium acclimation shift (°C) against temperature; this is the A_target diagnostic."""
    T = np.linspace(25, 50, 300)
    A = [acclimation_target(t, A_max, T_acclim, beta) for t in T]
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)
    ax.plot(T, A, c="black", lw=1)
    ax.axvline(T_acclim, ls="--", lw=1, c="darkgrey")
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("$A_{\\mathrm{target}}$ [°C]")
    ax.grid(alpha=0.25)
    ax.text(
        -0.2, 1.08, "(a)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontweight="bold")
    fig.tight_layout()
    plt.show()


def plot_reference_acclimation(A_series, df_temp_acc, df_acc_mean, df_acc_std):
    """Plot the reference experiment's T50 evolution with treatment means and SD; A_series is in °C."""
    df_temp_acc = df_temp_acc.copy()  # Plotting must not modify the run script's temperature table.
    df_temp_acc["T50"] = df_acc_mean.loc[(df_acc_mean["Period"]=="Pre") & (df_acc_mean["Treatment"]=="Heat"), "T50"].values[0] + A_series
    df_temp_acc = df_temp_acc.loc[df_temp_acc.index > pd.Timestamp("2024-07-24")]
    subset = df_acc_mean[(df_acc_mean["Treatment"]=="Heat") & (df_acc_mean["Date"]> pd.Timestamp("2024-07-24"))]
    subset_std = df_acc_std[(df_acc_std["Treatment"]=="Heat") & (df_acc_std["Date"]> pd.Timestamp("2024-07-24"))]

    control = df_acc_mean[(df_acc_mean["Treatment"]=="Control") & (df_acc_mean["Date"]> pd.Timestamp("2024-07-24"))]
    control_std = df_acc_std[(df_acc_std["Treatment"]=="Control") & (df_acc_std["Date"]> pd.Timestamp("2024-07-24"))]

    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=300)
    ax.errorbar(subset["Date"], subset["T50"], yerr=subset_std["T50_std"],
                label="Heat", color="darkgrey", ecolor="darkgrey",
                markeredgecolor="white", markeredgewidth=0.6,
                markerfacecolor="darkgrey", marker="o", markersize=9, zorder=10,
                capsize=2, elinewidth=0.5, lw=0)
    ax.scatter(subset["Date"], subset["T50"], s=90, c="darkgrey", edgecolors="white")

    ax.errorbar(control["Date"], control["T50"], yerr=control_std["T50_std"],
                label="Control", color="dimgrey", ecolor="dimgrey",
                markeredgecolor="dimgrey", markeredgewidth=0.4,
                markerfacecolor="white", marker="o", markersize=8, zorder=10,
                capsize=2, elinewidth=0.5, lw=0)
    ax.plot(df_temp_acc.index, df_temp_acc["T50"], label="Modelled", c="black", lw=0.7, zorder=100)

    ax.set_xticks([pd.Timestamp("2024-07-24 00:00:00"), pd.Timestamp("2024-08-03 00:00:00"), pd.Timestamp("2024-08-09 00:00:00"), pd.Timestamp("2024-08-24 00:00:00"), pd.Timestamp("2024-08-30 00:00:00")])
    ax.set_xticklabels(["Pre-HW", "HW 1", "Post-HW 1", "HW 2", "Post-HW 2"], rotation=30, horizontalalignment="center")
    ax.xaxis.set_tick_params(pad=1)
    ax.set_xlabel("")
    ax.set_ylabel("$T_{\\mathrm{50,15min}}$ [°C]")
    ax.grid(alpha=0.25)
    ax.legend(title="", handlelength=1.5)

    ax.axvspan(pd.Timestamp("2024-08-03 00:00:00"), pd.Timestamp("2024-08-09 00:00:00"), color="lightgrey", alpha=0.5, zorder=-10)
    ax.axvspan(pd.Timestamp("2024-08-24 00:00:00"), pd.Timestamp("2024-08-30 00:00:00"), color="lightgrey", alpha=0.5, zorder=-10)

    #     mdates.DayLocator(interval=14)
    #     mdates.DateFormatter("%d-%m")

    ax.text(
        -0.2, 1.08, "(a)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontweight="bold")
    ax.set_ylim(49.2, 53.8)
    ax.set_xlim(pd.Timestamp("2024-07-21"), pd.Timestamp("2024-09-29"))
    fig.tight_layout()
    plt.show()


def plot_recovery_diagnostics(
    recovery_fit,
    hue="period"
):
    """
    Diagnostics for all observed recovery intervals, including intervals excluded from fitting.

    Produces:
        1. observed vs predicted Fv/Fm
        2. residual vs starting impairment
    """

    df = recovery_fit["data"].copy()

    # 1. Observed versus predicted

    fig, ax = plt.subplots(
        figsize=(4.5, 4.2),
        dpi=300
    )

    sns.scatterplot(
        data=df,
        x="Fvfm_obs",
        y="Fvfm_pred",
        style=hue, #"used_for_fit"
        s=70,
        c="darkgrey",
        ec="white",
        lw=0.6,
        ax=ax
    )

    valid = df[
        ["Fvfm_obs", "Fvfm_pred"]
    ].dropna()

    lim_min = min(
        valid["Fvfm_obs"].min(),
        valid["Fvfm_pred"].min()
    )

    lim_max = max(
        valid["Fvfm_obs"].max(),
        valid["Fvfm_pred"].max()
    )

    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        color="black",
        ls="--",
        lw=1
    )

    r2 = r2_score(valid["Fvfm_obs"], valid["Fvfm_pred"])

    ax.text(
        0.05,
        0.95,
        f"R² = {r2:.2f}\n",
        transform=ax.transAxes,
        va="top",
    )

    ax.set_xlabel("Observed relative $F_{v}/F_{m}$")
    ax.set_ylabel("Predicted relative $F_{v}/F_{m}$")

    ax.grid(alpha=0.3)
    ax.legend(title="Period", handletextpad=0.2)
    fig.tight_layout()
    plt.show()

    # 2. Residual versus impairment
    fig, ax = plt.subplots(
        figsize=(4.5, 4.2),
        dpi=300
    )

    sns.scatterplot(
        data=df,
        x="impairment_start",
        y="residual",
        style="period",
        s=70,
        c="darkgrey",
        ec="white",
        lw=0.6,
        ax=ax
    )

    ax.axhline(
        0,
        color="black",
        ls="--",
        lw=1
    )

    subset = df[
        ["impairment_start", "residual"]
    ].dropna()

    if len(subset) > 1:

        reg = LinearRegression().fit(
            subset[["impairment_start"]],
            subset["residual"]
        )

        x_pred = np.linspace(
            subset["impairment_start"].min(),
            subset["impairment_start"].max(),
            100
        )

        y_pred = reg.predict(pd.DataFrame({"impairment_start": x_pred}))

        ax.plot(
            x_pred,
            y_pred,
            color="black",
            lw=1
        )

        r2 = reg.score(
            subset[["impairment_start"]],
            subset["residual"]
        )

        ax.text(
            0.05,
            0.95,
            f"Slope = {reg.coef_[0]:.3f}\n"
            f"R² = {r2:.2f}",
            transform=ax.transAxes,
            va="top"
        )

    ax.set_xlabel(
        "Initial impairment (1 - relative $F_{v}/F_{m}$)"
    )

    ax.set_ylabel(
        "Residual (observed - predicted)"
    )

    ax.grid(alpha=0.3)
    ax.legend(title="Period", handletextpad=0.2)
    fig.tight_layout()
    plt.show()


