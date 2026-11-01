"""Compare static and cumulative thermal safety margins for synthetic heat scenarios.

Temperatures and temperature margins are in °C; time is in minutes. Thermal load
is dimensionless, normalised so load = 1 corresponds to the T50 endpoint under
constant-temperature exposure. Additive accumulation is assumed, with no recovery.
The scenarios are illustrative, not measured leaf-temperature observations.

Dependencies: numpy, pandas, scipy, matplotlib, seaborn. 
"""

# ------------------------ #
# Imports and run settings #
# ------------------------ #
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.stats import norm

RANDOM_SEED = 4
TIME_STEP_MIN = 15
MAXIMUM_TEMPERATURE = 45
ASSAY_DURATIONS = (15, 30, 60, 240)
SHOW_PLOTS = True  # Set False for non-interactive runs.
SPECIES_NAMES = {
    "AP": "Acer platanoides", "CS": "Castanea sativa",
    "FS": "Fagus sylvatica", "QP": "Quercus petraea",
}
SPECIES_COLORS = dict(zip(SPECIES_NAMES.values(), ["darkblue", "darkmagenta", "goldenrod", "crimson"]))
DURATION_LIGHTNESS = [0.8, 0.55, 0.30, 0.05]

# ----------------------------------------- #
# Input validation and thermal-load helpers #
# ----------------------------------------- #
def validate_inputs(assay_data, tdt_parameters):
    """Validate required values and return a copy with consistent species abbreviations."""
    assay_columns = {"Species", "Duration", "T50"}
    parameter_columns = {"Species", "a_mean", "b_mean"}
    for data, required, label in ((assay_data, assay_columns, "assay"),
                                  (tdt_parameters, parameter_columns, "TDT parameter")):
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {sorted(missing)}")
    assay_data = assay_data.copy()
    expected_codes = assay_data["Species"].map({name: code for code, name in SPECIES_NAMES.items()})
    if expected_codes.isna().any():
        raise ValueError("The assay contains missing or unsupported species names.")
    if "Species_short" not in assay_data:
        assay_data["Species_short"] = expected_codes
    if not assay_data["Species_short"].eq(expected_codes).all():
        raise ValueError("Species_short does not match the scientific species names.")
    if not np.isfinite(assay_data[["Duration", "T50"]].to_numpy(dtype=float)).all():
        raise ValueError("Assay durations and T50 estimates must be finite numeric values.")
    if (assay_data["Duration"] <= 0).any() or assay_data.duplicated(["Species", "Duration"]).any():
        raise ValueError("Assay durations must be positive, with one estimate per species and duration.")
    for code, name in SPECIES_NAMES.items():
        available = set(assay_data.loc[assay_data["Species_short"] == code, "Duration"])
        if not set(ASSAY_DURATIONS) <= available:
            raise ValueError(f"Missing assay durations for {name}: {sorted(set(ASSAY_DURATIONS) - available)}")
        parameters = tdt_parameters.loc[tdt_parameters["Species"] == name, ["a_mean", "b_mean"]]
        if len(parameters) != 1 or not np.isfinite(parameters.to_numpy(dtype=float)).all():
            raise ValueError(f"Expected one finite TDT parameter pair for {name}.")
        if parameters["a_mean"].iloc[0] >= 0:
            raise ValueError(f"A negative TDT slope is required for {name}.")
    return assay_data

def thermal_load_rate(temperature, a, b):
    """Return thermal-load accumulation rate (min⁻¹), normalised to the T50 endpoint.

    T50(t) = a * ln(t) + b implies time_to_T50(T) = exp((T - b) / a).
    This rate accumulates thermal load, not a measured percentage of impairment.
    """
    return 1 / np.exp((temperature - b) / a)

# ---------------------------------- #
# 1. Synthetic temperature scenarios #
# ---------------------------------- #
def simulate_temperatures():
    """Return prolonged-heat (Day I) and short-spike (Day II) scenarios on a 15-min grid."""
    rng = np.random.RandomState(RANDOM_SEED)    
    times = pd.date_range("2023-07-01", periods=96, freq="15min")
    sample_index = np.arange(96)
    day_factor = np.clip(np.sin((sample_index - 12) / 96 * np.pi), 0, 1)
    _ = rng.normal(0, 0.8, len(sample_index))

    # 1.1 Day I: blend a sinusoid towards a broad, elevated afternoon plateau.
    baseline_a = 32 + 13 * np.sin((sample_index - 34) / 96 * 2 * np.pi)
    plateau_weight_a = np.exp(-0.5 * ((sample_index - 58) / 10) ** 2)
    temperature_a = (1 - plateau_weight_a) * baseline_a + plateau_weight_a * 43.3
    temperature_a += rng.normal(0, 0.9, len(sample_index)) * day_factor
    #temperature_a += MAXIMUM_TEMPERATURE - temperature_a.max()

    # 1.2 Day II: use a narrower afternoon spike with a separate noise realisation.
    baseline_b = 30 + 13 * np.sin((sample_index - 36) / 96 * 2 * np.pi)
    plateau_weight_b = np.exp(-0.5 * ((sample_index - 62) / 1) ** 2)
    temperature_b = (1 - plateau_weight_b) * baseline_b + plateau_weight_b * 43.3
    temperature_b += rng.normal(0, 1, len(sample_index)) * day_factor
   # temperature_b += MAXIMUM_TEMPERATURE - temperature_b.max()
    return pd.DataFrame({
        "Datetime": np.tile(times, 2), "Temperature": np.concatenate([temperature_a, temperature_b]),
        "Scenario": ["Day I"] * 96 + ["Day II"] * 96,
    })

# --------------------------------- #
# 2. Dynamic thermal safety margins #
# --------------------------------- #
def calculate_dynamic_margins(scenario_data, tdt_parameters):
    """Calculate load increments, remaining margins, and equivalent temperatures.

    Dose/Dose_cum are dimensionless loads; TSM_rel is the remaining budget in %. 
    T_eq_* and TSM_eq_* are in °C.
    Time_to_fail is in minutes; 
    A remaining budget/margin of zero marks the T50 endpoint; negative margins indicate exceedance.
    """
    margin_frames = []
    for code, name in SPECIES_NAMES.items():
        parameters = tdt_parameters.loc[tdt_parameters["Species"] == name].iloc[0]
        a, b = parameters["a_mean"], parameters["b_mean"]
        for scenario, scenario_temperatures in scenario_data.groupby("Scenario", sort=False):
            # 2.1 Integrate the rate using a rectangular sum; every sample represents 15 min.
            # The first cumulative value already includes the first interval's load.
            leaf_temperature = scenario_temperatures["Temperature"].to_numpy()
            accumulation_rate = thermal_load_rate(leaf_temperature, a, b)
            load_increment = accumulation_rate * TIME_STEP_MIN
            cumulative_load = load_increment.cumsum(axis=0)
            remaining_load_percent = 100 - cumulative_load * 100

            # 2.2 Express the same cumulative load as a constant-temperature exposure.
            equivalent_temperature_15 = b - a * np.log(cumulative_load / 15)
            equivalent_temperature_240 = b - a * np.log(cumulative_load / 240)
            temperature_margin_15 = a * np.log(15) + b - equivalent_temperature_15
            temperature_margin_240 = a * np.log(240) + b - equivalent_temperature_240
            # Both margins equal a * ln(cumulative_load); reference duration cancels.

            # Time remaining assumes the current temperature persists after the current interval.
            time_to_endpoint = np.maximum(0, (1 - cumulative_load) / accumulation_rate)
            margin_frames.append(pd.DataFrame({
                "Species": code, "Scenario": scenario, "Dose": load_increment, "Dose_cum": cumulative_load,
                "TSM_rel": remaining_load_percent, "T_eq_15": equivalent_temperature_15,
                "TSM_eq_15": temperature_margin_15, "T_eq_240": equivalent_temperature_240,
                "TSM_eq_240": temperature_margin_240, "Time_to_fail": time_to_endpoint, 
                "Time": scenario_temperatures["Datetime"].to_numpy()
            }))
    return pd.concat(margin_frames, ignore_index=True)

# -------------------------------- #
# 3. Static thermal safety margins #
# -------------------------------- #
def calculate_static_margins(assay_data, scenario_data):
    """Return assay/scenario tables with static thermal safety margins for duration-dependent thermal thresholds 
    and their minima across the scenarios.
    Static margin = observed assay T50 minus instantaneous or rolling-mean temperature (matching assay durations).

    """
    assay_data, scenario_data = assay_data.copy(), scenario_data.copy()
    # 3.1 The ranking panels use the common maximum temperature for both scenarios.
    assay_data["tsm"] = assay_data["T50"] - MAXIMUM_TEMPERATURE
    for scenario in scenario_data["Scenario"].unique():
        scenario_mask = scenario_data["Scenario"] == scenario
        temperatures = scenario_data.loc[scenario_mask].set_index("Datetime").sort_index()["Temperature"]
        # 3.2 Calculate rolling mean temperatures matching the assay durations.
        temperature_by_window = {15: temperatures.to_numpy()}
        temperature_by_window.update(
            {window: temperatures.rolling(f'{window}min').mean().to_numpy() for window in (30, 60, 240)},
        )
        for duration in ASSAY_DURATIONS:
            for code in SPECIES_NAMES:
                assay_mask = (assay_data["Species_short"] == code) & (assay_data["Duration"] == duration)
                threshold_temperature = assay_data.loc[assay_mask, "T50"].iloc[0]
                for window, leaf_temperature in temperature_by_window.items():
                    column = f"TSM_{code}_{duration}_{window}min"
                    scenario_data.loc[scenario_mask, column] = threshold_temperature - leaf_temperature
                    assay_data.loc[assay_mask, f'tsm_{window}_{scenario}'] = scenario_data.loc[scenario_mask, column].min(
                    )

    # 3.3 Retain minima where the rolling-mean window equals the assay exposure duration.
    margin_columns = [column for column in assay_data if column.startswith("tsm_")]
    static_minima = assay_data.melt(
        id_vars=[column for column in assay_data if column not in margin_columns], value_vars=margin_columns,
        var_name="variable", value_name="tsm_min",
    )
    static_minima[["tleaf_resolution", "scenario"]] = static_minima["variable"].str.extract(
        'tsm_(\\d+)_Day\\s*([A-Za-z]+)',
    )
    static_minima["tleaf_resolution"] = static_minima["tleaf_resolution"].astype(int)
    static_minima = static_minima.drop(columns="variable")
    static_minima = static_minima[static_minima["tleaf_resolution"] == static_minima["Duration"]].copy()
    return assay_data, scenario_data, static_minima

# -------------- #
# Figure helpers #
# -------------- #
def lighten_color(color, amount=0.5):
    """Mix a colour with white: amount=0 retains the colour; amount=1 returns white."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)

def save_and_show_figure(fig, name, directory):
    """Save PDF and 300-dpi PNG copies before optionally displaying the figure."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f'{name}.{extension}', dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

# --------------------------- #
# 4. Static manuscript figure #
# --------------------------- #
def plot_static_margins(assay_data, scenario_data, figures_dir):
    """Reproduce the static-margin figure using the original panel layout and styling.

    Species panels use running minima of T50 minus instantaneous 15-min temperatures.
    Exported static minima instead match the trailing-mean window to assay duration.
    """
    species_names, species_colors = SPECIES_NAMES, SPECIES_COLORS
    species_codes = list(SPECIES_NAMES)
    maximum_temperature = MAXIMUM_TEMPERATURE
    duration_lightness = DURATION_LIGHTNESS
    fig, axes = plt.subplots(
        2, 8, figsize=(13, 4.5), dpi=300, sharey="col", sharex="col",
        gridspec_kw={"width_ratios": [1, 0.2, 1, 1, 1, 1, 0, 1], "wspace": 0.2, "hspace": 0.2},
    )
    for scenario_index, scenario in enumerate(["Day I", "Day II"]):
        # 4.1 Show the same synthetic temperature scenario in the first panel.
        scenario_temperatures = scenario_data[scenario_data["Scenario"] == scenario].copy()
        sns.lineplot(
            data=scenario_temperatures, x="Datetime", y="Temperature", style="Scenario", c="black",
            ax=axes[scenario_index, 0], lw=0.7,
        )
        axes[scenario_index, 0].grid(which="both", alpha=0.3)
        axes[scenario_index, 0].set_xlim(
            pd.Timestamp("2023-07-01 03:00:00"), pd.Timestamp("2023-07-02 00:00:00"),
        )
        axes[scenario_index, 0].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        axes[scenario_index, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        axes[scenario_index, 0].set_xlabel("")
        axes[scenario_index, 0].set_ylabel("")
        axes[scenario_index, 0].get_legend().set_visible(False)
        axes[scenario_index, 0].set_ylim(10, 50)
        axes[scenario_index, 0].axhline(
            maximum_temperature, ls="dashed", lw=1, c="grey", label=f'{maximum_temperature}°C',
        )
        axes[scenario_index, 0].text(
            0.08, 0.83, f'{maximum_temperature}°C', c="grey", backgroundcolor="white",
            transform=axes[scenario_index, 0].transAxes,
        )
        axes[0, 0].set_title("Leaf temperature")
        axes[1, 0].set_title("")

        # 4.2 Show the running minimum margin for each assay duration.
        for species_panel, species in enumerate(species_codes):
            species_panel += 2
            margin_columns = [f"TSM_{species}_{duration}_15min" for duration in ASSAY_DURATIONS]

            for duration_index, margin_column in enumerate(margin_columns):
                scenario_temperatures[f'{margin_column}_min'] = scenario_temperatures[f'{margin_column}'].cummin(
                )
                axes[scenario_index, species_panel].fill_between(
                    scenario_temperatures["Datetime"], 0, scenario_temperatures[f'{margin_column}_min'],
                    alpha=1,
                    color=lighten_color(species_colors[species_names[species]], duration_lightness[duration_index]),
                    lw=1,
                )
            axes[scenario_index, species_panel].grid(which="both", alpha=0.3)
            axes[scenario_index, species_panel].set_ylabel("")
            axes[scenario_index, species_panel].set_xlabel("")
            axes[scenario_index, species_panel].set_ylim(-4, 11)
            axes[scenario_index, species_panel].xaxis.set_major_locator(
                mdates.HourLocator(byhour=range(0, 24, 6)),
            )
            axes[scenario_index, species_panel].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            if scenario_index == 0:
                axes[scenario_index, species_panel].set_title(species)
            axes[scenario_index, species_panel].axhline(0, lw=1, c="black")

        # Empty panels
        axes[scenario_index, 6].remove()
        axes[scenario_index, 1].remove()

        # 4.3 Rank species using T50 minus the shared 45 °C scenario maxima.
        sns.barplot(
            assay_data, x="Duration", y="tsm", hue="Species", ax=axes[scenario_index, 7],
            palette=species_colors,
        )

        axes[scenario_index, 7].set_ylabel("")
        axes[scenario_index, 7].set_xlabel('$T_{50}$ assay\nduration [min]')
        axes[scenario_index, 7].legend().set_visible(False)
        axes[scenario_index, 7].grid(which="both", alpha=0.3)
        axes[scenario_index, 7].set_ylim(-4, 11)
        axes[0, 7].set_title("Species ranking")
        axes[scenario_index, 7].axhline(0, lw=1, c="black")

    fig.text(0.05, 0.98, "(a)", weight="bold")

    # 4.4 Add the original figure labels and legend; preserve placement and formatting.
    fig.text(0.08, 0.5, "[°C]", va="center", rotation="vertical")
    fig.text(0.243, 0.5, "TSM [°C]", va="center", rotation="vertical")
    fig.text(0.16, 0.0, "Time", va="center", rotation="horizontal")
    fig.text(0.06, 0.7, "(I) Prolonged heat", rotation=90, ha="center", va="center", weight="bold")
    fig.text(0.06, 0.27, "(II) Short hot spike", rotation=90, ha="center", va="center", weight="bold")
    fig.text(0.52, 0.98, "Static TSMs", ha="center", va="center", weight="bold")

    # Legend
    labels = ["TSM$_{15\\min}$", "TSM$_{30\\min}$", "TSM$_{60\\min}$", "TSM$_{240\\min}$"]

    legend_handles = [
        Patch(facecolor=lighten_color("grey", duration_lightness[duration_index]), alpha=1, label=label)
        for duration_index, label in enumerate(labels)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=False, title="",
        bbox_to_anchor=(0.52, -0.05),
    )

    # Hide repeated tick labels in shared panels.
    for panel in axes[0,:].flatten():
        panel.tick_params(labelbottom=False)

    for panel in axes[:, 3:].flatten():
        panel.tick_params(labelleft=False)

    plt.tight_layout()
    save_and_show_figure(fig, "thermal_safety_margins_static", figures_dir)

# --------------------------------- #
# 5. Dynamic TSMs manuscript figure #
# --------------------------------- #
def plot_dynamic_margins(dynamic_margins, dynamic_minima, scenario_data, figures_dir):
    """Reproduce the dynamic-margin figure using the original panel layout and styling."""
    species_names, species_colors = SPECIES_NAMES, SPECIES_COLORS
    species_codes = list(SPECIES_NAMES)
    maximum_temperature = MAXIMUM_TEMPERATURE
    fig, axes = plt.subplots(
        2, 8, figsize=(13, 4.5), dpi=300, sharey="col", sharex="col",
        gridspec_kw={"width_ratios": [1, 0.2, 1, 1, 1, 1, 0, 1], "wspace": 0.2, "hspace": 0.2},
    )
    for scenario_index, scenario in enumerate(["Day I", "Day II"]):
        # 5.1 Show the same synthetic temperature scenario in the first panel.
        scenario_temperatures = scenario_data[scenario_data["Scenario"] == scenario].copy()
        sns.lineplot(
            data=scenario_temperatures, x="Datetime", y="Temperature", style="Scenario", c="black",
            ax=axes[scenario_index, 0], lw=0.7,
        )
        axes[scenario_index, 0].grid(which="both", alpha=0.3)
        axes[scenario_index, 0].set_xlim(
            pd.Timestamp("2023-07-01 03:00:00"), pd.Timestamp("2023-07-02 00:00:00"),
        )
        axes[scenario_index, 0].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        axes[scenario_index, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
        axes[scenario_index, 0].set_xlabel("")
        axes[scenario_index, 0].set_ylabel("")
        axes[scenario_index, 0].get_legend().set_visible(False)
        axes[scenario_index, 0].set_ylim(10, 50)
        axes[scenario_index, 0].axhline(maximum_temperature, ls="dashed", lw=1, c="grey")
        axes[scenario_index, 0].text(
            0.08, 0.83, f'{maximum_temperature}°C', c="grey", backgroundcolor="white",
            transform=axes[scenario_index, 0].transAxes,
        )
        axes[0, 0].set_title("Leaf temperature")
        axes[1, 0].set_title("")

        # 5.2 Overlay the remaining load budget (TLSM_rel) and equivalent-temperature margin (TLSM_eq).
        for species_panel, species in enumerate(species_codes):
            species_panel += 2
            # Relative margin: percent of the normalised thermal-load budget remaining.
            selected = (dynamic_margins["Scenario"] == scenario) & (dynamic_margins["Species"] == species)
            species_margins = dynamic_margins.loc[selected]

            axes[scenario_index, species_panel].fill_between(
                species_margins["Time"], 0, species_margins["TSM_rel"], alpha=0.3,
                color=species_colors[species_names[species]], lw=1, ls="solid",
            )

            axes[scenario_index, species_panel].grid(which="both", alpha=0.3)
            axes[scenario_index, species_panel].set_xlim(
                pd.Timestamp("2023-07-01 03:00:00"), pd.Timestamp("2023-07-02 00:00:00"),
            )
            axes[scenario_index, species_panel].xaxis.set_major_locator(
                mdates.HourLocator(byhour=range(0, 24, 3)),
            )
            axes[scenario_index, species_panel].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

            axes[scenario_index, species_panel].set_ylabel("")
            axes[scenario_index, species_panel].set_xlabel("")
            axes[scenario_index, species_panel].set_ylim(-50, 110)
            axes[scenario_index, species_panel].xaxis.set_major_locator(
                mdates.HourLocator(byhour=range(0, 24, 6)),
            )
            axes[scenario_index, species_panel].xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            if scenario_index == 0:
                axes[scenario_index, species_panel].set_title(species)

            # Equivalent margin in °C; the 240-minute reference is retained for the manuscript.
            equivalent_margin_ax = axes[scenario_index, species_panel].twinx()
            equivalent_margin_ax.fill_between(
                species_margins["Time"], 0, species_margins["TSM_eq_240"], alpha=0.4,
                color=species_colors[species_names[species]], lw=1, ls="solid", hatch="//", hatch_linewidth=3,
            )

            equivalent_margin_ax.set_ylim(-5, 11)
            equivalent_margin_ax.set_ylabel("")
            equivalent_margin_ax.tick_params(labelright=False)

            axes[scenario_index, species_panel].axhline(0, lw=1, c="black")

        # Empty panels
        axes[scenario_index, 6].remove()
        axes[scenario_index, 1].remove()

        # 5.3 Compare the minimum dynamic margins reached during each scenario.
        scenario_minima = dynamic_minima[dynamic_minima["Scenario"] == scenario].copy()
        scenario_minima["x_TSM"] = "TLSM$_{\\mathrm{rel}}$\n[%]"
        scenario_minima["x_Teq"] = "TLSM$_{\\mathrm{eq}}$\n[°C]"
        sns.barplot(
            scenario_minima, x="x_TSM", y="TSM_rel", hue="Species_long", ax=axes[scenario_index, 7],
            palette=species_colors, width=0.6,
        )
        axes[scenario_index, 7].legend().set_visible(False)
        axes[scenario_index, 7].grid(which="both", alpha=0.3)
        axes[scenario_index, 7].set_ylabel("")
        axes[scenario_index, 7].set_xlabel("")
        axes[scenario_index, 7].set_ylim(-50, 110)
        axes[scenario_index, 7].tick_params(labelright=False)
        axes[0, 7].set_title("Species ranking")

        ranking_equivalent_ax = axes[scenario_index, 7].twinx()
        sns.barplot(
            scenario_minima, x="x_Teq", y="TSM_eq_240", hue="Species_long", ax=ranking_equivalent_ax,
            palette=species_colors, width=0.6,
        )
        ranking_equivalent_ax.legend().set_visible(False)
        ranking_equivalent_ax.set_ylabel("")
        ranking_equivalent_ax.set_xlabel("")
        ranking_equivalent_ax.set_ylim(-5, 11)

        axes[scenario_index, 7].axhline(0, lw=1, c="black")
        ranking_equivalent_ax.axhline(0, lw=1, c="black")

    # 5.4 Add the original figure labels and legend; preserve placement and formatting.
    fig.text(0.08, 0.5, "[°C]", va="center", rotation="vertical")
    fig.text(0.236, 0.5, 'TLSM$_{\\mathrm{rel}}$ [%]', va="center", rotation="vertical")
    fig.text(0.93, 0.5, 'TLSM$_{\\mathrm{eq}}$ [°C]', va="center", rotation="vertical")
    fig.text(0.16, 0.0, "Time", va="center", rotation="horizontal")
    fig.text(0.06, 0.7, "(I) Prolonged heat", rotation=90, ha="center", va="center", weight="bold")
    fig.text(0.06, 0.27, "(II) Short hot spike", rotation=90, ha="center", va="center", weight="bold")
    fig.text(0.52, 0.98, "Dynamic TSMs", ha="center", va="center", weight="bold")
    fig.text(0.05, 0.98, "(b)", weight="bold")

    # Legend
    alphas = [0.3, 0.6]
    labels = ["TLSM$_{\\mathrm{rel}}$", "TLSM$_{\\mathrm{eq}}$"]
    hatches = ["", "//"]
    plt.rcParams["hatch.linewidth"] = 3
    legend_handles = [
        Patch(facecolor="grey", alpha=opacity, label=label, hatch=hatch)
        for opacity, label, hatch in zip(alphas, labels, hatches)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=2, frameon=False, title="",
        bbox_to_anchor=(0.52, -0.05),
    )

    # Hide repeated tick labels in shared panels.
    for panel in axes[0,:].flatten():
        panel.tick_params(labelbottom=False)

    for panel in axes[:, 3:].flatten():
        panel.tick_params(labelleft=False)

    save_and_show_figure(fig, "thermal_safety_margins_dynamic", figures_dir)

# ------------- #
# Main workflow #
# ------------- #
def main():
    """Load study inputs, calculate margins, reproduce figures, and export results."""
    # Read assay thresholds and parameters from the preceding thermal tolerance analysis.
    project_root = Path(__file__).resolve().parents[1]
    data_dir, results_dir = project_root / "data", project_root / "results"
    figures_dir = results_dir / "manuscript_figures"
    assay_file = data_dir / "2025_tolerance_assay" / "thermal_thresholds_by_duration.csv"
    assay_data = pd.read_csv(assay_file)
    tdt_parameters = pd.read_csv(results_dir / "thermal_tolerance_tables" / "tdt_parameters.csv")
    assay_data = validate_inputs(assay_data, tdt_parameters)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", font_scale=1.2, rc={
        "axes.linewidth": 1.0, "xtick.major.size": 5, "xtick.major.width": 1,
        "ytick.major.size": 5, "ytick.major.width": 1,
    })

    # Generate heat scenarios and calculate thermal safety margins.
    scenario_data = simulate_temperatures()
    dynamic_margins = calculate_dynamic_margins(scenario_data, tdt_parameters)
    assay_data, scenario_data, static_minima = calculate_static_margins(assay_data, scenario_data)
    dynamic_minima = dynamic_margins.groupby(["Scenario", "Species"])[["TSM_rel", "TSM_eq_240"]].min().reset_index(
    )
    dynamic_minima["Species_long"] = dynamic_minima["Species"].map(SPECIES_NAMES)
    plot_static_margins(assay_data, scenario_data, figures_dir)
    plot_dynamic_margins(dynamic_margins, dynamic_minima, scenario_data, figures_dir)

if __name__ == "__main__":
    main()
