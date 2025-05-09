import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "features"))
)

from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]  # Drop resting periods

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

df_bench = df[df["label"] == "bench"]
df_squat = df[df["label"] == "squat"]
df_row = df[df["label"] == "row"]
df_ohp = df[df["label"] == "ohp"]
df_dead = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = df_bench

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()

# The acc data show patterns more than the gyr data
# But we need some of smoothing, so we will use LowOassFilter

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200  # Frequency sampling
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = df_bench[df_bench["set"] == df_bench["set"].unique()[0]]
ohe_set = df_ohp[df_ohp["set"] == df_ohp["set"].unique()[0]]
squat_set = df_squat[df_squat["set"] == df_squat["set"].unique()[0]]
dead_set = df_dead[df_dead["set"] == df_dead["set"].unique()[0]]
row_set = df_row[df_row["set"] == df_row["set"].unique()[0]]

column = "acc_r"
LowPass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)
bench_set["acc_r_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

bench_set["acc_r"].values


def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    data[column + "_lowpass"].values
    indexes = argrelextrema(
        data[column + "_lowpass"].values, np.greater
    )  # Calculate the relative extrema of data
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(data[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = data["label"].iloc[0].title()
    category = data["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()

    return len(peaks)


count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(ohe_set, cutoff=0.9)
count_reps(row_set, cutoff=0.65, column="gyr_x")
count_reps(dead_set, cutoff=0.4)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
rep_df[rep_df["label"] == "bench"]
rep_df["reps_pred"] = 0

for s in df["set"].unique():
    subset = df[df["set"] == s]
    column = "acc_r"
    cutoff = 0.4

    if subset["label"][0] == "squar":
        cutoff = 0.35

    if subset["label"][0] == "row":
        cutoff = 0.65
        column = "gyr_x"

    if subset["label"][0] == "ohp" and subset["category"][0] == "heavy":
        cutoff = 0.35

    if subset["label"][0] == "ohp" and subset["category"][0] == "medium":
        cutoff = 0.8

    reps = count_reps(subset, column=column, cutoff=cutoff)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

mean_absolute_error(rep_df["reps"], rep_df["reps_pred"])
rep_df["reps"].mean()
rep_df.groupby(["label", "category"])["reps", "reps_pred"].mean().plot(kind="bar")
plt.show()
