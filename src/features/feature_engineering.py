import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import KMeans

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "features"))
)

from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from temporal_abstraction import NumericalAbstraction
from frequency_abstraction import FourierTransformation

plt.style.use("seaborn-v0_8-deep")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
df.info()

predictor_columns = list(df.columns[:6])

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

# Verify missing values
subset = df[df["set"] == 50]["gyr_y"]
subset.plot()
plt.show()

for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

# df[df["set"] == 10]["acc_y"].plot()
# plt.show()

# delta_duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
# delta_duration.seconds

# for s in df["set"].unique():
#     start = df[df["set"] == s].index[0]
#     stop = df[df["set"] == s].index[-1]

#     duration = stop - start
#     df.loc[(df["set"] == s), "duration"] = duration.seconds

# duration_df = df.groupby(["category"])["duration"].mean()

# # Heavy set divided by the amount of repetitions
# duration_df.iloc[0] / 5
# # Medium set divided by the amount of repetitions
# duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
lowpass = LowPassFilter()

# Between each record there is a 200ms interval
sampling_frequency = 1000 / 200  # Five instances per second
cutoff_frequency = 1.2  # The lower -> smoother

df_lowpass = lowpass.low_pass_filter(
    df_lowpass, "acc_y", sampling_frequency, cutoff_frequency, order=5
)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

# Verify whether the data is now smoother
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="Raw Data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="Butterworth Filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
plt.show()

# Filter all columns
for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(
        df_lowpass, col, sampling_frequency, cutoff_frequency, order=5
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
pca = PrincipalComponentAnalysis()

pca_values = pca.determine_pc_explained_variance(df_pca, predictor_columns)

# Elbow technique
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pca_values)
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance")
plt.show()

# 3 is the optimal number of components
df_pca = pca.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 45]
subset[["pca_1", "pca_2", "pca_3"]].plot()
plt.show()

# --------------------------------------------------------------
# Sum of squares
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 45]
subset[["acc_r", "gyr_r"]].plot(subplots=True)
plt.show()

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000 / 200)  # 5 instances per second

# for col in predictor_columns:
#     df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
#     df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

# df_temporal_list = []
# for s in df_temporal["set"].unique():
#     subset = df_temporal[df_temporal["set"] == s].copy()
#     for col in predictor_columns:
#         df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
#         df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")
#     df_temporal_list.append(subset)

df_temporal_list = []
for s in df_temporal["set"].unique():
    print(f"Apply Temporal Abstraction to set {s}")
    subset = df_temporal[df_temporal["set"] == s].copy()
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "mean")
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()

subset[["acc_r", "acc_r_temp_mean_ws_5", "acc_r_temp_std_ws_5"]].plot()
subset[["gyr_r", "gyr_r_temp_mean_ws_5", "gyr_r_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)  # 5 instances per second
ws = int(2800 / 200)  # 7.4 instances per second

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

df_freq.columns

subset = df_freq[df_freq["set"] == 45]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Apply Fourier Transformation to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows (15%)
# --------------------------------------------------------------

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Sum of Squared Distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()

# Plot accelerometer data
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-Axis")
ax.set_ylabel("Y-Axis")
ax.set_zlabel("Z-Axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
