from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
from minisom import MiniSom

## ========= data processing functions ========== ##

## Input the file path
def get_input_files():
    archive_dir = Path(r"D:\python\NU\MLproject\archive")
    return sorted(archive_dir.glob("*.csv"))

## load data
def load_data(file_path):
    return pd.read_csv(file_path)

## check the feature names
def check_feature_names(df):
    feature_names = ['X1_ActualPosition', 'X1_ActualVelocity', 'X1_ActualAcceleration', 
                     'X1_CommandPosition', 'X1_CommandVelocity', 'X1_CommandAcceleration', 
                     'X1_CurrentFeedback', 'X1_DCBusVoltage', 'X1_OutputCurrent', 
                     'X1_OutputVoltage', 'X1_OutputPower', 'Y1_ActualPosition', 
                     'Y1_ActualVelocity', 'Y1_ActualAcceleration', 'Y1_CommandPosition', 
                     'Y1_CommandVelocity', 'Y1_CommandAcceleration', 'Y1_CurrentFeedback', 
                     'Y1_DCBusVoltage', 'Y1_OutputCurrent', 'Y1_OutputVoltage', 
                     'Y1_OutputPower', 'Z1_ActualPosition', 'Z1_ActualVelocity', 
                     'Z1_ActualAcceleration', 'Z1_CommandPosition', 'Z1_CommandVelocity', 
                     'Z1_CommandAcceleration', 'Z1_CurrentFeedback', 'Z1_DCBusVoltage', 
                     'Z1_OutputCurrent', 'Z1_OutputVoltage', 'S1_ActualPosition', 
                     'S1_ActualVelocity', 'S1_ActualAcceleration', 'S1_CommandPosition', 
                     'S1_CommandVelocity', 'S1_CommandAcceleration', 'S1_CurrentFeedback', 
                     'S1_DCBusVoltage', 'S1_OutputCurrent', 'S1_OutputVoltage', 
                     'S1_OutputPower', 'S1_SystemInertia', 'M1_CURRENT_PROGRAM_NUMBER', 
                     'M1_sequence_number', 'M1_CURRENT_FEEDRATE', 'Machining_Process']
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    return

## leave processing rows
def leave_processing(df):
    process = [
        "Layer 1 Up", "Layer 1 Down",
        "Layer 2 Up", "Layer 2 Down",
        "Layer 3 Up", "Layer 3 Down"
    ]

    if "Machining_Process" not in df.columns:
        raise KeyError("Column 'Machining_Process' not found")

    # Only keep rows where 'Machining_Process' is in process
    filtered_df = df[df["Machining_Process"].isin(process)].copy()
    return filtered_df

## check for not accurately reflect the operation of the CNC machine
## Only check the rows where the machining process is during process
def remove_inaccurate_rows(processed_df):
    # Define the conditions for inaccurate rows
    conditions = (
        (processed_df["M1_CURRENT_FEEDRATE"] == 50) &
        (processed_df["X1_ActualPosition"] == 198) &
        (processed_df["M1_CURRENT_PROGRAM_NUMBER"] != 0)
    )

    removed_idx = processed_df.index[conditions]
    cleaned_df = processed_df.loc[~conditions].copy()  
    return cleaned_df, removed_idx

def select_model_features(cleaned_df):
    feature_df = cleaned_df.copy()

    # Derived features: control errors
    feature_df["X1_Position_Error"] = feature_df["X1_ActualPosition"] - feature_df["X1_CommandPosition"]
    feature_df["Y1_Position_Error"] = feature_df["Y1_ActualPosition"] - feature_df["Y1_CommandPosition"]
    feature_df["Z1_Position_Error"] = feature_df["Z1_ActualPosition"] - feature_df["Z1_CommandPosition"]
    feature_df["S1_Position_Error"] = feature_df["S1_ActualPosition"] - feature_df["S1_CommandPosition"]

    feature_df["X1_Velocity_Error"] = feature_df["X1_ActualVelocity"] - feature_df["X1_CommandVelocity"]
    feature_df["Y1_Velocity_Error"] = feature_df["Y1_ActualVelocity"] - feature_df["Y1_CommandVelocity"]
    feature_df["Z1_Velocity_Error"] = feature_df["Z1_ActualVelocity"] - feature_df["Z1_CommandVelocity"]
    feature_df["S1_Velocity_Error"] = feature_df["S1_ActualVelocity"] - feature_df["S1_CommandVelocity"]

    feature_df["X1_Acceleration_Error"] = feature_df["X1_ActualAcceleration"] - feature_df["X1_CommandAcceleration"]
    feature_df["Y1_Acceleration_Error"] = feature_df["Y1_ActualAcceleration"] - feature_df["Y1_CommandAcceleration"]
    feature_df["Z1_Acceleration_Error"] = feature_df["Z1_ActualAcceleration"] - feature_df["Z1_CommandAcceleration"]
    feature_df["S1_Acceleration_Error"] = feature_df["S1_ActualAcceleration"] - feature_df["S1_CommandAcceleration"]

    # Metadata columns
    meta_cols = ["experiment_id", "Machining_Process"]

    # Selected sensor features
    selected_features = [
        # Actual velocity
        "X1_ActualVelocity", "Y1_ActualVelocity", "Z1_ActualVelocity", "S1_ActualVelocity",

        # Actual acceleration
        "X1_ActualAcceleration", "Y1_ActualAcceleration", "Z1_ActualAcceleration", "S1_ActualAcceleration",

        # Current
        "X1_CurrentFeedback", "Y1_CurrentFeedback", "Z1_CurrentFeedback", "S1_CurrentFeedback",
        "X1_OutputCurrent", "Y1_OutputCurrent", "Z1_OutputCurrent", "S1_OutputCurrent",

        # Power
        "X1_OutputPower", "Y1_OutputPower", "S1_OutputPower",

        # Feedrate
        "M1_CURRENT_FEEDRATE",

        # Error features
        "X1_Position_Error", "Y1_Position_Error", "Z1_Position_Error", "S1_Position_Error",
        "X1_Velocity_Error", "Y1_Velocity_Error", "Z1_Velocity_Error", "S1_Velocity_Error",
        "X1_Acceleration_Error", "Y1_Acceleration_Error", "Z1_Acceleration_Error", "S1_Acceleration_Error"
    ]

    keep_cols = meta_cols + selected_features
    feature_df = feature_df[keep_cols].copy()

    return feature_df

## Cutting the data into windows for future use
def cut_into_windows(feature_df, window_size=20):
    windows = []
    global_window_id = 1

    # 50% overlap
    step = window_size // 2

    # Split by Machining_Process first
    for process_name, process_df in feature_df.groupby("Machining_Process"):
        process_df = process_df.reset_index(drop=True)
        total_rows = len(process_df)

        # Case 1: this whole process segment is shorter than one full window
        # Keep it only if its length is at least 50% of window_size
        if total_rows < window_size:
            if total_rows >= window_size // 2:
                window = process_df.copy()
                window.insert(0, "window_id", global_window_id)
                window.insert(1, "window_size", window_size)
                window.insert(2, "window_length", len(window))
                windows.append(window)
                global_window_id += 1
            continue

        # Case 2: normal overlapping full windows
        starts = list(range(0, total_rows - window_size + 1, step))

        # Check remaining rows after the last full window
        last_start = starts[-1] + window_size if starts else 0
        remainder = total_rows - last_start

        # If the remainder is at least 50% of the window size,
        # keep one last window aligned to the end
        if remainder >= window_size // 2:
            candidate_start = total_rows - window_size
            if candidate_start not in starts:
                starts.append(candidate_start)

        # Build windows
        for start in starts:
            window = process_df.iloc[start:start + window_size].copy()
            window.insert(0, "window_id", global_window_id)
            window.insert(1, "window_size", window_size)
            window.insert(2, "window_length", len(window))
            windows.append(window)
            global_window_id += 1

    return windows

## Extracting features for each window for future use
## mean, standard deviation, root mean square value, and maximum and minimum values
def extract_window_features(windows):
    feature_rows = []

    for window in windows:
        feature_dict = {}

        # Keep metadata for each window
        feature_dict["experiment_id"] = window["experiment_id"].iloc[0]
        feature_dict["window_id"] = window["window_id"].iloc[0]
        feature_dict["window_length"] = window["window_length"].iloc[0]
        feature_dict["Machining_Process"] = window["Machining_Process"].iloc[0]

        # Do not calculate statistics for metadata columns
        exclude_cols = [
            "experiment_id",
            "window_id",
            "window_length",
            "Machining_Process"
        ]

        for col in window.columns:
            if col in exclude_cols:
                continue

            values = window[col]

            feature_dict[f"{col}_mean"] = values.mean()
            feature_dict[f"{col}_abs_diff_mean"] = np.mean(np.abs(np.diff(values)))
            feature_dict[f"{col}_std"] = values.std()
            feature_dict[f"{col}_max"] = values.max()
            feature_dict[f"{col}_min"] = values.min()
            feature_dict[f"{col}_range"] = values.max() - values.min()
            feature_dict[f"{col}_rms"] = np.sqrt((values ** 2).mean())

        feature_rows.append(feature_dict)

    return pd.DataFrame(feature_rows)

## Doing the standardization for future use
def standardize_features(window_features_df):
    meta_cols = [
        "experiment_id",
        "window_id",
        "window_length",
        "Machining_Process"
    ]

    # Only keep numeric features for scaling
    aim = window_features_df.drop(columns=meta_cols)

    scaler = StandardScaler()
    aim_scaled = scaler.fit_transform(aim)

    aim_scaled_df = pd.DataFrame(aim_scaled, columns=aim.columns)

    # Add metadata back to the scaled DataFrame
    aim_scaled_df.insert(0, "Machining_Process", window_features_df["Machining_Process"].reset_index(drop=True))
    aim_scaled_df.insert(0, "window_length", window_features_df["window_length"].reset_index(drop=True))
    aim_scaled_df.insert(0, "window_id", window_features_df["window_id"].reset_index(drop=True))
    aim_scaled_df.insert(0, "experiment_id", window_features_df["experiment_id"].reset_index(drop=True))

    return aim_scaled_df

## ========== end of data processing functions ========== ##


## ========== first level clustering functions ========== ##

## ========== K-means clustering for each machining process separately ========== ##

## Train a K-means clustering model
def train_kmeans(aim_scaled_df, use_pca=True, n_components=0.9):
    meta_cols = [
        "experiment_id",
        "window_id",
        "window_length",
        "Machining_Process"
    ]
    
    X = aim_scaled_df.drop(columns=meta_cols).copy()

    if use_pca:
        pca = PCA(n_components=n_components)
        X_model = pca.fit_transform(X)
    else:
        X_model = X.values

    best_k = None
    best_silhouette = -1
    metrics_rows = []

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X_model)

        sil = silhouette_score(X_model, labels)
        db = davies_bouldin_score(X_model, labels)
        ch = calinski_harabasz_score(X_model, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_size_dict = dict(zip(unique, counts))

        metrics_rows.append({
            "k": k,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
            "calinski_harabasz_score": ch,
            "cluster_sizes": str(cluster_size_dict)
        })

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    final_labels = final_kmeans.fit_predict(X_model)

    result_df = aim_scaled_df.copy()
    result_df.insert(0, "cluster", final_labels)

    metrics_df = pd.DataFrame(metrics_rows)
    
    return result_df, best_k, metrics_df

## Train K-means clustering models for each machining process separately and save the results
def run_kmeans_by_process(aim_scaled_df, window_size):
    all_results = []
    summary_rows = []
    all_metrics = []

    for process in aim_scaled_df["Machining_Process"].unique():
        subset = aim_scaled_df[
            aim_scaled_df["Machining_Process"] == process
        ].reset_index(drop=True)

        if len(subset) < 10:
            continue

        result_df, best_k, metrics_df = train_kmeans(subset)

        all_results.append(result_df)
        summary_rows.append({
            "window_size": window_size,
            "Machining_Process": process,
            "best_k": best_k,
            "num_windows": len(subset)
        })

        metrics_with_info = metrics_df.copy()
        metrics_with_info.insert(0, "window_size", window_size) 
        metrics_with_info.insert(1, "Machining_Process", process)
        metrics_with_info.insert(2, "num_windows", len(subset))
        all_metrics.append(metrics_with_info)

    if all_results:
        final_result_df = pd.concat(all_results, ignore_index=True)
    else:
        final_result_df = pd.DataFrame(columns=["cluster"] + aim_scaled_df.columns.tolist())

    summary_df = pd.DataFrame(summary_rows)

    if all_metrics:
        metrics_all_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_all_df = pd.DataFrame(
            columns=[
                "window_size",
                "Machining_Process",
                "num_windows",
                "k",
                "silhouette_score",
                "davies_bouldin_score",
                "calinski_harabasz_score",
                "cluster_sizes",
            ]
        )

    return final_result_df, summary_df, metrics_all_df

## ========== end of K-means clustering for each machining process separately ========== ##

## ========== Gaussian Mixture Model clustering for each machining process separately ========== ##   

## Train a Gaussian Mixture Model
def train_gmm(aim_scaled_df, use_pca=True, n_components=0.9):
    meta_cols = [
        "experiment_id",
        "window_id",
        "window_length",
        "Machining_Process"
    ]
    
    X = aim_scaled_df.drop(columns=meta_cols).copy()

    if use_pca:
        pca = PCA(n_components=n_components)
        X_model = pca.fit_transform(X)
    else:
        X_model = X.values

    best_k = None
    best_bic = np.inf
    metrics_rows = []

    for k in range(2, 10):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42
        )
        labels = gmm.fit_predict(X_model)

        unique_labels = np.unique(labels)

        if len(unique_labels) >= 2:
            sil = silhouette_score(X_model, labels)
            db = davies_bouldin_score(X_model, labels)
            ch = calinski_harabasz_score(X_model, labels)
        else:
            sil = np.nan
            db = np.nan
            ch = np.nan

        bic = gmm.bic(X_model)
        aic = gmm.aic(X_model)

        unique, counts = np.unique(labels, return_counts=True)
        cluster_size_dict = dict(zip(unique, counts))

        metrics_rows.append({
            "k": k,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
            "calinski_harabasz_score": ch,
            "bic": bic,
            "aic": aic,
            "cluster_sizes": str(cluster_size_dict)
        })

        # GMM selects the best k based on the lowest BIC score, which balances model fit and complexity
        if bic < best_bic:
            best_bic = bic
            best_k = k

    if best_k is None:
        result_df = aim_scaled_df.copy()
        result_df.insert(0, "cluster", -1)
        metrics_df = pd.DataFrame(metrics_rows)
        return result_df, None, metrics_df

    final_gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=42
    )
    final_labels = final_gmm.fit_predict(X_model)

    result_df = aim_scaled_df.copy()
    result_df.insert(0, "cluster", final_labels)

    metrics_df = pd.DataFrame(metrics_rows)

    return result_df, best_k, best_bic, metrics_df

## Train Gaussian Mixture Model clustering models for each machining process separately and save the results
def run_gmm_by_process(aim_scaled_df, window_size):
    all_results = []
    summary_rows = []
    all_metrics = []

    for process in aim_scaled_df["Machining_Process"].unique():
        subset = aim_scaled_df[
            aim_scaled_df["Machining_Process"] == process
        ].reset_index(drop=True)

        if len(subset) < 10:
            continue

        result_df, best_k, best_bic, metrics_df = train_gmm(subset)

        all_results.append(result_df)
        summary_rows.append({
            "window_size": window_size,
            "Machining_Process": process,
            "best_k": best_k,
            "best_bic": best_bic,
            "num_windows": len(subset)
        })

        metrics_with_info = metrics_df.copy()
        metrics_with_info.insert(0, "window_size", window_size)
        metrics_with_info.insert(1, "Machining_Process", process)
        metrics_with_info.insert(2, "num_windows", len(subset))
        all_metrics.append(metrics_with_info)

    if all_results:
        final_result_df = pd.concat(all_results, ignore_index=True)
    else:
        final_result_df = pd.DataFrame(columns=["cluster"] + aim_scaled_df.columns.tolist())

    summary_df = pd.DataFrame(summary_rows)

    if all_metrics:
        metrics_all_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_all_df = pd.DataFrame(
            columns=[
                "window_size",
                "Machining_Process",
                "num_windows",
                "k",
                "silhouette_score",
                "davies_bouldin_score",
                "calinski_harabasz_score",
                "bic",
                "aic",
                "cluster_sizes",
            ]
        )

    return final_result_df, summary_df, metrics_all_df

## ========== end of Gaussian Mixture Model clustering for each machining process separately ========== ##   

## ========== SOM clustering for each machining process separately ========== ## 

## Train a Self-Organizing Map with topology selection
def train_som(aim_scaled_df, topology_candidates=None, use_pca=True, n_components=0.9, num_iteration=1000):
    meta_cols = [
        "experiment_id",
        "window_id",
        "window_length",
        "Machining_Process"
    ]

    if topology_candidates is None:
        topology_candidates = [(2, 1), (3, 1), (4, 1), (2, 2)]

    X = aim_scaled_df.drop(columns=meta_cols).copy()

    if use_pca:
        pca = PCA(n_components=n_components)
        X_model = pca.fit_transform(X)
    else:
        X_model = X.values

    best_sil = -np.inf
    best_result_df = None
    best_som_x = None
    best_som_y = None
    best_labels = None
    metrics_rows = []

    for som_x, som_y in topology_candidates:
        som = MiniSom(
            x=som_x,
            y=som_y,
            input_len=X_model.shape[1],
            sigma=1.0,
            learning_rate=0.5,
            random_seed=42
        )

        som.random_weights_init(X_model)
        som.train_random(X_model, num_iteration)

        labels = []
        for x in X_model:
            winner = som.winner(x)   # (i, j)
            cluster_id = winner[0] * som_y + winner[1]
            labels.append(cluster_id)

        labels = np.array(labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) >= 2:
            sil = silhouette_score(X_model, labels)
            db = davies_bouldin_score(X_model, labels)
            ch = calinski_harabasz_score(X_model, labels)
        else:
            sil = np.nan
            db = np.nan
            ch = np.nan

        unique, counts = np.unique(labels, return_counts=True)
        cluster_size_dict = dict(zip(unique, counts))

        metrics_rows.append({
            "som_x": som_x,
            "som_y": som_y,
            "num_neurons": som_x * som_y,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
            "calinski_harabasz_score": ch,
            "cluster_sizes": str(cluster_size_dict)
        })

        if not np.isnan(sil) and sil > best_sil:
            best_sil = sil
            best_som_x = som_x
            best_som_y = som_y
            best_labels = labels.copy()

    metrics_df = pd.DataFrame(metrics_rows)

    if best_labels is None:
        result_df = aim_scaled_df.copy()
        result_df.insert(0, "cluster", -1)
        return result_df, None, None, metrics_df

    best_result_df = aim_scaled_df.copy()
    best_result_df.insert(0, "cluster", best_labels)

    return best_result_df, best_som_x, best_som_y, metrics_df


## Train SOM models for each machining process separately and save the results
def run_som_by_process( aim_scaled_df, window_size, topology_candidates=None, use_pca=True, n_components=0.9, num_iteration=1000):

    all_results = []
    summary_rows = []
    all_metrics = []

    for process in aim_scaled_df["Machining_Process"].unique():
        subset = aim_scaled_df[
            aim_scaled_df["Machining_Process"] == process
        ].reset_index(drop=True)

        if len(subset) < 10:
            continue

        result_df, best_som_x, best_som_y, metrics_df = train_som(
            subset,
            topology_candidates=topology_candidates,
            use_pca=use_pca,
            n_components=n_components,
            num_iteration=num_iteration
        )

        all_results.append(result_df)

        summary_rows.append({
            "window_size": window_size,
            "Machining_Process": process,
            "best_som_x": best_som_x,
            "best_som_y": best_som_y,
            "best_num_neurons": None if best_som_x is None or best_som_y is None else best_som_x * best_som_y,
            "num_windows": len(subset)
        })

        metrics_with_info = metrics_df.copy()
        metrics_with_info.insert(0, "window_size", window_size)
        metrics_with_info.insert(1, "Machining_Process", process)
        metrics_with_info.insert(2, "num_windows", len(subset))
        all_metrics.append(metrics_with_info)

    if all_results:
        final_result_df = pd.concat(all_results, ignore_index=True)
    else:
        final_result_df = pd.DataFrame(columns=["cluster"] + aim_scaled_df.columns.tolist())

    summary_df = pd.DataFrame(summary_rows)

    if all_metrics:
        metrics_all_df = pd.concat(all_metrics, ignore_index=True)
    else:
        metrics_all_df = pd.DataFrame(
            columns=[
                "window_size",
                "Machining_Process",
                "num_windows",
                "som_x",
                "som_y",
                "num_neurons",
                "silhouette_score",
                "davies_bouldin_score",
                "calinski_harabasz_score",
                "cluster_sizes",
            ]
        )

    return final_result_df, summary_df, metrics_all_df

## ========== end of SOM clustering for each machining process separately ========== ## 

## ========== end of first level clustering functions ========== ##


## ========= second level clustering functions ========== ##

## Second level clustering preprocessing: only keep the cluster labels and metadata for each window, and add a new column for the first level cluster label
def second_level_data_selected(w10_final_df, w20_final_df, w30_final_df):
    
    meta_cols = [
        "experiment_id",
        "Machining_Process"
    ]

    w10_preprocessed_df = w10_final_df[meta_cols + ["cluster"]].copy()
    w20_preprocessed_df = w20_final_df[meta_cols + ["cluster"]].copy()
    w30_preprocessed_df = w30_final_df[meta_cols + ["cluster"]].copy()

    return w10_preprocessed_df, w20_preprocessed_df, w30_preprocessed_df

## This function computes the distribution of clusters within each experiment and machining process group for the second level clustering
def build_second_level_distribution(preprocessed_df):
    # First, compute the ratio of each cluster
    # within every experiment + machining process group.
    distribution_df = (
        preprocessed_df.groupby(["experiment_id", "Machining_Process"])["cluster"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )

    cluster_cols = [col for col in distribution_df.columns if col not in ["experiment_id", "Machining_Process"]]
    distribution_df = distribution_df.rename(columns={col: f"cluster_{col}_ratio" for col in cluster_cols})

    return distribution_df

## change distribution data into experiment level data
def build_experiment_vectors(distribution_df):
    ratio_cols = [col for col in distribution_df.columns if col not in ["experiment_id", "Machining_Process"]]

    experiment_df = distribution_df.pivot(
        index="experiment_id",
        columns="Machining_Process",
        values=ratio_cols
    )

    experiment_df = experiment_df.fillna(0)

    experiment_df.columns = [
        f"{process}_{ratio_col}"
        for ratio_col, process in experiment_df.columns
    ]

    experiment_df = experiment_df.reset_index()

    # Drop columns that are all zeros across all experiments
    zero_cols = [
        col for col in experiment_df.columns
        if col != "experiment_id" and (experiment_df[col] == 0).all()
    ]
    experiment_df = experiment_df.drop(columns=zero_cols)

    return experiment_df

## This function runs KMeans clustering on the second level data and adds the cluster labels back to the DataFrame for future use
def cluster_experiments_second_level(experiment_df, n_clusters=3):
    X = experiment_df.drop(columns=["experiment_id"]).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    final_result_df = pd.DataFrame({
        "experiment_id": experiment_df["experiment_id"].values,
        "behavior_cluster": labels,
    })

    return final_result_df

## ========= end of second level clustering functions ========== ##


## ============== verification functions ============== ##

def verify_data_preprocessing(experiment_result, window_size, model_name):
    train_df = load_data(r"D:\python\NU\MLproject\origin data\train.csv")

    train_df["experiment_id"] = train_df["No"].apply(lambda x: f"experiment_{int(x):02d}")
    
    label_df = train_df[["experiment_id", "machining_finalized", "passed_visual_inspection"]].copy()

    merged = experiment_result.merge(label_df, on="experiment_id", how="left")

    # Treat missing labels as "no" for downstream verification charts.
    merged["machining_finalized"] = merged["machining_finalized"].fillna("no")
    merged["passed_visual_inspection"] = merged["passed_visual_inspection"].fillna("no")
   
    save_dir = Path(rf"D:\python\NU\MLproject\verify\{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Cluster vs whether the machining process is completed
    ct_finished = pd.crosstab(
        merged["behavior_cluster"],
        merged["machining_finalized"],
        normalize="index"
    )

    plt.figure()
    sns.heatmap(ct_finished, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Cluster vs Machining Completed")
    plt.savefig(save_dir / f"window{window_size}_{model_name}_cluster_vs_completed.png")
    plt.close()

    # cluster vs whether the product passed visual inspection (look at all data, not just finished ones)
    ct_all_quality = pd.crosstab(
        merged["behavior_cluster"],
        merged["passed_visual_inspection"],
        normalize="index"
    )

    plt.figure()
    sns.heatmap(ct_all_quality, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Cluster vs Inspection (All Data)")
    plt.savefig(save_dir / f"window{window_size}_{model_name}_cluster_vs_inspection_all.png")
    plt.close()

    # cluster vs whether the product passed visual inspection (only look at finished ones)
    finished = merged[merged["machining_finalized"] == "yes"]

    ct_finished_quality = pd.crosstab(
        finished["behavior_cluster"],
        finished["passed_visual_inspection"],
        normalize="index"
    )

    plt.figure()
    sns.heatmap(ct_finished_quality, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Cluster vs Inspection (Finished Only)")
    plt.savefig(save_dir / f"window{window_size}_{model_name}_cluster_vs_inspection_finished.png")
    plt.close()
    
    return merged

# compare the results of the verification for different window sizes and summarize the findings in a report
def compare_three_windows(w10_merged, w20_merged, w30_merged, model_name):
    save_dir = Path(rf"D:\python\NU\MLproject\verify\{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Completed heatmaps side by side 
    ct10 = pd.crosstab(
        w10_merged["behavior_cluster"],
        w10_merged["machining_finalized"],
        normalize="index"
    )
    ct20 = pd.crosstab(
        w20_merged["behavior_cluster"],
        w20_merged["machining_finalized"],
        normalize="index"
    )
    ct30 = pd.crosstab(
        w30_merged["behavior_cluster"],
        w30_merged["machining_finalized"],
        normalize="index"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(ct10, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title("Window 10")
    axes[0].set_ylabel("Behavior Cluster")
    axes[0].set_xlabel("Machining Completed")

    sns.heatmap(ct20, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Window 20")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Machining Completed")

    sns.heatmap(ct30, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2])
    axes[2].set_title("Window 30")
    axes[2].set_ylabel("")
    axes[2].set_xlabel("Machining Completed")

    plt.suptitle("Behavior Cluster vs Machining Completed")
    plt.tight_layout()
    plt.savefig(save_dir / f"compare_windows_cluster_vs_completed_{model_name}.png")
    plt.close()

    # 2. Inspection heatmaps side by side (all data) 
    ct10_q = pd.crosstab(
        w10_merged["behavior_cluster"],
        w10_merged["passed_visual_inspection"],
        normalize="index"
    )
    ct20_q = pd.crosstab(
        w20_merged["behavior_cluster"],
        w20_merged["passed_visual_inspection"],
        normalize="index"
    )
    ct30_q = pd.crosstab(
        w30_merged["behavior_cluster"],
        w30_merged["passed_visual_inspection"],
        normalize="index"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.heatmap(ct10_q, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title("Window 10")
    axes[0].set_ylabel("Behavior Cluster")
    axes[0].set_xlabel("Inspection Result")

    sns.heatmap(ct20_q, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Window 20")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("Inspection Result")

    sns.heatmap(ct30_q, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2])
    axes[2].set_title("Window 30")
    axes[2].set_ylabel("")
    axes[2].set_xlabel("Inspection Result")

    plt.suptitle("Behavior Cluster vs Inspection Result (All Data)")
    plt.tight_layout()
    plt.savefig(save_dir / f"compare_windows_cluster_vs_inspection_all_{model_name}.png")
    plt.close()

# Using NMI to compare the consistency of the clustering results with the zctual quality
def compute_nmi_scores(experiment_k_result, window_size, model_name):
    train_df = load_data(r"D:\python\NU\MLproject\origin data\train.csv")

    train_df["experiment_id"] = train_df["No"].apply(lambda x: f"experiment_{int(x):02d}")
    label_df = train_df[["experiment_id", "machining_finalized", "passed_visual_inspection", "tool_condition"]].copy()

    merged = experiment_k_result.merge(label_df, on="experiment_id", how="left")

    save_dir = Path(rf"D:\python\NU\MLproject\verify\{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # NMI for machining finalized
    nmi_finished_df = merged.dropna(subset=["machining_finalized", "behavior_cluster"]).copy()
    nmi_finalized = normalized_mutual_info_score(
        nmi_finished_df["machining_finalized"],
        nmi_finished_df["behavior_cluster"]
    )

    # NMI for passed visual inspection (finished data only)
    nmi_inspection_df = merged[
        (merged["machining_finalized"] == "yes") &
        (merged["passed_visual_inspection"].notna()) &
        (merged["behavior_cluster"].notna())
    ].copy()

    nmi_inspection_finished = normalized_mutual_info_score(
        nmi_inspection_df["passed_visual_inspection"],
        nmi_inspection_df["behavior_cluster"]
    )

    nmi_tool_df = merged[
        (merged["machining_finalized"] == "yes") &
        (merged["tool_condition"].notna()) &
        (merged["behavior_cluster"].notna())
    ].copy()

    nmi_tool = normalized_mutual_info_score(
        nmi_tool_df["tool_condition"],
        nmi_tool_df["behavior_cluster"]
    )

    nmi_df = pd.DataFrame({
        "window_size": [window_size],

        "nmi_finalized": [nmi_finalized],
        "nmi_inspection_finished_only": [nmi_inspection_finished],
        "nmi_tool_condition": [nmi_tool],

        "n_finalized_samples": [len(nmi_finished_df)],
        "n_inspection_samples": [len(nmi_inspection_df)],
        "n_tool_samples": [len(nmi_tool_df)]
    })

    return nmi_df

## ============= end of verification functions ============== ##

## ============== comparison functions ============== ##

## compare three different window sizes and find the one that can best capture the differences in machining behavior patterns across different experiments and different machining processes.
def select_best_window(nmi_df, model_name):
    result = nmi_df.copy()

    result["final_score"] = (
        0.6 * result["nmi_finalized"] +
        0.2 * result["nmi_inspection_finished_only"] +
        0.2 * result["nmi_tool_condition"]
    )

    result["model_name"] = model_name

    best_row = result.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True).iloc[0]

    return result, best_row

def compare_best_models(kmeans_nmi_df, gmm_nmi_df, som_nmi_df):
    kmeans_scored, kmeans_best = select_best_window(kmeans_nmi_df, "kmeans")
    gmm_scored, gmm_best = select_best_window(gmm_nmi_df, "gmm")
    som_scored, som_best = select_best_window(som_nmi_df, "som")

    best_models_df = pd.DataFrame([
        kmeans_best,
        gmm_best,
        som_best
    ])

    best_models_df = best_models_df.sort_values(
        by="final_score",
        ascending=False
    ).reset_index(drop=True)

    overall_best = best_models_df.iloc[0]

    return kmeans_scored, gmm_scored, som_scored, best_models_df, overall_best

## ============== end of comparison functions ============== ##

## ============== pipeline functions ============== ##

## This function runs the entire data preprocessing pipeline
def data_preprocessing_pipeline():
    input_files = get_input_files()

    summary_rows = []   # List to store summary information about removed rows for each file
    all_windows = {10: [], 20: [], 30: []}  # Dictionary to store windows of different sizes

    for file_path in input_files:
        # save the original data for later use
        df = load_data(file_path)

        # make sure the feature names are correct
        check_feature_names(df)

        # leave processing rows for future use
        processed_df = leave_processing(df)

        # remove inaccurate rows for future use
        cleaned_df, removed_idx = remove_inaccurate_rows(processed_df)

        # Make sure to keep track of the source file and row indices for the cleaned data
        cleaned_df = cleaned_df.copy()
        cleaned_df.insert(0, "experiment_id", file_path.stem)  

        # select the features for future use
        feature_df = select_model_features(cleaned_df)

        # Cut the cleaned data into windows for future use
        w10 = cut_into_windows(feature_df, window_size=10)
        w20 = cut_into_windows(feature_df, window_size=20)
        w30 = cut_into_windows(feature_df, window_size=30)

        if w10:
            all_windows[10].extend(w10)
        if w20:
            all_windows[20].extend(w20)
        if w30:
            all_windows[30].extend(w30)

        # Add summary information about the removed rows for the current file to the summary list
        summary_rows.append({
            "experiment_id": file_path.stem,
            "original_rows": len(df),
            "Not_processing_rows": len(df) - len(processed_df),
            "processed_rows": len(processed_df),
            "removed_rows": len(removed_idx),
            "cleaned_rows": len(cleaned_df),
            "w10_windows": len(w10),
            "w20_windows": len(w20),
            "w30_windows": len(w30),    
        })

    after_processing_dir = Path(r"D:\python\NU\MLproject\after process data")        
    after_processing_dir.mkdir(parents=True, exist_ok=True)

    # Create a summary DataFrame from the summary information collected for each file
    summary_df = pd.DataFrame(summary_rows).sort_values("experiment_id").reset_index(drop=True)
    summary_df.to_csv(after_processing_dir / "experiment_summary.csv", index=False)

    # Extract features for each window for future use
    w10_features = extract_window_features(all_windows[10])
    w20_features = extract_window_features(all_windows[20])
    w30_features = extract_window_features(all_windows[30])

    # Standardize the extracted features for each window size
    w10_standardized = standardize_features(w10_features)
    w20_standardized = standardize_features(w20_features)
    w30_standardized = standardize_features(w30_features)

    # Save the extracted features for each window size to separate CSV files
    w10_standardized.to_csv(after_processing_dir / "all_experiments_window10_features_standardized.csv", index=False)
    w20_standardized.to_csv(after_processing_dir / "all_experiments_window20_features_standardized.csv", index=False)
    w30_standardized.to_csv(after_processing_dir / "all_experiments_window30_features_standardized.csv", index=False)

    return w10_standardized, w20_standardized, w30_standardized

## ====================== KMeans pipeline functions ================== === ##

## This function runs the entire KMeans clustering pipeline for all window sizes and saves the results
def kmeans_clustering_pipeline(w10_standardized, w20_standardized, w30_standardized):
    kmean_dir = Path(r"D:\python\NU\MLproject\kmeans")
    kmean_dir.mkdir(parents=True, exist_ok=True)

    # Run KMeans separately for each Machining_Process within each window size
    w10_k_final_df, w10_summary, w10_metrics = run_kmeans_by_process(w10_standardized, 10)
    w20_k_final_df, w20_summary, w20_metrics = run_kmeans_by_process(w20_standardized, 20)
    w30_k_final_df, w30_summary, w30_metrics = run_kmeans_by_process(w30_standardized, 30)

    # Save the final clustering results for each window size to separate CSV files
    w10_k_final_df.to_csv(kmean_dir / "all_experiments_window10_kmeans_results.csv", index=False)
    w20_k_final_df.to_csv(kmean_dir / "all_experiments_window20_kmeans_results.csv", index=False)
    w30_k_final_df.to_csv(kmean_dir / "all_experiments_window30_kmeans_results.csv", index=False)

    # Save per-k metrics for each window size
    w10_metrics.to_csv(kmean_dir / "window10_kmeans_metrics_by_process.csv", index=False)
    w20_metrics.to_csv(kmean_dir / "window20_kmeans_metrics_by_process.csv", index=False)
    w30_metrics.to_csv(kmean_dir / "window30_kmeans_metrics_by_process.csv", index=False)

    # Save merged metrics for all window sizes
    all_metrics_df = pd.concat([w10_metrics, w20_metrics, w30_metrics], ignore_index=True)
    all_metrics_df.to_csv(kmean_dir / "all_windows_kmeans_metrics_by_process.csv", index=False)

    # Collect summary DataFrames for each window size into a list for merging
    summary_dfs = []

    summary_dfs.append(w10_summary)
    summary_dfs.append(w20_summary)
    summary_dfs.append(w30_summary)

    # Merge all summaries into one file
    final_summary_df = pd.concat(summary_dfs, ignore_index=True)
    final_summary_df = final_summary_df.sort_values(
        ["window_size", "Machining_Process"]
    ).reset_index(drop=True)

    final_summary_df.to_csv(kmean_dir / "classification_kmeans_summary_by_process.csv", index=False)

    return w10_k_final_df, w20_k_final_df, w30_k_final_df

## This function runs the entire second level clustering pipeline based on the results of the first level KMeans clustering and saves the results
def kmeans_second_level_clustering_pipeline(w10_k_final_df, w20_k_final_df, w30_k_final_df):
    # Save the preprocessed data for second level clustering to separate CSV files
    second_level_dir = Path(r"D:\python\NU\MLproject\second_level_clustering\kmeans")
    second_level_dir.mkdir(parents=True, exist_ok=True)

    # second level clustering preprocessing: only keep the cluster labels and metadata for each window, and add a new column for the first level cluster label
    w10_preprocessed_df, w20_preprocessed_df, w30_preprocessed_df = second_level_data_selected(w10_k_final_df, w20_k_final_df, w30_k_final_df)

    second_level_w10_distribution = build_second_level_distribution(w10_preprocessed_df)
    second_level_w20_distribution = build_second_level_distribution(w20_preprocessed_df)
    second_level_w30_distribution = build_second_level_distribution(w30_preprocessed_df)

    experiment_vectors_w10 = build_experiment_vectors(second_level_w10_distribution)
    experiment_vectors_w20 = build_experiment_vectors(second_level_w20_distribution)
    experiment_vectors_w30 = build_experiment_vectors(second_level_w30_distribution)

    experiment_vectors_w10.to_csv(second_level_dir / "window10_experiment_vectors.csv", index=False)
    experiment_vectors_w20.to_csv(second_level_dir / "window20_experiment_vectors.csv", index=False)
    experiment_vectors_w30.to_csv(second_level_dir / "window30_experiment_vectors.csv", index=False)

    # second level clustering: cluster the experiments based on the distribution of first level clusters within each experiment
    final_w10_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w10, n_clusters=3)
    final_w20_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w20, n_clusters=3)
    final_w30_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w30, n_clusters=3)

    final_w10_experiment_clusters.to_csv(second_level_dir / "window10_experiment_clusters.csv", index=False)
    final_w20_experiment_clusters.to_csv(second_level_dir / "window20_experiment_clusters.csv", index=False)
    final_w30_experiment_clusters.to_csv(second_level_dir / "window30_experiment_clusters.csv", index=False)

    return final_w10_experiment_clusters, final_w20_experiment_clusters, final_w30_experiment_clusters

# This function runs the entire verification pipeline to check the relationship between the second level clusters and the quality of the products
def kmeans_verification_pipeline(w10_experiment_k_result, w20_experiment_k_result, w30_experiment_k_result):
    w10_merged = verify_data_preprocessing(w10_experiment_k_result, 10, "kmeans")
    w20_merged = verify_data_preprocessing(w20_experiment_k_result, 20, "kmeans")
    w30_merged = verify_data_preprocessing(w30_experiment_k_result, 30, "kmeans")

    compare_three_windows(w10_merged, w20_merged, w30_merged, "kmeans")

    # Compute NMI scores for each window
    w10_nmi = compute_nmi_scores(w10_experiment_k_result, 10, "kmeans")
    w20_nmi = compute_nmi_scores(w20_experiment_k_result, 20, "kmeans")
    w30_nmi = compute_nmi_scores(w30_experiment_k_result, 30, "kmeans")

    nmi_df = pd.concat([w10_nmi, w20_nmi, w30_nmi], ignore_index=True)
    nmi_df.to_csv(r"D:\python\NU\MLproject\verify\kmeans\kmeans_nmi.csv", index=False)

    return nmi_df

## ====================== end of KMeans pipeline functions ================== === ##

## ====================== Gaussian pipeline functions ================== === ##

## This function runs the entire Gaussian clustering pipeline for all window sizes and saves the results
def gaussian_clustering_pipeline(w10_standardized, w20_standardized, w30_standardized):
    gaussian_dir = Path(r"D:\python\NU\MLproject\gaussian")
    gaussian_dir.mkdir(parents=True, exist_ok=True)

    # Run Gaussian clustering separately for each Machining_Process within each window size
    w10_g_final_df, w10_summary, w10_metrics = run_gmm_by_process(w10_standardized, 10)
    w20_g_final_df, w20_summary, w20_metrics = run_gmm_by_process(w20_standardized, 20)
    w30_g_final_df, w30_summary, w30_metrics = run_gmm_by_process(w30_standardized, 30)

    # Save the final clustering results for each window size to separate CSV files
    w10_g_final_df.to_csv(gaussian_dir / "all_experiments_window10_gaussian_results.csv", index=False)
    w20_g_final_df.to_csv(gaussian_dir / "all_experiments_window20_gaussian_results.csv", index=False)
    w30_g_final_df.to_csv(gaussian_dir / "all_experiments_window30_gaussian_results.csv", index=False)

    # Save per-k metrics for each window size
    w10_metrics.to_csv(gaussian_dir / "window10_gaussian_metrics_by_process.csv", index=False)
    w20_metrics.to_csv(gaussian_dir / "window20_gaussian_metrics_by_process.csv", index=False)
    w30_metrics.to_csv(gaussian_dir / "window30_gaussian_metrics_by_process.csv", index=False)

    # Save merged metrics for all window sizes
    all_metrics_df = pd.concat([w10_metrics, w20_metrics, w30_metrics], ignore_index=True)
    all_metrics_df.to_csv(gaussian_dir / "all_windows_gaussian_metrics_by_process.csv", index=False)

    # Collect summary DataFrames for each window size into a list for merging
    summary_dfs = []

    summary_dfs.append(w10_summary)
    summary_dfs.append(w20_summary)
    summary_dfs.append(w30_summary)

    # Merge all summaries into one file
    final_summary_df = pd.concat(summary_dfs, ignore_index=True)
    final_summary_df = final_summary_df.sort_values(
        ["window_size", "Machining_Process"]
    ).reset_index(drop=True)

    final_summary_df.to_csv(gaussian_dir / "classification_gaussian_summary_by_process.csv", index=False)

    return w10_g_final_df, w20_g_final_df, w30_g_final_df

## This function runs the entire second level clustering pipeline based on the results of the first level Gaussian clustering and saves the results
def gaussian_second_level_clustering_pipeline(w10_g_final_df, w20_g_final_df, w30_g_final_df):
    # Save the preprocessed data for second level clustering to separate CSV files
    second_level_dir = Path(r"D:\python\NU\MLproject\second_level_clustering\gaussian")
    second_level_dir.mkdir(parents=True, exist_ok=True)

    # second level clustering preprocessing: only keep the cluster labels and metadata for each window, and add a new column for the first level cluster label
    w10_preprocessed_df, w20_preprocessed_df, w30_preprocessed_df = second_level_data_selected(w10_g_final_df, w20_g_final_df, w30_g_final_df)

    second_level_w10_distribution = build_second_level_distribution(w10_preprocessed_df)
    second_level_w20_distribution = build_second_level_distribution(w20_preprocessed_df)
    second_level_w30_distribution = build_second_level_distribution(w30_preprocessed_df)

    experiment_vectors_w10 = build_experiment_vectors(second_level_w10_distribution)
    experiment_vectors_w20 = build_experiment_vectors(second_level_w20_distribution)
    experiment_vectors_w30 = build_experiment_vectors(second_level_w30_distribution)

    experiment_vectors_w10.to_csv(second_level_dir / "window10_experiment_vectors.csv", index=False)
    experiment_vectors_w20.to_csv(second_level_dir / "window20_experiment_vectors.csv", index=False)
    experiment_vectors_w30.to_csv(second_level_dir / "window30_experiment_vectors.csv", index=False)

    # second level clustering: cluster the experiments based on the distribution of first level clusters within each experiment
    final_w10_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w10, n_clusters=3)
    final_w20_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w20, n_clusters=3)
    final_w30_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w30, n_clusters=3)

    final_w10_experiment_clusters.to_csv(second_level_dir / "window10_experiment_clusters.csv", index=False)
    final_w20_experiment_clusters.to_csv(second_level_dir / "window20_experiment_clusters.csv", index=False)
    final_w30_experiment_clusters.to_csv(second_level_dir / "window30_experiment_clusters.csv", index=False)

    return final_w10_experiment_clusters, final_w20_experiment_clusters, final_w30_experiment_clusters

# This function runs the entire verification pipeline to check the relationship between the second level clusters and the quality of the products
def gmm_verification_pipeline(w10_experiment_g_result, w20_experiment_g_result, w30_experiment_g_result):
    w10_merged = verify_data_preprocessing(w10_experiment_g_result, 10, "gaussian")
    w20_merged = verify_data_preprocessing(w20_experiment_g_result, 20, "gaussian")
    w30_merged = verify_data_preprocessing(w30_experiment_g_result, 30, "gaussian")

    compare_three_windows(w10_merged, w20_merged, w30_merged, "gaussian")

    # Compute NMI scores for each window
    w10_nmi = compute_nmi_scores(w10_experiment_g_result, 10, "gaussian")
    w20_nmi = compute_nmi_scores(w20_experiment_g_result, 20, "gaussian")
    w30_nmi = compute_nmi_scores(w30_experiment_g_result, 30, "gaussian")

    nmi_df = pd.concat([w10_nmi, w20_nmi, w30_nmi], ignore_index=True)
    nmi_df.to_csv(r"D:\python\NU\MLproject\verify\gaussian\gaussian_nmi.csv", index=False)

    return nmi_df

## ====================== end of Gaussian pipeline functions ================== === ##

## ====================== SOM pipeline functions ================== === ##

## This function runs the entire SOM clustering pipeline for all window sizes and saves the results
def som_clustering_pipeline(w10_standardized, w20_standardized, w30_standardized):
    som_dir = Path(r"D:\python\NU\MLproject\som")
    som_dir.mkdir(parents=True, exist_ok=True)

    # Run SOM clustering separately for each Machining_Process within each window size
    w10_s_final_df, w10_summary, w10_metrics = run_som_by_process(w10_standardized, 10)
    w20_s_final_df, w20_summary, w20_metrics = run_som_by_process(w20_standardized, 20)
    w30_s_final_df, w30_summary, w30_metrics = run_som_by_process(w30_standardized, 30)

    # Save the final clustering results for each window size to separate CSV files
    w10_s_final_df.to_csv(som_dir / "all_experiments_window10_som_results.csv", index=False)
    w20_s_final_df.to_csv(som_dir / "all_experiments_window20_som_results.csv", index=False)
    w30_s_final_df.to_csv(som_dir / "all_experiments_window30_som_results.csv", index=False)

    # Save clustering metrics for each window size
    w10_metrics.to_csv(som_dir / "window10_som_metrics_by_process.csv", index=False)
    w20_metrics.to_csv(som_dir / "window20_som_metrics_by_process.csv", index=False)
    w30_metrics.to_csv(som_dir / "window30_som_metrics_by_process.csv", index=False)

    # Save merged metrics for all window sizes
    all_metrics_df = pd.concat([w10_metrics, w20_metrics, w30_metrics], ignore_index=True)
    all_metrics_df.to_csv(som_dir / "all_windows_som_metrics_by_process.csv", index=False)

    # Collect summary DataFrames for each window size into a list for merging
    summary_dfs = []

    summary_dfs.append(w10_summary)
    summary_dfs.append(w20_summary)
    summary_dfs.append(w30_summary)

    # Merge all summaries into one file
    final_summary_df = pd.concat(summary_dfs, ignore_index=True)
    final_summary_df = final_summary_df.sort_values(
        ["window_size", "Machining_Process"]
    ).reset_index(drop=True)

    final_summary_df.to_csv(som_dir / "classification_som_summary_by_process.csv", index=False)

    return w10_s_final_df, w20_s_final_df, w30_s_final_df

## This function runs the entire second level clustering pipeline based on the results of the first level SOM clustering and saves the results
def som_second_level_clustering_pipeline(w10_s_final_df, w20_s_final_df, w30_s_final_df):
    # Save the preprocessed data for second level clustering to separate CSV files
    second_level_dir = Path(r"D:\python\NU\MLproject\second_level_clustering\som")
    second_level_dir.mkdir(parents=True, exist_ok=True)

    # second level clustering preprocessing: only keep the cluster labels and metadata for each window, and add a new column for the first level cluster label
    w10_preprocessed_df, w20_preprocessed_df, w30_preprocessed_df = second_level_data_selected(w10_s_final_df, w20_s_final_df, w30_s_final_df)

    second_level_w10_distribution = build_second_level_distribution(w10_preprocessed_df)
    second_level_w20_distribution = build_second_level_distribution(w20_preprocessed_df)
    second_level_w30_distribution = build_second_level_distribution(w30_preprocessed_df)

    experiment_vectors_w10 = build_experiment_vectors(second_level_w10_distribution)
    experiment_vectors_w20 = build_experiment_vectors(second_level_w20_distribution)
    experiment_vectors_w30 = build_experiment_vectors(second_level_w30_distribution)

    experiment_vectors_w10.to_csv(second_level_dir / "window10_experiment_vectors.csv", index=False)
    experiment_vectors_w20.to_csv(second_level_dir / "window20_experiment_vectors.csv", index=False)
    experiment_vectors_w30.to_csv(second_level_dir / "window30_experiment_vectors.csv", index=False)

    # second level clustering: cluster the experiments based on the distribution of first level clusters within each experiment
    final_w10_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w10, n_clusters=3)
    final_w20_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w20, n_clusters=3)
    final_w30_experiment_clusters = cluster_experiments_second_level(experiment_vectors_w30, n_clusters=3)

    final_w10_experiment_clusters.to_csv(second_level_dir / "window10_experiment_clusters.csv", index=False)
    final_w20_experiment_clusters.to_csv(second_level_dir / "window20_experiment_clusters.csv", index=False)
    final_w30_experiment_clusters.to_csv(second_level_dir / "window30_experiment_clusters.csv", index=False)

    return final_w10_experiment_clusters, final_w20_experiment_clusters, final_w30_experiment_clusters

# This function runs the entire verification pipeline to check the relationship between the second level clusters and the quality of the products
def som_verification_pipeline(w10_experiment_s_result, w20_experiment_s_result, w30_experiment_s_result):
    w10_merged = verify_data_preprocessing(w10_experiment_s_result, 10, "som")
    w20_merged = verify_data_preprocessing(w20_experiment_s_result, 20, "som")
    w30_merged = verify_data_preprocessing(w30_experiment_s_result, 30, "som")

    compare_three_windows(w10_merged, w20_merged, w30_merged, "som")

    # Compute NMI scores for each window
    w10_nmi = compute_nmi_scores(w10_experiment_s_result, 10, "som")
    w20_nmi = compute_nmi_scores(w20_experiment_s_result, 20, "som")
    w30_nmi = compute_nmi_scores(w30_experiment_s_result, 30, "som")

    nmi_df = pd.concat([w10_nmi, w20_nmi, w30_nmi], ignore_index=True)
    nmi_df.to_csv(r"D:\python\NU\MLproject\verify\som\som_nmi.csv", index=False)

    return nmi_df

## ====================== end of SOM pipeline functions ================== === ##

## ====================== comparison pipeline functions ================== === ##
# Compare models
def compare_models_pipeline(kmeans_nmi_df, gmm_nmi_df, som_nmi_df):
    kmeans_scored, gmm_scored, som_scored, best_models_df, overall_best = compare_best_models(kmeans_nmi_df, gmm_nmi_df, som_nmi_df)
    
    all_results_df = pd.concat([kmeans_scored, gmm_scored, som_scored],ignore_index=True).copy()
    all_results_df["record_type"] = "all_result"

    best_models_to_save = best_models_df.copy()
    best_models_to_save["record_type"] = "best_within_model"

    overall_best_to_save = overall_best.copy()
    overall_best_to_save["record_type"] = "overall_best"

    final_output_df = pd.concat(
        [all_results_df, best_models_to_save, overall_best_to_save],
        ignore_index=True
    )

    save_dir = Path(r"D:\python\NU\MLproject\final_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)

    final_output_df.to_csv(save_dir / "all_models_with_best_summary.csv", index=False)

    print("Best model:")
    print(overall_best)

    return 
    
## ====================== end of comparison pipeline functions ================== === ##

## ========== end of pipeline functions ========== ##


## ============= main program ============== ##

## main program
if __name__ == "__main__":
    w10_standardized, w20_standardized, w30_standardized = data_preprocessing_pipeline()

    w10_k_final_df, w20_k_final_df, w30_k_final_df = kmeans_clustering_pipeline(w10_standardized, w20_standardized, w30_standardized)
    w10_experiment_k_result, w20_experiment_k_result, w30_experiment_k_result = kmeans_second_level_clustering_pipeline(w10_k_final_df, w20_k_final_df, w30_k_final_df)
    kmeans_nmi_df = kmeans_verification_pipeline(w10_experiment_k_result, w20_experiment_k_result, w30_experiment_k_result)
    
    w10_g_final_df, w20_g_final_df, w30_g_final_df = gaussian_clustering_pipeline(w10_standardized, w20_standardized, w30_standardized)
    w10_experiment_g_result, w20_experiment_g_result, w30_experiment_g_result = gaussian_second_level_clustering_pipeline(w10_g_final_df, w20_g_final_df, w30_g_final_df)
    gmm_nmi_df = gmm_verification_pipeline(w10_experiment_g_result, w20_experiment_g_result, w30_experiment_g_result)

    w10_s_final_df, w20_s_final_df, w30_s_final_df = som_clustering_pipeline(w10_standardized, w20_standardized, w30_standardized)
    w10_experiment_s_result, w20_experiment_s_result, w30_experiment_s_result = som_second_level_clustering_pipeline(w10_s_final_df, w20_s_final_df, w30_s_final_df)
    som_nmi_df = som_verification_pipeline(w10_experiment_s_result, w20_experiment_s_result, w30_experiment_s_result)

    compare_models_pipeline(kmeans_nmi_df, gmm_nmi_df, som_nmi_df)
    
## ========== end of main program ========== ##