import logging
import pathlib
import pickle
from datetime import datetime, timedelta

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GRAPH_FILE_PATH = pathlib.Path("data") / "adj_METR-LA.pkl"
DATA_FILE_PATH = pathlib.Path("data") / "METR-LA.h5"


def load_graph() -> tuple[list[str], dict[str, int], np.ndarray]:
    """
    Load the graph adjacency matrix and sensor mapping.

    Returns:
        sensor_ids: list of sensor ID strings (e.g., '773869')
        sensor_id_to_ind: dict mapping sensor ID -> index
        adj_mx: adjacency matrix of shape [207, 207] with float weights
    """
    with open(GRAPH_FILE_PATH, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

    logger.info(f"Graph loaded: {len(sensor_ids)} sensors")
    logger.info(
        f"Adjacency matrix shape: {adj_mx.shape}, sparsity: {(adj_mx == 0).sum() / adj_mx.size * 100:.1f}% zeros"
    )

    return sensor_ids, sensor_id_to_ind, adj_mx


"""
- H5 Data Structure:
- block0_values: The actual traffic speed data - Shape (34272, 207)
    - 34,272 timesteps (5-minute intervals from March-June 2012)
    - 207 sensors (matching the graph)
    - Values are traffic speeds in mph (looks like range ~0-70 mph)
- axis0 / block0_items: Sensor IDs - Shape (207,)
- Matches the sensor IDs from the graph pickle
- axis1: Timestamps - Shape (34272,)
- Unix nanoseconds (1330560000000000000 = March 1, 2012)
"""


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load METR-LA traffic speed data from HDF5 file (pandas DataFrame format).

    Returns:
        data: numpy array of shape [34272, 207, 1] with traffic speeds (mph)
        timestamps: numpy array of datetime strings
        sensor_ids: list of sensor ID strings
    """
    with h5py.File(DATA_FILE_PATH, "r") as f:
        # The HDF5 contains a pandas DataFrame stored in PyTables format
        df_group = f["df"]

        # Extract the actual data
        speeds = df_group["block0_values"][:]  # Shape: [34272, 207]
        sensor_ids = [
            s.decode("utf-8") for s in df_group["axis0"][:]
        ]  # Decode bytes to strings
        timestamp_ns = df_group["axis1"][:]  # Unix nanoseconds

        # Verify we have the right sensor IDs
        logger.info(f"Data shape: {speeds.shape} (timesteps, sensors)")
        logger.info(f"Time range: {len(timestamp_ns)} timesteps")
        logger.info(f"Sensor IDs: {len(sensor_ids)} sensors")

        # Convert nanoseconds to datetime
        # Unix nanoseconds since 1970-01-01
        timestamps = []
        for ns in timestamp_ns[:5]:  # Show first few as example
            dt = datetime(1970, 1, 1) + timedelta(seconds=ns / 1e9)
            timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        logger.info(f"First 5 timestamps: {timestamps}")

        # Convert all nanoseconds to datetime strings
        all_timestamps = []
        for ns in timestamp_ns:
            dt = datetime(1970, 1, 1) + timedelta(seconds=ns / 1e9)
            all_timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        timestamps = np.array(all_timestamps)

    # Reshape to add feature dimension: [timesteps, sensors, features]
    data = speeds.reshape(speeds.shape[0], speeds.shape[1], 1)

    # Basic statistics
    logger.info(f"Data range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}] mph")
    logger.info(f"Mean speed: {np.nanmean(data):.2f} mph")
    logger.info(f"Missing values: {np.isnan(data).sum() / data.size * 100:.2f}%")

    return data, timestamps, sensor_ids


def get_sensor_mapping(
    sensor_ids_from_data: list[str], sensor_ids_from_graph: list[str]
) -> dict[int, int]:
    """
    Create mapping from data indices to graph indices.
    Ensures sensors are aligned correctly.
    """
    # Create mapping from sensor ID to graph index
    id_to_graph_idx = {sid: i for i, sid in enumerate(sensor_ids_from_graph)}

    # Map data column indices to graph indices
    data_to_graph_idx = []
    for i, sid in enumerate(sensor_ids_from_data):
        if sid in id_to_graph_idx:
            data_to_graph_idx.append(id_to_graph_idx[sid])
        else:
            logger.warning(f"Sensor {sid} not found in graph!")

    return np.array(data_to_graph_idx)


def create_windows(data: np.ndarray, input_len: int, output_len: int, stride: int = 1):
    """
    Create sliding windows for forecasting.

    Args:
        data: [timesteps, sensors, features]
        input_len: number of past timesteps
        output_len: number of future timesteps to predict
        stride: step between windows

    Returns:
        X: [num_windows, input_len, sensors, features]
        Y: [num_windows, output_len, sensors, features]
    """
    windows = []
    targets = []

    n = data.shape[0]
    for i in range(0, n - input_len - output_len + 1, stride):
        windows.append(data[i : i + input_len])
        targets.append(data[i + input_len : i + input_len + output_len])

    X = np.array(windows)
    Y = np.array(targets)

    logger.info(f"Created {len(X)} windows: X {X.shape}, Y {Y.shape}")
    return X, Y


def split_data(data: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.1):
    """Chronological split."""
    n = data.shape[0]
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    train = data[:train_idx]
    val = data[train_idx:val_idx]
    test = data[val_idx:]

    logger.info(f"Split: train {train.shape}, val {val.shape}, test {test.shape}")
    return train, val, test


def normalize_data(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    """
    Normalize using training statistics.
    Returns normalized data and the (mean, std) used.
    """
    # Per-sensor statistics (ignore first dimension which is time)
    mean = np.nanmean(train, axis=0, keepdims=True)  # [1, sensors, features]
    std = np.nanstd(train, axis=0, keepdims=True)
    std[std == 0] = 1.0  # Avoid division by zero

    train_norm = (train - mean) / std
    val_norm = (val - mean) / std
    test_norm = (test - mean) / std

    logger.info(
        f"Normalized - train mean: {np.nanmean(train_norm):.3f}, std: {np.nanstd(train_norm):.3f}"
    )

    return train_norm, val_norm, test_norm, mean, std


if __name__ == "__main__":
    # Load graph
    graph_sensor_ids, sensor_id_to_ind, adj_mx = load_graph()

    # Load data
    data, timestamps, data_sensor_ids = load_data()

    # Verify alignment between graph and data sensors
    data_to_graph = get_sensor_mapping(data_sensor_ids, graph_sensor_ids)
    logger.info(f"Data to graph mapping: {len(data_to_graph)} sensors mapped")

    # Split data
    train, val, test = split_data(data)

    # Normalize
    train_norm, val_norm, test_norm, mean, std = normalize_data(train, val, test)

    # Create windows for forecasting
    X_train, Y_train = create_windows(train_norm, input_len=12, output_len=12)
    X_val, Y_val = create_windows(val_norm, input_len=12, output_len=12)
    X_test, Y_test = create_windows(test_norm, input_len=12, output_len=12)

    # Save processed data for later use
    processed_data = {
        "train": train_norm,
        "val": val_norm,
        "test": test_norm,
        "mean": mean,
        "std": std,
        "adj_mx": adj_mx,
        "sensor_ids": graph_sensor_ids,
        "data_to_graph_idx": data_to_graph,
    }

    # Save to numpy file for quick loading later
    np.savez("data/processed_metr_la.npz", **processed_data)
    logger.info("Processed data saved to data/processed_metr_la.npz")
