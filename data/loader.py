import logging
import pathlib
import pickle
from datetime import datetime, timedelta
from typing import List, Tuple

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class METRLoader:
    """Load and preprocess METR-LA traffic data."""

    def __init__(self, graph_path: pathlib.Path, data_path: pathlib.Path):
        self.graph_path = graph_path
        self.data_path = data_path
        self.sensor_ids = None
        self.adj_mx = None
        self.data = None
        self.timestamps = None
        self.data_sensor_ids = None

    def load_graph(self) -> Tuple[List[str], np.ndarray]:
        """Load graph adjacency matrix and sensor IDs."""
        with open(self.graph_path, "rb") as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

        self.sensor_ids = sensor_ids
        self.adj_mx = adj_mx

        logger.info(f"Graph loaded: {len(sensor_ids)} sensors")
        logger.info(f"Adjacency matrix shape: {adj_mx.shape}")
        logger.info(f"Sparsity: {(adj_mx == 0).sum() / adj_mx.size * 100:.1f}% zeros")
        logger.info(f"Weight range: [{adj_mx.min():.3f}, {adj_mx.max():.3f}]")

        return sensor_ids, adj_mx

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load traffic speed data from HDF5."""
        with h5py.File(self.data_path, "r") as f:
            # METR-LA is stored as pandas DataFrame in HDF5
            df_group = f["df"]

            # Extract data
            speeds = df_group["block0_values"][:]  # [34272, 207]
            sensor_ids_bytes = df_group["axis0"][:]  # [207]
            timestamp_ns = df_group["axis1"][:]  # [34272]

        # Convert sensor IDs from bytes to strings
        self.data_sensor_ids = [sid.decode("utf-8") for sid in sensor_ids_bytes]

        # Convert nanosecond timestamps to datetime strings
        timestamps = []
        for ns in timestamp_ns:
            dt = datetime(1970, 1, 1) + timedelta(seconds=ns / 1e9)
            timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        self.timestamps = np.array(timestamps)

        # Add feature dimension: [T, N] -> [T, N, 1]
        self.data = speeds.reshape(speeds.shape[0], speeds.shape[1], 1)

        logger.info(f"Data loaded: {self.data.shape} (timesteps, sensors, features)")
        logger.info(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        logger.info(
            f"Speed range: [{np.nanmin(self.data):.1f}, {np.nanmax(self.data):.1f}] mph"
        )
        logger.info(
            f"Missing values: {np.isnan(self.data).sum() / self.data.size * 100:.2f}%"
        )

        return self.data, self.timestamps, self.data_sensor_ids

    def align_sensors(self) -> np.ndarray:
        """
        Create mapping from data indices to graph indices.

        Returns:
            data_to_graph_idx: Array of shape [207] where value at position i
                               gives the corresponding graph index for data sensor i.
        """
        if self.sensor_ids is None or self.data_sensor_ids is None:
            raise ValueError(
                "Load graph and data first using load_graph() and load_data()"
            )

        # Create mapping from sensor ID to graph index
        id_to_graph = {sid: idx for idx, sid in enumerate(self.sensor_ids)}

        # Map each data sensor to its graph index
        data_to_graph = []
        missing_sensors = []

        for i, sid in enumerate(self.data_sensor_ids):
            if sid in id_to_graph:
                data_to_graph.append(id_to_graph[sid])
            else:
                missing_sensors.append(sid)
                data_to_graph.append(-1)  # Placeholder for missing

        data_to_graph = np.array(data_to_graph)

        if missing_sensors:
            logger.warning(f"Found {len(missing_sensors)} sensors in data not in graph")
            logger.warning(f"First few missing: {missing_sensors[:5]}")

        logger.info(
            f"Sensor alignment complete. Valid mappings: {(data_to_graph >= 0).sum()}/{len(data_to_graph)}"
        )

        return data_to_graph

    def prepare_for_modeling(self, data_to_graph_idx: np.ndarray):
        """
        Reorder data sensors to match graph order for consistent graph convolutions.

        Args:
            data_to_graph_idx: Mapping from data indices to graph indices

        Returns:
            reordered_data: Data with sensors ordered according to graph
        """
        # Create inverse mapping: graph index -> data index
        graph_to_data = {}
        for data_idx, graph_idx in enumerate(data_to_graph_idx):
            if graph_idx >= 0:
                graph_to_data[graph_idx] = data_idx

        # Reorder data along sensor dimension
        reordered_data = np.zeros_like(self.data)
        valid_sensors = 0

        for graph_idx in range(len(self.sensor_ids)):
            if graph_idx in graph_to_data:
                data_idx = graph_to_data[graph_idx]
                reordered_data[:, graph_idx, :] = self.data[:, data_idx, :]
                valid_sensors += 1
            else:
                # Fill with NaN for missing sensors (should not happen in METR-LA)
                reordered_data[:, graph_idx, :] = np.nan

        logger.info(
            f"Data reordered: {valid_sensors}/{len(self.sensor_ids)} sensors aligned"
        )

        return reordered_data


if __name__ == "__main__":
    pass
