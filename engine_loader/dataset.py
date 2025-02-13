from torch.utils.data import Dataset
from torch import tensor, Tensor
from pathlib import Path
import polars as pl
import h5py as h5
import numpy as np


class EngineDataset(Dataset):
    def __init__(self, parquet_path: Path, h5_path: Path, max_energy: float):
        self.dataframe = pl.read_parquet(parquet_path)
        self.data_file = h5.File(h5_path, "r")
        self.max_energy = max_energy

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        raw_data = self.data_file["samples"][f"sample_{index}"][:].copy()  # type: ignore
        # Scale and shift
        raw_data[:, :] *= 0.001
        raw_data[:, :2] += 0.3

        data = tensor(raw_data)  # type: ignore
        row = self.dataframe.filter(pl.col("sample") == index).to_dict()
        result = tensor(
            [
                row["energy"][-1] / self.max_energy,
                row["polar"][-1] / np.pi,
                row["azimuthal"][-1] / (2.0 * np.pi),
                (row["vertex_x"][-1] + 300.0) * 1.0e-3,
                (row["vertex_y"][-1] + 300.0) * 1.0e-3,
                row["vertex_z"][-1] * 1.0e-3,
            ]
        )
        return data, result


def transform_result_to_real_units(result: np.ndarray, model_max_energy: float):
    result[0] *= model_max_energy
    result[1] *= np.pi
    result[2] *= 2.0 * np.pi
    result[3] -= 0.3
    result[4] -= 0.3
