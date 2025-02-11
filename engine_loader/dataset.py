from torch.utils.data import Dataset
from torch import tensor, Tensor
from pathlib import Path
import polars as pl
import h5py as h5


class EngineDataset(Dataset):
    def __init__(self, parquet_path: Path, h5_path: Path):
        self.dataframe = pl.read_parquet(parquet_path)
        self.data_file = h5.File(h5_path, "r")

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
                row["energy"][-1],
                row["polar"][-1],
                row["azimuthal"][-1],
                row["vertex_x"][-1],
                row["vertex_y"][-1],
                row["vertex_z"][-1],
            ]
        )
        return data, result
