from pathlib import Path
from attpc_engine.detector.simulator import SimParticle, dict_to_points, NUM_TB
from attpc_engine.detector.parameters import Config
from attpc_engine.detector.writer import convert_to_spyral
from attpc_engine.detector.response import get_response
from attpc_engine import nuclear_map
from dataclasses import dataclass
import numpy as np
from numpy.random import Generator
from numba import types
from numba.typed import Dict
import h5py as h5
import polars as pl
from tqdm import tqdm


@dataclass
class RangeParameter:
    value_min: float
    value_max: float


def sample(param: RangeParameter, rng: Generator) -> float:
    return param.value_min + rng.random() * param.value_max


@dataclass
class DatasetParameters:
    n_samples: int
    z: int
    a: int
    energy: RangeParameter
    polar: RangeParameter
    azimuthal: RangeParameter
    vertex_x: RangeParameter
    vertex_y: RangeParameter
    vertex_z: RangeParameter


def sample_parameters(
    dset_params: DatasetParameters, mass: float, rng: Generator
) -> tuple[np.ndarray, np.ndarray, tuple]:
    energy = sample(dset_params.energy, rng)
    p = energy * np.sqrt(energy + 2.0 * mass)
    polar = sample(dset_params.polar, rng)
    azimuthal = sample(dset_params.azimuthal, rng)
    vertex_x = sample(dset_params.vertex_x, rng)
    vertex_y = sample(dset_params.vertex_y, rng)
    vertex_z = sample(dset_params.vertex_z, rng)

    return (
        np.array(
            [
                p * np.sin(polar) * np.cos(azimuthal),
                p * np.sin(polar) * np.sin(azimuthal),
                p * np.cos(polar),
                energy + mass,
            ]
        ),
        np.array([vertex_x * 0.001, vertex_y * 0.001, vertex_z * 0.001]),
        (energy, polar, azimuthal, vertex_x, vertex_y, vertex_z),
    )


def bin_point_cloud(cloud: np.ndarray) -> np.ndarray:
    binned_data = np.zeros((1000, 3), dtype=np.float32)
    bin_min = 0.0
    bin_max = 0.0
    bin_step = 1.0
    for idx in range(0, 1000):
        bin_min = bin_max
        bin_max = bin_max + bin_step
        points = cloud[np.logical_and(bin_min < cloud[:, 2], cloud[:, 2] <= bin_max)]
        if len(points) == 0:
            continue
        binned_data[idx, :] = np.average(points[:, :3], axis=0)
    return binned_data


def create_point_cloud(
    dset_params: DatasetParameters,
    sim_params: Config,
    rng: Generator,
    mass: float,
    response: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    p_vec, v_pos, sample_vals = sample_parameters(dset_params, mass, rng)
    particle = SimParticle(p_vec, v_pos, dset_params.z, dset_params.a)
    # Simulate...
    points = Dict.empty(key_type=types.int64, value_type=types.int64)
    particle.generate_point_cloud(sim_params, rng, points)
    point_array = dict_to_points(points)
    # Format...
    point_array[:, 1] += rng.uniform(low=0.0, high=1.0, size=len(point_array))
    point_array = point_array[
        np.logical_and(0 <= point_array[:, 1], point_array[:, 1] < NUM_TB)
    ]
    spyral_points = convert_to_spyral(
        point_array,
        sim_params.elec_params.windows_edge,
        sim_params.elec_params.micromegas_edge,
        sim_params.det_params.length,
        response,
        sim_params.pad_centers,  # type: ignore
        sim_params.pad_sizes,
        sim_params.elec_params.adc_threshold,
    )
    # Make sure we're still sorted in z
    spyral_points = spyral_points[np.argsort(spyral_points[:, 2])]
    # Transform...
    binned_data = bin_point_cloud(spyral_points)
    # Make sure we're still sorted in z
    return (binned_data, sample_vals)


def create_dataset(
    dset_params: DatasetParameters,
    sim_params: Config,
    parquet_path: Path,
    h5_path: Path,
):
    h5_file = h5.File(h5_path, "w")
    sample_dict = {
        "sample": np.zeros(dset_params.n_samples, dtype=np.int64),
        "energy": np.zeros(dset_params.n_samples, dtype=np.float32),
        "polar": np.zeros(dset_params.n_samples, dtype=np.float32),
        "azimuthal": np.zeros(dset_params.n_samples, dtype=np.float32),
        "vertex_x": np.zeros(dset_params.n_samples, dtype=np.float32),
        "vertex_y": np.zeros(dset_params.n_samples, dtype=np.float32),
        "vertex_z": np.zeros(dset_params.n_samples, dtype=np.float32),
    }
    sample_group = h5_file.create_group("samples")
    sample_group.attrs["n_samples"] = dset_params.n_samples
    mass = nuclear_map.get_data(dset_params.z, dset_params.a).mass
    rng = np.random.default_rng()
    response: np.ndarray = get_response(sim_params).copy()
    for idx in tqdm(range(dset_params.n_samples)):
        binned_data, sample_vals = create_point_cloud(
            dset_params, sim_params, rng, mass, response
        )
        sample_group.create_dataset(f"sample_{idx}", data=binned_data)
        sample_dict["sample"][idx] = idx
        sample_dict["energy"][idx] = sample_vals[0]
        sample_dict["polar"][idx] = sample_vals[1]
        sample_dict["azimuthal"][idx] = sample_vals[2]
        sample_dict["vertex_x"][idx] = sample_vals[3]
        sample_dict["vertex_y"][idx] = sample_vals[4]
        sample_dict["vertex_z"][idx] = sample_vals[5]
    df = pl.DataFrame(sample_dict)
    df.write_parquet(parquet_path)
