from attpc_engine.detector import (
    DetectorParams,
    ElectronicsParams,
    PadParams,
    Config,
)

from attpc_engine import nuclear_map
from spyral_utils.nuclear.target import GasTarget, load_target
from pathlib import Path
from engine_loader.create import RangeParameter, create_dataset, DatasetParameters
from numpy import pi

# h5_path = Path("proton_dataset.h5")
# df_path = Path("proton_dataset.parquet")
h5_path = Path("proton_test_dataset.h5")
df_path = Path("proton_test_dataset.parquet")


gas = load_target(Path("./deuterium_gas.json"), nuclear_map)
if not isinstance(gas, GasTarget):
    raise Exception("Could not load gas target!")

dset_params = DatasetParameters(
    n_samples=10_000,
    z=1,
    a=1,
    energy=RangeParameter(0.1, 10.0),
    polar=RangeParameter(0.0, pi),
    azimuthal=RangeParameter(0.0, 2.0 * pi),
    vertex_x=RangeParameter(-25.0, 25.0),
    vertex_y=RangeParameter(-25.0, 25.0),
    vertex_z=RangeParameter(0.0, 1000.0),
)

detector = DetectorParams(
    length=1.0,
    efield=45000.0,
    bfield=2.85,
    mpgd_gain=175000,
    gas_target=gas,
    diffusion=0.277,
    fano_factor=0.2,
    w_value=34.0,
)

electronics = ElectronicsParams(
    clock_freq=6.25,
    amp_gain=900,
    shaping_time=1000,
    micromegas_edge=10,
    windows_edge=560,
    adc_threshold=10,
)

pads = PadParams()

config = Config(detector, electronics, pads)


def main():
    create_dataset(dset_params, config, df_path, h5_path)


if __name__ == "__main__":
    main()
