import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "/home/dominic/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla
from carla_simulation.dataset import Dataset
from shapely import speedups

if speedups.available:
    speedups.enable()
    print("Speedups enabled")

from pathlib import Path
import yaml

# create a world

# first define a client
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)  # seconds

town = "Town01"
world = client.load_world(town)
settings = world.get_settings()
settings.synchronous_mode = True  # True
# TODO
settings.fixed_delta_seconds = 0.1

world.apply_settings(settings)

spectator = world.get_spectator()


dataset = Dataset(world)
# TODO: setup folder structure
BASEPATH = "/home/dominic/data/carla"
base = Path(BASEPATH)

root = base / town
root.mkdir(parents=True, exist_ok=True)

steps = 10
for i in range(steps):
    sample_dir = root / ("sample_%s" % i)
    sample_dir.mkdir(exist_ok=True)
    # generate one sample from every tick
    world.tick()

    sample = dataset.get_sample()

    sample_dict = {}
    sample_dict["ego_pose"] = dataset.ego_pose
    for name, lidar in dataset.lidars.items():
        export_dict = {}
        export_file = lidar.exportPCD(sample_dir)
        export_dict["data"] = str(export_file.relative_to(root))
        export_dict["extrinsic"] = lidar.M
        sample_dict["sensors"][lidar.id] = export_dict

    for name, cam in dataset.cameras.items():
        export_dict = {}
        export_file = lidar.write_data(sample_dir)
        export_dict["data"] = str(export_file.relative_to(root))
        export_dict["extrinsic"] = lidar.M
        export_dict["intrinsic"] = lidar.K
        sample_dict["sensors"][cam.id] = export_dict

    sample_file = sample_dir / "sample.pkl"
    with sample_file.open("w+") as f:
        yaml.safe_dump(sample_dict)
