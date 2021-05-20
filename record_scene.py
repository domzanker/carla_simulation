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
from dataset import Dataset
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
step_delta = 0.05
duration = 120

world = client.load_world(town)

world.set_weather(carla.WeatherParameters.ClearNoon)

settings = world.get_settings()
world.apply_settings(settings)
