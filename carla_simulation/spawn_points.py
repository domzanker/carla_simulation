import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
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

import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

from map_bridge import MapBridge

from shapely import speedups
import numpy as np

if speedups.available:
    speedups.enable()
    print("Speedups enabled")

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--map", default="Town01")
parser.add_argument("--save", default=None)

args = parser.parse_args()

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)  # seconds

world = client.load_world(args.map)

map = MapBridge(world)
map.load_lane_polygons()

fig, ax = plt.subplots()

map.plot_polys(ax)

sp = world.get_map().get_spawn_points()
xy = np.asarray([[t.location.x, t.location.y] for t in sp])

colors = np.arange(0, len(sp), dtype=np.float32)
sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap=plt.get_cmap("plasma"), zorder=100)
plt.colorbar(sc)

if args.save is not None:
    path = Path(args.save) / f"{args.map}_spawn_points.png"
    fig.savefig(path)
    print(path)
plt.show()
