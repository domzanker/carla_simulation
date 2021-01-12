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
import numpy as np

from dataset_utilities.grid_map import GridMap
from dataset_utilities.camera import Lidar
from dataset_utilities.transformation import Isometry


def test_update():
    import matplotlib.pyplot as plt

    grid = GridMap(cell_size=0.1)

    veh_T_lidar = carla.Transform(carla.Location(0, 0, 2), carla.Rotation())
    veh_T_lidar = Isometry.from_carla_transform(veh_T_lidar)

    pc = np.concatenate((np.arange(0, 100, 0.5)[None, :], np.zeros([3, 200])))
    for i in range(9):
        print("Update %s" % i)
        loc = carla.Location(0, 0, 0)
        rot = carla.Rotation(roll=0, pitch=0, yaw=10 * i)  # test yaw
        pose = carla.Transform(loc, rot)
        world_T_veh = Isometry.from_carla_transform(pose)

        grid.update_u(pc, veh_T_sensor=veh_T_lidar, world_T_veh=world_T_veh)

    g = grid.render(debug=True)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(g[:, :, :3])
    ax[1].imshow(g[:, :, 6:9])

    plt.show()


if __name__ == "__main__":
    test_update()
