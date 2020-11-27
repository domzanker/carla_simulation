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
from map_bridge import MapBridge
from sensor_platform import SensorPlatform

from queue import Queue
import numpy as np
import cv2
import shapely
from math import ceil, floor, degrees

from copy import deepcopy, copy

from city_scapes_cm import apply_cityscapes_cm
from dataset_utilities.transformation import Isometry

from scipy.spatial.transform import Rotation


def isometry_to_carla(isometry: Isometry):
    location = carla.Location(
        x=isometry.translation[0], y=isometry.translation[1], z=isometry.translation[2]
    )
    rot = isometry.rotation.unit
    rotation = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w])
    # y, z, x = rotation.as_euler("YZX", degrees=True)
    y, z, x = rotation.as_euler("yzx", degrees=True)
    print([x, y, z])
    rotation = carla.Rotation(-y, z, -x)
    return carla.Transform(location, rotation)


class Dataset:
    def __init__(self, world):
        self.cameras = {}
        self.camera_queues = {}

        camera_setup = {
            "top_view": {
                "extrinsic": {
                    "location": carla.Location(),
                    "rotation": carla.Rotation(pitch=-90, yaw=0),
                },
                "intrinsic": {"image_size_x": 1000, "image_size_y": 2000},
            }
        }

        self.sensor_platform = SensorPlatform(world)

        cam, queue = self.sensor_platform.add_camera(
            "top_view",
            transform=carla.Transform(**camera_setup["top_view"]["extrinsic"]),
            blueprint="sensor.camera.semantic_segmentation",
            **camera_setup["top_view"]["intrinsic"],
        )
        self.cameras["top_view"] = cam
        self.camera_queues["top_view"] = queue

        self.map_bridge = MapBridge(world)
        self.map_bridge.load_lane_polygons()

        self.ego_pose = carla.Transform(carla.Location(z=2), carla.Rotation())

    def get_sample(self):
        # get images
        top_view_data = self.camera_queues["top_view"].get()
        cc = carla.ColorConverter.CityScapesPalette
        top_view_data.save_to_disk("semantic_test.png", cc)
        np_img = np.frombuffer(top_view_data.raw_data, dtype=np.uint8)
        np_img = np.reshape(np_img, (top_view_data.height, top_view_data.width, 4))
        np_img = np_img[:, :, :3]
        np_img = np_img[:, :, ::-1].copy()
        # np_img = (255 * np_img).astype(np.uint8)
        np_img = apply_cityscapes_cm(np_img)

        bev = self.cameras["top_view"].transform(data=np_img)
        # get ego_pose from sensor pose
        # world_T_sensor -> world_T_veh * veh_T_sensor
        sensor_T_veh = self.cameras["top_view"].extrinsic
        world_T_sensor = Isometry.from_carla_transform(top_view_data.transform)
        world_T_veh = world_T_sensor @ sensor_T_veh
        self.ego_pose = world_T_veh

        # trafo = self.sensor_platform.ego_vehicle.get_transform()

        roi = [40, 20]
        bb, poly = self.map_bridge.get_map_patch(roi, self.ego_pose)

        # transform polygon to image frame
        veh_T_world = self.ego_pose.inverse()
        coefficient_list = np.ravel(veh_T_world.matrix[:3, :3]).tolist()
        coefficient_list += np.ravel(veh_T_world.matrix[:3, 3]).tolist()

        # road polygons in vehicle pose
        veh_poly = shapely.affinity.affine_transform(poly, coefficient_list)

        # we need the polygons in image scope of the roi (ie image frame)
        resolution = 0.02
        coeffs = [
            0,
            1 / resolution,
            -1 / resolution,
            0,
            top_view_data.width / 2,
            top_view_data.height / 2,
        ]
        img_poly = shapely.affinity.affine_transform(veh_poly, coeffs)
        if img_poly.geom_type == "MultiPolygon":
            img_poly = shapely.geometry.MultiPolygon(
                [
                    shapely.geometry.Polygon(
                        [(floor(x), floor(y)) for x, y in geom.exterior.coords]
                    )
                    for geom in img_poly
                ]
            )
        else:
            img_poly = shapely.geometry.asPolygon(
                [(floor(x), floor(y)) for x, y in img_poly.exterior.coords]
            )

        return np_img, bb, poly, bev, img_poly, veh_poly


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map_bridge import plot_polygon
    from shapely import speedups
    from pyquaternion import Quaternion

    """

    print("test identity... ", end="")
    i = Isometry()
    c = isometry_to_carla(i)
    np.testing.assert_allclose(i.matrix, c.get_matrix())
    print("passed")

    print("test roll... ", end="")
    i = Isometry(rotation=Quaternion(axis=[1, 0, 0], degrees=45))
    c = isometry_to_carla(i)
    np.testing.assert_allclose(i.matrix, c.get_matrix())
    print("passed")

    print("test pitch... ", end="")
    i = Isometry(rotation=Quaternion(axis=[0, 1, 0], degrees=45))
    c = isometry_to_carla(i)
    np.testing.assert_allclose(i.matrix, c.get_matrix())
    print("passed")

    print("test yaw... ", end="")
    i = Isometry(rotation=Quaternion(axis=[0, 0, 1], degrees=45))
    c = isometry_to_carla(i)
    np.testing.assert_allclose(i.matrix, c.get_matrix())
    print("passed")

    print("test vector... ", end="")
    i = Isometry(rotation=Quaternion(axis=[0, 1, 1], degrees=45))
    c = isometry_to_carla(i)
    v = np.array([4.4, 9.0, 10.2, 1])
    pi = i @ v[:3]
    pc = c.get_matrix() @ v
    np.testing.assert_allclose(pi, pc)
    print("passed")

    for i in range(5):
        print("test random #%s... " % i, end="")
        i = Isometry.random()
        c = isometry_to_carla(i)
        np.testing.assert_allclose(i.matrix, c.get_matrix(), atol=1e-1)
        print("passed")
    """
    if speedups.available:
        speedups.enable()
        print("Speedups enabled")

    # first define a client
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)  # seconds

    world = client.load_world("Town03")
    settings = world.get_settings()
    settings.synchronous_mode = False  # True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    dataset = Dataset(world)

    spectator = world.get_spectator()

    frames = 100
    for frame in range(frames):

        world.tick()

        image, bb, p, bev, ip, vp = dataset.get_sample()
        spectator.set_transform(
            carla.Transform(
                carla.Location(*dataset.ego_pose.translation), carla.Rotation()
            )
        )

        fig, ax = plt.subplots(1, 4)
        (x_min, y_min, x_max, y_max) = dataset.map_bridge.lane_polyons.bounds
        margin = 2
        ax[2].set_xlim([x_min - margin, x_max + margin])
        ax[2].set_ylim([y_min - margin, y_max + margin])

        ax[0].imshow(image)
        ax[1].imshow(image)
        dataset.map_bridge.plot_polys(ax[2])
        plot_polygon(ax[1], ip, fc="blue", ec="black", alpha=0.4)

        plot_polygon(ax[2], bb, fc="blue", ec="blue", alpha=0.5)
        # (x_min, y_min, x_max, y_max) = vp.bounds
        x_min = -20
        x_max = 20
        y_min = -20
        y_max = 20
        margin = 2
        ax[3].set_xlim([x_min - margin, x_max + margin])
        ax[3].set_ylim([y_min - margin, y_max + margin])
        plot_polygon(ax[3], vp)
        ax[3].plot([0], [0], marker="o", markersize=3)

        plt.show()
