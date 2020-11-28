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
from dataset_utilities.camera import Camera, BirdsEyeView

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

        self.roi = [30, 20]
        self.resolution = 0.04

        self.sensor_platform = SensorPlatform(world)

        top_view = {
            "top_view": {
                "extrinsic": {
                    "location": carla.Location(),
                    "rotation": carla.Rotation(pitch=-90, yaw=0),
                },
                "intrinsic": {},
            }
        }

        cam, queue = self.sensor_platform.add_topview(
            "top_view",
            veh_T_sensor=carla.Transform(**top_view["top_view"]["extrinsic"]),
            blueprint="sensor.camera.semantic_segmentation",
            roi=self.roi,
            resolution=self.resolution,
            **top_view["top_view"]["intrinsic"],
        )
        self.cameras["top_view"] = cam
        self.camera_queues["top_view"] = queue

        camera_setup = {
            "cam_front": {
                "extrinsic": {
                    "location": carla.Location(x=2.0, y=0.0, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=0),
                },
                "intrinsic": {"image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_front_left": {
                "extrinsic": {
                    "location": carla.Location(x=1.5, y=-0.5, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=-45),
                },
                "intrinsic": {"image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_front_right": {
                "extrinsic": {
                    "location": carla.Location(x=1.5, y=0.5, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=45),
                },
                "intrinsic": {"image_size_x": 1920, "image_size_y": 1080},
            },
        }
        # "cam_front_left": {},
        # "cam_front_right": {},
        # "cam_back": {},
        # "cam_back_left": {},
        # "cam_back_right": {},
        for cam, configs in camera_setup.items():
            bev, q_ = self.sensor_platform.add_camera(
                name=cam,
                veh_T_sensor=carla.Transform(**configs["extrinsic"]),
                **configs["intrinsic"],
            )
            self.cameras[cam] = bev
            self.camera_queues[cam] = q_

        self.map_bridge = MapBridge(world)
        self.map_bridge.load_lane_polygons()

        self.ego_pose = carla.Transform(carla.Location(z=2), carla.Rotation())

    def get_sample(self):
        # get images
        for name, cam in self.cameras.items():
            if name == "top_view":
                cam_data = self.camera_queues["top_view"].get()
                cc = carla.ColorConverter.CityScapesPalette
                cam_data.save_to_disk("semantic_test.png", cc)
                np_img = np.frombuffer(
                    cam_data.raw_data, dtype=np.uint8
                ).copy()  # .copy()
                np_img = np.reshape(np_img, (cam_data.height, cam_data.width, 4))
                np_img = np_img[:, :, :3]
                np_img = np_img[:, :, ::-1]
                np_img = apply_cityscapes_cm(np_img)
                top_view_img = np_img

            else:
                cam_data = self.camera_queues[name].get()
                cam_data.save_to_disk(name + "_test.png")
                np_img = np.frombuffer(
                    cam_data.raw_data, dtype=np.uint8
                ).copy()  # .copy()
                np_img = np.reshape(np_img, (cam_data.height, cam_data.width, 4))
                np_img = np_img[:, :, :3]
                np_img = np_img[:, :, ::-1]
                self.cameras[name].load_data(data=np_img)

            # bev = self.cameras["top_view"].transform(data=np_img)
            # get ego_pose from sensor pose
            # world_T_sensor -> world_T_veh * veh_T_sensor
            sensor_T_veh = self.cameras[name].extrinsic
            world_T_sensor = Isometry.from_carla_transform(cam_data.transform)
            world_T_veh = world_T_sensor @ sensor_T_veh
            self.ego_pose = world_T_veh

        bb, poly = self.map_bridge.get_map_patch(self.roi, self.ego_pose)
        # transform polygon to image frame
        veh_T_world = self.ego_pose.inverse()
        coefficient_list = np.ravel(veh_T_world.matrix[:3, :3]).tolist()
        coefficient_list += np.ravel(veh_T_world.matrix[:3, 3]).tolist()
        # road polygons in vehicle pose
        veh_poly = shapely.affinity.affine_transform(poly, coefficient_list)

        # we need the polygons in image scope of the roi (ie image frame)
        """
        img_poly = shapely.affinity.rotate(veh_poly, angle=90, origin=(0, 0))
        img_poly = shapely.affinity.translate(
            img_poly, xoff=self.roi[0] / 2, yoff=self.roi[1] / 2
        )
        img_poly = shapely.affinity.scale(
            img_poly,
            xfact=1 / self.resolution,
            yfact=1 / self.resolution,
            origin=(0, 0),
        )

        """
        coeffs = [
            0,
            -1 / self.resolution,
            -1 / self.resolution,
            0,
            self.roi[1] / (2 * self.resolution),
            self.roi[0] / (2 * self.resolution),
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

        return {
            "image": top_view_img,
            "query_box": bb,
            "world_poly": poly,
            "image_poly": img_poly,
            "vehicle_poly": veh_poly,
        }

    def polylines_from_shapely(self, shapes):
        raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map_bridge import plot_polygon
    from shapely import speedups

    if speedups.available:
        speedups.enable()
        print("Speedups enabled")

    # first define a client
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)  # seconds

    world = client.load_world("Town01")
    settings = world.get_settings()
    settings.synchronous_mode = False  # True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    dataset = Dataset(world)

    spectator = world.get_spectator()

    frames = 100
    for frame in range(frames):

        world.tick()

        sample = dataset.get_sample()
        spectator.set_transform(
            carla.Transform(
                carla.Location(
                    dataset.ego_pose.translation[0],
                    -dataset.ego_pose.translation[1],
                    dataset.ego_pose.translation[2],
                ),
                carla.Rotation(pitch=-30),
            )
        )

        fig, ax = plt.subplots(2, 4)
        (x_min, y_min, x_max, y_max) = dataset.map_bridge.lane_polyons.bounds
        margin = 2
        ax[0][2].set_xlim([x_min - margin, x_max + margin])
        ax[0][2].set_ylim([y_min - margin, y_max + margin])

        ax[0][0].imshow(sample["image"])
        ax[0][1].imshow(sample["image"])
        dataset.map_bridge.plot_polys(ax[0][2])
        plot_polygon(ax[0][1], sample["image_poly"], fc="blue", ec="black", alpha=0.4)

        plot_polygon(ax[0][2], sample["query_box"], fc="blue", ec="blue", alpha=0.5)
        ax[0][2].set_aspect("equal")
        # (x_min, y_min, x_max, y_max) = vp.bounds
        x_max = dataset.roi[0] // 2
        x_min = -x_max
        y_max = dataset.roi[1] // 2
        y_min = -y_max
        margin = 2
        ax[0][3].set_xlim([x_min - margin, x_max + margin])
        ax[0][3].set_ylim([y_min - margin, y_max + margin])
        plot_polygon(ax[0][3], sample["vehicle_poly"])
        ax[0][3].plot([0], [0], marker="o", markersize=3)
        ax[0][3].set_aspect("equal")

        ax[1][1].imshow(dataset.cameras["cam_front"].data)
        ax[1][0].imshow(dataset.cameras["cam_front_left"].data)
        ax[1][2].imshow(dataset.cameras["cam_front_right"].data)

        plt.show()
