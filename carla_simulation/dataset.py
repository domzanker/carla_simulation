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
    def __init__(self, world, vehicle_spawn_point=0, sensor_tick=5):
        self.cameras = {}
        self.camera_queues = {}
        self.sensor_calibrations = {}

        self.roi = [30, 20]
        self.resolution = 0.04

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[vehicle_spawn_point]
        self.sensor_platform = SensorPlatform(world, spawn_point, sensor_tick)

        top_view = {
            "top_view": {
                "extrinsic": {
                    "location": carla.Location(),
                    "rotation": carla.Rotation(pitch=-90, yaw=0),
                },
                "intrinsic": {},
            }
        }

        extrinsic = carla.Transform(**top_view["top_view"]["extrinsic"])
        cam, queue, vTs = self.sensor_platform.add_topview(
            "top_view",
            veh_T_sensor=extrinsic,
            blueprint="sensor.camera.semantic_segmentation",
            roi=self.roi,
            resolution=self.resolution,
            **top_view["top_view"]["intrinsic"],
        )
        self.cameras["top_view"] = cam
        self.camera_queues["top_view"] = queue
        # self.sensor_calibrations["top_view"] = Isometry.from_carla_transform(vTs)
        self.sensor_calibrations["top_view"] = vTs

        camera_setup = {
            "cam_front": {
                "extrinsic": {
                    "location": carla.Location(x=1.0, y=0.0, z=2),
                    "rotation": carla.Rotation(roll=0, pitch=0, yaw=0),
                },
                "intrinsic": {"fov": 110, "image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_front_left": {
                "extrinsic": {
                    "location": carla.Location(x=1.5, y=-0.5, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=-45),
                },
                "intrinsic": {"fov": 110, "image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_front_right": {
                "extrinsic": {
                    "location": carla.Location(x=1.5, y=0.5, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=45),
                },
                "intrinsic": {"fov": 110, "image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_back": {
                "extrinsic": {
                    "location": carla.Location(x=-1.0, y=0.0, z=2),
                    "rotation": carla.Rotation(roll=0, pitch=0, yaw=180),
                },
                "intrinsic": {"fov": 110, "image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_back_left": {
                "extrinsic": {
                    "location": carla.Location(x=-1.5, y=-0.5, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=-135),
                },
                "intrinsic": {"fov": 110, "image_size_x": 1920, "image_size_y": 1080},
            },
            "cam_back_right": {
                "extrinsic": {
                    "location": carla.Location(x=-1.5, y=0.5, z=1.5),
                    "rotation": carla.Rotation(roll=0, pitch=-12, yaw=135),
                },
                "intrinsic": {"fov": 110, "image_size_x": 1920, "image_size_y": 1080},
            },
        }
        # "cam_front_left": {},
        # "cam_front_right": {},
        # "cam_back": {},
        # "cam_back_left": {},
        # "cam_back_right": {},
        for cam, configs in camera_setup.items():
            bev, q_, vTs = self.sensor_platform.add_camera(
                name=cam,
                veh_T_sensor=carla.Transform(**configs["extrinsic"]),
                **configs["intrinsic"],
            )
            self.cameras[cam] = bev
            self.camera_queues[cam] = q_
            self.sensor_calibrations[cam] = Isometry.from_carla_transform(vTs)

        self.lidars = {}
        self.lidar_queues = {}
        lidar_setup = {
            "lidar_top": {
                "extrinsic": {
                    "location": carla.Location(x=0, y=0, z=2),
                    "rotation": carla.Rotation(roll=0, pitch=0, yaw=0),
                },
                "intrinsic": {
                    "rotation_frequency": 20,  #  / sensor_tick,
                    "points_per_second": 90000,
                    "range": 35,
                    "channels": 32,
                    "lower_fov": -15,
                    "upper_fov": 5,
                    "dropoff_general_rate": 0.30,
                },
            }
        }
        for lidar, configs in lidar_setup.items():
            l, q_, veh_T_sensor = self.sensor_platform.add_lidar(
                name=lidar,
                veh_T_sensor=carla.Transform(**configs["extrinsic"]),
                **configs["intrinsic"],
            )
            self.lidar_queues[lidar] = q_
            self.lidars[lidar] = l
            self.sensor_calibrations[lidar] = Isometry.from_carla_transform(
                veh_T_sensor
            )

        self.map_bridge = MapBridge(world)
        self.map_bridge.load_lane_polygons()

        self.ego_pose = spawn_point

    def get_sample(self, frame_id, include_map: bool = True):
        # get images
        """
        for k, i in self.camera_queues.items():
            print(k)
            print(i.qsize())
        for k, i in self.lidar_queues.items():
            print(k)
            print(i.qsize())
        """
        for name, lidar in self.lidars.items():
            lidar_data = self.lidar_queues[name].get()
            # frame_id = lidar_data.frame
            while lidar_data.frame != frame_id:
                if self.lidar_queues[name].empty():
                    print(name + " empty")
                    return False
                lidar_data = self.lidar_queues[name].get()

            point_cloud = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(
                [-1, 4]
            )
            point_cloud = np.row_stack(
                [
                    point_cloud[:, 0],
                    point_cloud[:, 1],
                    point_cloud[:, 2],
                    point_cloud[:, 3],
                ]
            )
            lidar.load_data(data=point_cloud)
            lidar.transformLidarToVehicle()

        imu = self.sensor_platform.ego_pose.get()
        while imu.frame != frame_id:
            if self.sensor_platform.ego_pose.empty():
                print("imu empty")
                return False
            imu = self.sensor_platform.ego_pose.get()
        self.ego_pose = imu.transform

        for name, cam in self.cameras.items():
            cam_data = self.camera_queues[name].get()
            while cam_data.frame != frame_id:
                if self.camera_queues[name].empty():
                    print(name + " empty")
                    return False
                cam_data = self.camera_queues[name].get()
            if name == "top_view":
                cc = carla.ColorConverter.CityScapesPalette
                np_img = np.frombuffer(
                    cam_data.raw_data, dtype=np.uint8
                ).copy()  # .copy()
                np_img = np.reshape(np_img, (cam_data.height, cam_data.width, 4))
                np_img = np_img[:, :, :3]
                np_img = np_img[:, :, ::-1]
                np_img = apply_cityscapes_cm(np_img)
                top_view_img = np_img
                self.cameras[name].load_data(data=np_img)

            else:
                np_img = np.frombuffer(
                    cam_data.raw_data, dtype=np.uint8
                ).copy()  # .copy()
                np_img = np.reshape(np_img, (cam_data.height, cam_data.width, 4))
                np_img = np_img[:, :, :3]
                np_img = np_img[:, :, ::-1]
                self.cameras[name].load_data(data=np_img)

        if include_map:
            # extract map patch
            bb, poly = self.map_bridge.get_map_patch(
                self.roi, np.array(self.ego_pose.get_matrix())
            )
            # transform polygon to image frame
            veh_T_world = np.array(self.ego_pose.get_inverse_matrix())

            # convert points to image frame
            coefficient_list = np.ravel(veh_T_world[:3, :3]).tolist()
            coefficient_list += np.ravel(veh_T_world[:3, 3]).tolist()
            # road polygons in vehicle pose
            veh_poly = shapely.affinity.affine_transform(poly, coefficient_list)

            im_poly_exterior = []
            if isinstance(veh_poly, shapely.geometry.MultiPolygon):
                ipe = []
                for geom in veh_poly.geoms:
                    for x, y in geom.exterior.coords:
                        image_points = self.cameras["top_view"].transformGroundToImage(
                            np.array([x, y])
                        )
                        if np.all(image_points >= 0):
                            image_points = (
                                np.squeeze(image_points).astype("int").tolist()
                            )
                            ipe.append(image_points)
                im_poly_exterior.append(ipe)

                boundaries = np.zeros(
                    [
                        int(self.roi[0] // self.resolution),
                        int(self.roi[0] // self.resolution),
                        1,
                    ]
                )
                p = []
                for i in im_poly_exterior:
                    for x, y in i:
                        try:
                            boundaries[y, x] = 255
                        except IndexError:
                            pass
                    p.append(shapely.geometry.Polygon(i))

                img_poly = shapely.geometry.MultiPolygon(p)  # , holes=im_poly_interior
            else:
                for geom in veh_poly.geoms:
                    for x, y in geom.exterior.coords:
                        image_points = self.cameras["top_view"].transformGroundToImage(
                            np.array([x, y])
                        )
                        if np.all(image_points >= 0):
                            image_points = (
                                np.squeeze(image_points).astype("int").tolist()
                            )
                            ipe.append(image_points)
                boundaries = np.zeros(
                    [
                        int(self.roi[0] // self.resolution),
                        int(self.roi[0] // self.resolution),
                        1,
                    ]
                )
                for x, y in im_poly_exterior:
                    try:
                        boundaries[y, x] = 255
                    except IndexError:
                        pass

                img_poly = shapely.geometry.Polygon(
                    im_poly_exterior  # , holes=im_poly_interior
                )

            """
            im_poly_interior = []
            for hole in veh_poly.interiors:
                h_ = []
                for x, y in hole.coords:
                    image_points = self.cameras["top_view"].transformGroundToImage(
                        np.array([x, y])
                    )
                    if np.all(image_points >= 0):
                        h_.append(image_points)
                im_poly_interior.append(h_)
            """

            """
            boundaries = cv2.polylines(
                boundaries, im_poly_exterior, isClosed=False, color=(255)
            )
            """

            return {
                "image": top_view_img,
                "query_box": bb,
                "world_poly": poly,
                "image_poly": img_poly,  # FIXME
                "vehicle_poly": veh_poly,
                "boundaries": boundaries,
            }
        else:
            return {}

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

    world = client.load_world("Town05")

    dataset = Dataset(world, 4, sensor_tick=0.5)

    spectator = world.get_spectator()
    spec = dataset.ego_pose
    spec.location.z = 2
    spectator.set_transform(spec)

    tm = client.get_trafficmanager(8000)
    # dataset.sensor_platform.ego_vehicle.set_target_velocity(carla.Vector3D(5, 0, 0))
    dataset.sensor_platform.ego_vehicle.set_autopilot(True, 8000)

    frames = 1000
    [world.tick() for i in range(10)]

    settings = world.get_settings()
    settings.synchronous_mode = True  # True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    for fr in range(frames):

        frame = world.tick()
        if fr % 25 != 0:
            continue
        print("Frame: %s" % frame)

        sample = dataset.get_sample(frame_id=frame - 5)
        if sample is False:
            continue

        spec = dataset.ego_pose
        spec.location.z = 2
        spectator.set_transform(spec)

        fig, ax = plt.subplots(3, 4)
        (x_min, y_min, x_max, y_max) = dataset.map_bridge.lane_polyons.bounds
        margin = 2
        ax[0][2].set_xlim([x_min - margin, x_max + margin])
        ax[0][2].set_ylim([y_min - margin, y_max + margin])

        ax[0][0].imshow(sample["image"])
        ax[0][1].imshow(sample["boundaries"])
        dataset.map_bridge.plot_polys(ax[0][2])
        # plot_polygon(ax[0][1], sample["image_poly"], fc="blue", ec="black", alpha=0.4)

        plot_polygon(ax[0][2], sample["query_box"], fc="blue", ec="blue", alpha=0.5)
        ax[0][2].set_aspect("equal")
        (x_min, y_min, x_max, y_max) = sample["vehicle_poly"].bounds
        """
        x_max = dataset.roi[0] // 2
        x_min = -x_max
        y_max = dataset.roi[1] // 2
        y_min = -y_max
        """
        margin = 2
        ax[0][3].set_xlim([x_min - margin, x_max + margin])
        ax[0][3].set_ylim([y_min - margin, y_max + margin])
        plot_polygon(ax[0][3], sample["image_poly"])
        if isinstance(sample["image_poly"], shapely.geometry.MultiPolygon):
            for g in sample["image_poly"].geoms:
                plot_polygon(ax[0][0], g, alpha=0.5)
        else:
            plot_polygon(ax[0][0], sample["image_poly"], alpha=0.5)
        ax[0][3].plot([0], [0], marker="o", markersize=3)
        ax[0][3].set_aspect("equal")

        ax[1][1].imshow(dataset.cameras["cam_front"].data)
        ax[1][0].imshow(dataset.cameras["cam_front_left"].data)
        ax[1][2].imshow(dataset.cameras["cam_front_right"].data)

        ax[1][3].set_xlim([x_min - margin, x_max + margin])
        ax[1][3].set_ylim([y_min - margin, y_max + margin])

        ax[1][3].scatter(
            dataset.lidars["lidar_top"].data[1, :],
            dataset.lidars["lidar_top"].data[0, :],
            s=1,
        )

        print()
        # transform bev
        # print("principal point on ground")
        # print(dataset.cameras["cam_front"].transformImageToGround(np.array([960, 540])))
        print("point 10,0,0")
        c_X = dataset.cameras["cam_front"].extrinsic.transform(np.array([10, 0, 0]))
        print("on image plane")
        print(dataset.cameras["cam_front"].transformGroundToImage(np.array([10, 0, 0])))

        bev_front = dataset.cameras["cam_front"].transform()
        ax[2][1].imshow(bev_front)

        plt.show()
