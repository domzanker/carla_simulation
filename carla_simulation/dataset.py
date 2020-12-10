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
from map_bridge import MapBridge
from sensor_platform import SensorPlatform

from queue import Queue, Empty
import numpy as np
import cv2
import shapely
from math import ceil, floor, degrees

from copy import deepcopy, copy

from city_scapes_cm import apply_cityscapes_cm
from dataset_utilities.transformation import Isometry
from dataset_utilities.camera import Camera, BirdsEyeView

from scipy.spatial.transform import Rotation
import yaml
import time
import logging


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
    def __init__(
        self,
        world,
        vehicle_spawn_point=0,
        sensor_tick=5,
        roi=[30, 20],
        resolution=0.04,
        sensor_setup="sensor_setup.yaml",
    ):

        self.cameras = {}
        self.camera_queues = {}
        self.sensor_calibrations = {}

        self.roi = roi
        self.resolution = resolution

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[vehicle_spawn_point]
        self.sensor_platform = SensorPlatform(world, spawn_point, sensor_tick)

        sensor_config = self.__load_sensor_setup__(sensor_setup)

        top_view = sensor_config["top_view"]

        location = carla.Location(**top_view["extrinsic"]["location"])
        rotation = carla.Rotation(**top_view["extrinsic"]["rotation"])
        extrinsic = carla.Transform(location=location, rotation=rotation)
        try:
            intrinsic = top_view["intrinsic"]
        except KeyError:
            intrinsic = {}
        cam, queue, vTs = self.sensor_platform.add_topview(
            "top_view",
            veh_T_sensor=extrinsic,
            blueprint="sensor.camera.rgb",
            roi=self.roi,
            resolution=self.resolution,
            **intrinsic,
        )
        self.cameras["top_view"] = cam
        self.camera_queues["top_view"] = queue
        self.sensor_calibrations["top_view"] = vTs

        for cam, configs in sensor_config["cameras"].items():

            location = carla.Location(**configs["extrinsic"]["location"])
            rotation = carla.Rotation(**configs["extrinsic"]["rotation"])
            extrinsic = carla.Transform(location=location, rotation=rotation)
            try:
                intrinsic = configs["intrinsic"]
            except KeyError:
                intrinsic = {}

            bev, q_, vTs = self.sensor_platform.add_camera(
                name=cam, veh_T_sensor=extrinsic, **intrinsic
            )
            self.cameras[cam] = bev
            self.camera_queues[cam] = q_
            self.sensor_calibrations[cam] = Isometry.from_carla_transform(vTs)

        self.lidars = {}
        self.lidar_queues = {}
        for lidar, configs in sensor_config["lidars"].items():
            location = carla.Location(**configs["extrinsic"]["location"])
            rotation = carla.Rotation(**configs["extrinsic"]["rotation"])
            extrinsic = carla.Transform(location=location, rotation=rotation)
            try:
                intrinsic = configs["intrinsic"]
            except KeyError:
                intrinsic = {}
            l, q_, veh_T_sensor = self.sensor_platform.add_lidar(
                name=lidar,
                veh_T_sensor=extrinsic,
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

        self.road_boundaries = []
        self.boundaries_img = np.zeros([100, 100, 1], np.uint8)

    def _query_queue(self, query_frame, query_queue):
        data = query_queue.get()
        frame_ = data.frame
        while frame_ < query_frame:
            data = query_queue.get()
            frame_ = data.frame
        if frame_ == query_frame:
            return data
        else:
            return False

    def get_sample(self, frame_id, include_map: bool = True):
        self.road_boundaries = []
        # get images
        for name, lidar in self.lidars.items():
            lidar_data = self._query_queue(
                query_frame=frame_id, query_queue=self.lidar_queues[name]
            )
            if lidar_data is False:
                logging.warn(name + " empty")
                return False

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

        imu = self._query_queue(
            query_frame=frame_id, query_queue=self.sensor_platform.ego_pose
        )
        if imu is False:
            logging.warn("imu empty")
            return False
        self.ego_pose = imu.transform

        for name, cam in self.cameras.items():
            cam_data = self._query_queue(
                query_frame=frame_id, query_queue=self.camera_queues[name]
            )
            if cam_data is False:
                logging.warn(name + " empty")
                return False

            np_img = np.frombuffer(cam_data.raw_data, dtype=np.uint8).copy()  # .copy()
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

            self.boundaries_img = np.full(
                [
                    self.cameras["top_view"].data.shape[0],
                    self.cameras["top_view"].data.shape[1],
                    1,
                ],
                255,
                dtype=np.uint8,
            )

            im_poly_exterior = []
            im_poly_interior = []
            veh_poly_exterior = []
            veh_poly_interior = []

            if veh_poly.type == "MultiPolygon":
                ipe = []
                vpe = []
                for geom in veh_poly.geoms:
                    for x, y in geom.exterior.coords:
                        image_points = self.cameras["top_view"].transformGroundToImage(
                            np.array([x, y])
                        )
                        # if np.all(image_points >= 0):
                        image_points = np.squeeze(image_points).astype("int").tolist()
                        ipe.append(image_points)
                        vpe.append((x, y))

                    for hole in geom.interiors:
                        h_ = []
                        vh_ = []
                        for x, y in hole.coords:
                            image_points = self.cameras[
                                "top_view"
                            ].transformGroundToImage(np.array([x, y]))
                            # if np.all(image_points >= 0):
                            h_.append(image_points)
                            vh_.append((x, y))

                        im_poly_interior.append(h_)
                        veh_poly_interior.append(vh_)
                    # poly = shapely.geometry.Polygon(ipe)
                im_poly_exterior.append(ipe)
                veh_poly_exterior.append(vpe)

            else:

                ipe = []
                vpe = []
                for x, y in veh_poly.exterior.coords:
                    image_points = self.cameras["top_view"].transformGroundToImage(
                        np.array([x, y])
                    )
                    # if np.all(image_points >= 0):
                    image_points = np.squeeze(image_points).astype("int").tolist()
                    ipe.append(image_points)
                    vpe.append((x, y))
                im_poly_exterior.append(ipe)
                veh_poly_exterior.append(vpe)

                for hole in veh_poly.interiors:
                    h_ = []
                    vh_ = []
                    for x, y in hole.coords:
                        image_points = self.cameras["top_view"].transformGroundToImage(
                            np.array([x, y])
                        )
                        # if np.all(image_points >= 0):
                        h_.append(image_points)
                        vh_.append((x, y))
                    im_poly_interior.append(h_)
                    veh_poly_interior.append(vh_)

            # create veh_poly
            self.road_boundaries.append(veh_poly_exterior)
            self.road_boundaries.append(veh_poly_interior)
            self.road_boundaries = np.asarray(self.road_boundaries, dtype=object)

            for exterior_bounds in im_poly_exterior:
                polyline = np.array(exterior_bounds, dtype=np.int32)
                self.boundaries_img = cv2.polylines(
                    self.boundaries_img,
                    polyline[np.newaxis, :, :],
                    # polyline,
                    isClosed=False,
                    color=0,
                )
            for interior_bounds in im_poly_interior:
                polyline = np.array(interior_bounds, dtype=np.int32)
                self.boundaries_img = cv2.polylines(
                    self.boundaries_img,
                    polyline[np.newaxis, :, :],
                    # polyline,
                    isClosed=True,
                    color=0,
                )

                return True
        else:
            return {}

    def polylines_from_shapely(self, shapes):
        raise NotImplementedError

    def __load_sensor_setup__(self, path):
        with open(path, "r") as f:
            configs = yaml.safe_load(f)
        return configs

    def destroy(self):
        # destroys all actors in dataset
        self.sensor_platform.destroy()
        self.sensor_platform = None
        return True


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

    world = client.load_world("Town03")

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

        # ax[0][0].imshow(sample["image"])
        ax[0][1].imshow(dataset.boundaries_img)
        dataset.map_bridge.plot_polys(ax[0][2])
        # plot_polygon(ax[0][1], sample["image_poly"], fc="blue", ec="black", alpha=0.4)

        # plot_polygon(ax[0][2], sample["query_box"], fc="blue", ec="blue", alpha=0.5)
        ax[0][2].set_aspect("equal")
        # (x_min, y_min, x_max, y_max) = sample["vehicle_poly"].bounds
        """
        x_max = dataset.roi[0] // 2
        x_min = -x_max
        y_max = dataset.roi[1] // 2
        y_min = -y_max
        """
        margin = 2
        ax[0][3].set_xlim([x_min - margin, x_max + margin])
        ax[0][3].set_ylim([y_min - margin, y_max + margin])
        # plot_polygon(ax[0][3], sample["image_poly"])
        """
        if sample["image_poly"].type == "MultiPolygon":

            for g in sample["image_poly"].geoms:
                plot_polygon(ax[0][0], g, alpha=0.5)
        else:
            plot_polygon(ax[0][0], sample["image_poly"], alpha=0.5)
        """
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

        bev_front = dataset.cameras["cam_front"].transform()
        ax[2][1].imshow(bev_front)

        plt.show()
