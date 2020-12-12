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
import pickle
from map_bridge import MapBridge
from sensor_platform import SensorPlatform

from multiprocessing.queues import Queue
import numpy as np
import cv2
import shapely
from math import ceil, floor, degrees

from copy import deepcopy, copy

from city_scapes_cm import apply_cityscapes_cm
from dataset_utilities.transformation import Isometry
from dataset_utilities.camera import Camera, BirdsEyeView
from dataset_utilities.pcd_parser import PCDParser

from scipy.spatial.transform import Rotation
import yaml
import time
import logging

from pathlib import Path
from tqdm import trange, tqdm

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Lock, Value, Process, Event, JoinableQueue
from threading import Thread


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
        root="",
        scene: int = 0,
        sensor_setup="sensor_setup.yaml",
    ):

        self.cameras = {}
        self.camera_queues = {}
        self.sensor_calibrations = {}

        self.spectator = world.get_spectator()

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

        self.sensor_lock = Lock()
        self.road_boundary_lock = Lock()

        # setup all variables needed to write a scene
        self.step = Value("i", 0)
        self.root = root
        self.scene_dir = self.root / f"scene_{scene}"

    def process_samples(self, write_event, number_of_samples):

        # spawn child processes
        io_queue = JoinableQueue()
        io_finished = Event()

        io_exec = ThreadPoolExecutor(max_workers=4)
        [io_exec.submit(self.to_disk, io_queue, io_finished) for i in range(4)]

        workers = min(8, number_of_samples)
        processes = [
            Thread(target=self.sample_pipeline, args=(io_queue, write_event))
            for _ in range(workers)
        ]
        # processes = ThreadPoolExecutor(max_workers=workers)

        with tqdm(
            total=number_of_samples, leave=False, smoothing=0, unit="sample"
        ) as pbar:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.sample_pipeline, io_queue, write_event): _
                    for _ in range(number_of_samples)
                }

                results = {}
                for future in as_completed(futures):
                    pbar.update(1)
                """
                process = processes[int(step % workers)]
                if process.is_alive():
                    process.join()
                process.start()
                """

        """
        for p in processes:
            try:
                p.join()
            except RuntimeError:
                continue
        [p.join() for p in processes]
        """
        io_queue.join()
        io_finished.set()
        io_exec.shutdown()

    def sample_pipeline(self, queue, write_trigger: Event, include_map: bool = True):
        """
        cameras:
            cam_name: camera
            ...
            ...
            boundaries_img: image
        lidars:
            lidar_name: lidar
            ...
        road_boundaries:
            boundaries: np.ndarray
            image

        ego_pose: carla.Transform

        sample_dir
        root
        """
        data_dict = {
            "lidars": {},
            "cameras": {},
            "road_boundaries": {},
            "ego_pose": {},
            "sample_dir": None,
            "root": None,
        }

        # lock until all data is loaded
        self.sensor_lock.acquire()
        lidar_data = self.lidar_queues["lidar_top"].get()

        sample_dir = self.scene_dir / ("sample_%s" % self.step.value)

        data_dict["sample_dir"] = sample_dir
        data_dict["root"] = self.root

        valid_data = True
        frame_id = lidar_data.frame

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

        data_dict["lidars"]["lidar_top"] = {}
        data_dict["lidars"]["lidar_top"]["data"] = point_cloud
        data_dict["lidars"]["lidar_top"]["extrinsic"] = self.lidars[
            "lidar_top"
        ].M.tolist()

        imu = self._query_queue(
            query_frame=frame_id, query_queue=self.sensor_platform.ego_pose
        )
        if imu is False:
            logging.warn("imu empty")
            valid_data = False
        else:
            ego_pose = imu.transform

        data_dict["ego_pose"] = {
            "rotation": {
                "roll": ego_pose.rotation.roll,
                "pitch": ego_pose.rotation.pitch,
                "yaw": ego_pose.rotation.yaw,
            },
            "location": [
                ego_pose.location.x,
                ego_pose.location.y,
                ego_pose.location.z,
            ],
        }

        spec = ego_pose
        spec.location.z += 2
        self.spectator.set_transform(spec)

        for name, cam_queue in self.camera_queues.items():
            data_dict["cameras"][name] = {}
            cam_data = self._query_queue(query_frame=frame_id, query_queue=cam_queue)
            if cam_data is False:
                logging.warn(name + " empty")
                valid_data = False
            else:
                np_img = np.frombuffer(
                    cam_data.raw_data, dtype=np.uint8
                ).copy()  # .copy()
                np_img = np.reshape(np_img, (cam_data.height, cam_data.width, 4))
                np_img = np_img[:, :, :3]
                np_img = np_img[:, :, ::-1]
                if np_img.dtype == np.float:
                    np_img = (255 * np_img).astype(np.uint8)
                data_dict["cameras"][name]["data"] = cv2.cvtColor(
                    np_img, cv2.COLOR_RGB2BGR
                )
                cam = self.cameras[name]
                data_dict["cameras"][name]["extrinsic"] = cam.M.tolist()
                data_dict["cameras"][name]["intrinsic"] = cam.K.tolist()

                # self.cameras[name].load_data(data=np_img)

        self.sensor_lock.release()

        with self.step.get_lock():
            self.step.value += 1
        write_trigger.set()

        if not valid_data:
            return

        # sample_dir.mkdir(parents=True, exist_ok=True)

        if include_map:
            # extract map patch
            self.road_boundary_lock.acquire()
            bb, poly = self.map_bridge.get_map_patch(
                self.roi, np.array(ego_pose.get_matrix())
            )
            # transform polygon to image frame
            veh_T_world = np.array(ego_pose.get_inverse_matrix())

            # convert points to image frame
            coefficient_list = np.ravel(veh_T_world[:3, :3]).tolist()
            coefficient_list += np.ravel(veh_T_world[:3, 3]).tolist()
            # road polygons in vehicle pose
            veh_poly = shapely.affinity.affine_transform(poly, coefficient_list)

            top_view = self.cameras["top_view"]

            self.road_boundary_lock.release()

            im_poly_exterior = []
            im_poly_interior = []
            veh_poly_exterior = []
            veh_poly_interior = []

            if veh_poly.type == "MultiPolygon":
                ipe = []
                vpe = []
                for geom in veh_poly.geoms:
                    for x, y in geom.exterior.coords:
                        image_points = top_view.transformGroundToImage(np.array([x, y]))
                        # if np.all(image_points >= 0):
                        image_points = np.squeeze(image_points).astype("int").tolist()
                        ipe.append(image_points)
                        vpe.append((x, y))

                    for hole in geom.interiors:
                        h_ = []
                        vh_ = []
                        for x, y in hole.coords:
                            image_points = top_view.transformGroundToImage(
                                np.array([x, y])
                            )
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
                    image_points = top_view.transformGroundToImage(np.array([x, y]))
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
                        image_points = top_view.transformGroundToImage(np.array([x, y]))
                        # if np.all(image_points >= 0):
                        h_.append(image_points)
                        vh_.append((x, y))
                    im_poly_interior.append(h_)
                    veh_poly_interior.append(vh_)

            # create veh_poly
            road_boundaries = []
            road_boundaries.append(veh_poly_exterior)
            road_boundaries.append(veh_poly_interior)
            road_boundaries = np.asarray(road_boundaries, dtype=object)

            data_dict["road_boundaries"]["boundaries"] = road_boundaries

            boundaries_img = np.full(
                [
                    int(self.roi[1] // self.resolution),
                    int(self.roi[0] // self.resolution),
                    1,
                ],
                255,
                dtype=np.uint8,
            )

            for exterior_bounds in im_poly_exterior:
                polyline = np.array(exterior_bounds, dtype=np.int32)
                boundaries_img = cv2.polylines(
                    boundaries_img,
                    polyline[np.newaxis, :, :],
                    # polyline,
                    isClosed=False,
                    color=0,
                )
            for interior_bounds in im_poly_interior:
                polyline = np.array(interior_bounds, dtype=np.int32)
                boundaries_img = cv2.polylines(
                    boundaries_img,
                    polyline[np.newaxis, :, :],
                    # polyline,
                    isClosed=True,
                    color=0,
                )

            data_dict["road_boundaries"]["image"] = boundaries_img

            queue.put(data_dict)

        else:
            queue.put(data_dict)

    def to_disk(self, queue: Queue, deactive: Event):

        while not deactive.is_set():
            """
            cameras:
                cam_name: camera
                ...
                ...
                boundaries_img: image
            lidars:
                lidar_name: lidar
                ...
            road_boundaries:
                boundaries: np.ndarray
                image

            ego_pose: carla.Transform

            sample_dir
            root
            """
            try:
                data_dict = queue.get(timeout=0.5)
                # data_dict = queue.get()
            except:
                continue

            pcd_parser = PCDParser()

            root = Path(data_dict["root"])
            sample_dir = data_dict["sample_dir"]

            sample_dir.mkdir(exist_ok=True, parents=True)

            sample_dict = {}
            ego_pose = data_dict["ego_pose"]

            sample_dict["ego_pose"] = ego_pose
            sample_dict["sensors"] = {}

            lidars = data_dict["lidars"]
            for name, lidar in lidars.items():
                export_dict = {}
                pcd_parser.addCalibration(
                    calibration=Isometry.from_matrix(np.asarray(lidar["extrinsic"]))
                )
                pcd_file = sample_dir / name
                pcd_parser.write(
                    point_cloud=np.swapaxes(lidar["data"], 0, 1),
                    file_name=str(pcd_file),
                )
                export_dict["data"] = (
                    pcd_file.relative_to(root).with_suffix(".pcd").as_posix()
                )
                export_dict["extrinsic"] = lidar["extrinsic"]
                sample_dict["sensors"][name] = export_dict

            cameras = data_dict["cameras"]
            for name, cam in cameras.items():
                export_dict = {}
                export_file = sample_dir / f"{name}.png"
                cv2.imwrite(str(export_file), cam["data"])
                export_dict["data"] = export_file.relative_to(root).as_posix()
                export_dict["extrinsic"] = cam["extrinsic"]
                export_dict["intrinsic"] = cam["intrinsic"]
                sample_dict["sensors"][name] = export_dict

            road_boundaries = data_dict["road_boundaries"]["boundaries"]
            road_boundaries_img = data_dict["road_boundaries"]["image"]

            boundary_file = sample_dir / "road_polygon.pkl"
            with boundary_file.open("wb+") as f:
                pickle.dump(road_boundaries, f)
            cv2.imwrite(str(sample_dir / "road_boundary.png"), road_boundaries_img)

            sample_file = sample_dir / "sample.yaml"
            with sample_file.open("w+") as f:
                yaml.safe_dump(sample_dict, f)
            del sample_dict
            queue.task_done()

    def _query_queue(self, query_frame, query_queue):
        data = query_queue.get()
        frame_ = data.frame
        while frame_ < query_frame:
            try:
                data = query_queue.get()  # timeout=20.0)
                # print(f"{frame_} _ {query_frame}")
            except:
                return False
            frame_ = data.frame
        if frame_ == query_frame:
            return data
        else:
            return False

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
