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

import cv2
from dataset import Dataset
from shapely import speedups

if speedups.available:
    speedups.enable()
    print("Speedups enabled")

from pathlib import Path
import yaml
from write_scene import write_scene

from tqdm import tqdm, trange

import multiprocessing as mp
import threading as th

import concurrent.futures as concurrent
from dataset import Dataset
from map_bridge import MapBridge
import numpy as np
import logging
import shapely
import pickle
from queue import Empty
import random

from dataset_utilities.transformation import Isometry
from dataset_utilities.pcd_parser import PCDParser


def world_clock(
    server_addr,
    server_port,
    lock: mp.Lock,
    write_event: mp.Event,
    clock_terminate: mp.Event,
    global_tick: mp.Value,
):
    # global lock, write_event, end

    lock.acquire()
    client = carla.Client(server_addr, server_port)
    client.set_timeout(20.0)  # seconds

    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    lock.release()

    # [world.tick() for _ in trange(50, leave=False)]
    [world.tick() for _ in range(50)]
    tick = 0
    while not clock_terminate.is_set():
        # tick world
        # don't tick when there has been a write

        lock.acquire()
        with global_tick.get_lock():
            try:
                global_tick.value = world.tick()
            except RuntimeError:
                pass
        lock.release()
        tick += 1

        if tick % 10 == 0:
            write_event.wait()
            write_event.clear()


def translate_imu(in_queue, out_queues, terminate: mp.Event):
    while not terminate.is_set():
        try:
            data = in_queue.get(False)
        except Empty:
            continue

        ego_pose = {
            "rotation": {
                "roll": data.transform.rotation.roll,
                "pitch": data.transform.rotation.pitch,
                "yaw": data.transform.rotation.yaw,
            },
            "location": [
                data.transform.location.x,
                data.transform.location.y,
                data.transform.location.z,
            ],
        }

        out_queues.put({"frame": data.frame, "data": ego_pose})
        in_queue.task_done()
    out_queues.cancel_join_thread()
    out_queues.close()


def translate_lidar(in_queue, out_queues: mp.Queue, terminate: mp.Event):
    while not terminate.is_set():
        try:
            data = in_queue.get(False)
        except Empty:
            continue

        frame = data.frame

        point_cloud = np.frombuffer(data.raw_data, dtype=np.float32).reshape([-1, 4])

        out_queues.put({"frame": frame, "data": point_cloud})
        in_queue.task_done()
    out_queues.cancel_join_thread()
    out_queues.close()


def translate_camera(in_queue, out_queues, terminate: mp.Event):
    while not terminate.is_set():
        try:
            data = in_queue.get(False)
        except Empty:
            continue

        np_img = np.frombuffer(data.raw_data, dtype=np.uint8).copy()  # .copy()
        np_img = np.reshape(np_img, (data.height, data.width, 4))
        out_queues.put({"frame": data.frame, "data": np_img[:, :, :3]})
        in_queue.task_done()
    out_queues.cancel_join_thread()
    out_queues.close()


def sample_pipeline(
    # sensor_dict,
    # out_queue: mp.Queue,
    # write_trigger: mp.Event,
    # sensor_lock: mp.Lock,
    # shapely_lock: mp.Lock,
    # map_bridge: MapBridge,
    scene_dir,
    # step: mp.Value,
):
    global root, roi, s_dict, write_event, sensor_lock, shapely_lock, step, strtree

    def _query_queue(query_frame, query_queue):
        data = query_queue.get()
        frame_ = data["frame"]
        while frame_ < query_frame:
            data = query_queue.get()  # timeout=20.0)
            frame_ = data["frame"]
        if frame_ == query_frame:
            return data
        else:
            return False

    data_dict = {
        "lidars": {},
        "cameras": {},
        "road_boundaries": {},
        "ego_pose": {},
        "sample_dir": None,
        "root": None,
    }

    # lock until all data is loaded
    sensor_lock.acquire()

    lidar_data = s_dict["lidars"]["lidar_top"]["queue"].get()

    sample_dir = scene_dir / ("sample_%s" % step.value)

    data_dict["sample_dir"] = sample_dir
    data_dict["root"] = root

    valid_data = True
    frame_id = lidar_data["frame"]
    point_cloud = lidar_data["data"]
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
    data_dict["lidars"]["lidar_top"]["extrinsic"] = s_dict["lidars"]["lidar_top"][
        "extrinsic"
    ]

    imu = _query_queue(query_frame=frame_id, query_queue=s_dict["imu"]["queue"])
    # s_dict["imu"]["queue"].task_done()
    if imu is False:
        logging.warning("imu empty")
        valid_data = False
    else:
        ego_pose = carla.Transform(
            location=carla.Location(*imu["data"]["location"]),
            rotation=carla.Rotation(**imu["data"]["rotation"]),
        )

    data_dict["ego_pose"] = imu["data"]

    spec = imu["data"]
    spec["location"][2] += 2

    for name, cam_dict in s_dict["cameras"].items():
        data_dict["cameras"][name] = {}
        cam_data = _query_queue(query_frame=frame_id, query_queue=cam_dict["queue"])
        # cam_dict["queue"].task_done()
        if cam_data is False:
            logging.warning(name + " empty")
            valid_data = False
        else:
            np_img = cam_data["data"]
            np_img = np_img[:, :, ::-1]
            if np_img.dtype == np.float:
                np_img = (255 * np_img).astype(np.uint8)
            data_dict["cameras"][name]["data"] = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            data_dict["cameras"][name]["extrinsic"] = cam_dict["extrinsic"]
            data_dict["cameras"][name]["intrinsic"] = cam_dict["intrinsic"]

            # self.cameras[name].load_data(data=np_img)
    with step.get_lock():
        step.value += 1
    sensor_lock.release()

    write_event.set()

    if not valid_data:
        return

    # sample_dir.mkdir(parents=True, exist_ok=True)
    # extract map patch
    m_ = np.array(ego_pose.get_matrix())
    coefficient_list = np.ravel(m_[:3, :3]).tolist()
    coefficient_list += np.ravel(m_[:3, 3]).tolist()

    query_box = shapely.geometry.box(-roi[0] / 2, -roi[1] / 2, roi[0] / 2, roi[1] / 2)
    query_box = shapely.affinity.affine_transform(query_box, coefficient_list)

    # get all polys in this box
    shapely_lock.acquire()
    filtered_polygons = strtree.query(query_box)  # TODO import
    shapely_lock.release()
    filtered_polygons = [p.buffer(0.05) for p in filtered_polygons]
    union = shapely.ops.unary_union(filtered_polygons)
    union = union.buffer(-0.05)
    poly = union

    # transform polygon to image frame
    veh_T_world = np.array(ego_pose.get_inverse_matrix())

    # convert points to image frame
    coefficient_list = np.ravel(veh_T_world[:3, :3]).tolist()
    coefficient_list += np.ravel(veh_T_world[:3, 3]).tolist()
    # road polygons in vehicle pose
    veh_poly = shapely.affinity.affine_transform(poly, coefficient_list)

    veh_poly_exterior = []
    veh_poly_interior = []

    if veh_poly.type == "MultiPolygon":
        vpe = []
        for geom in veh_poly.geoms:
            for x, y in geom.exterior.coords:
                vpe.append((x, y))

            for hole in geom.interiors:
                vh_ = []
                for x, y in hole.coords:
                    vh_.append((x, y))

                veh_poly_interior.append(vh_)
            # poly = shapely.geometry.Polygon(ipe)
        veh_poly_exterior.append(vpe)

    else:

        vpe = []
        for x, y in veh_poly.exterior.coords:
            vpe.append((x, y))
        veh_poly_exterior.append(vpe)

        for hole in veh_poly.interiors:
            vh_ = []
            for x, y in hole.coords:
                vh_.append((x, y))
            veh_poly_interior.append(vh_)

    # create veh_poly
    road_boundaries = []
    road_boundaries.append(veh_poly_exterior)
    road_boundaries.append(veh_poly_interior)
    road_boundaries = np.asarray(road_boundaries, dtype=object)

    data_dict["road_boundaries"]["boundaries"] = road_boundaries

    # out_queue.put(data_dict)
    to_disk(data_dict=data_dict)
    return spec


def to_disk(data_dict):  # queue: mp.JoinableQueue, deactive: mp.Event):

    """
    while not deactive.is_set():
        try:
            data_dict = queue.get(timeout=0.5)
            # data_dict = queue.get()
        except:
            continue
    """

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
        export_dict["data"] = pcd_file.relative_to(root).with_suffix(".pcd").as_posix()
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
    # road_boundaries_img = data_dict["road_boundaries"]["image"]
    # cv2.imwrite(str(sample_dir / "road_boundary.png"), road_boundaries_img)

    boundary_file = sample_dir / "road_polygon.pkl"
    with boundary_file.open("wb+") as f:
        pickle.dump(road_boundaries, f)

    sample_file = sample_dir / "sample.yaml"
    with sample_file.open("w+") as f:
        yaml.safe_dump(sample_dict, f)
    del sample_dict
    # queue.task_done()


def main(args):
    global scene_dir, spectator

    number_of_samples = args.scene_length * sample_rate
    max_workers = min(args.processes, number_of_samples)
    with tqdm(total=number_of_samples, leave=False, smoothing=0, unit="sample") as pbar:
        """
        for i in range(number_of_samples):
            sample_pipeline(dataset.scene_dir)
            pbar.update(1)
        """
        with concurrent.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(sample_pipeline, scene_dir=scene_dir): _
                for _ in range(number_of_samples)
            }

            for future in concurrent.as_completed(futures):
                result = future.result()
                if result is not None:
                    spec = carla.Transform(
                        location=carla.Location(*result["location"]),
                        rotation=carla.Rotation(**result["rotation"]),
                    )
                    spectator.set_transform(spec)

                if future.exception() is not None:
                    print(future.exception())
                pbar.update(1)

        mp.active_children()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Town01")
    parser.add_argument("--server_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=2000)
    parser.add_argument("--tm_port", type=int, default=8000)
    parser.add_argument("--base_path", type=str, default="/tmp/carla")

    parser.add_argument("--step_delta", type=float, default=0.05)
    parser.add_argument("--scene_length", type=int, default=90)

    parser.add_argument(
        "--random_sample",
        action="store_true",
        default=False,
    )
    parser.add_argument("--spawn_point", type=int, default=0)
    parser.add_argument("--number_of_scenes", type=int, default=45)
    parser.add_argument("--processes", type=int, default=2)

    args = parser.parse_args()

    # logger = mp.log_to_stderr()
    # logger.setLevel(logging.DEBUG)

    client = carla.Client(args.server_addr, args.server_port)
    client.set_timeout(20.0)  # seconds

    cv = client.get_client_version()
    sv = client.get_server_version()
    print(f"Client version: {cv}")
    print(f"Server version: {sv}")

    world = client.load_world(args.map)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    tm = client.get_trafficmanager(args.tm_port)

    spectator = world.get_spectator()
    town = args.map

    base = Path(args.base_path)  # TODO define intital step
    root = base / town
    root.mkdir(parents=True, exist_ok=True)

    spawn_point = args.spawn_point

    # fix sensor rate at 2hz
    sample_rate = 2  # [hz]

    # start writing thread define all locks and events for communication between clock and dataset
    lock = mp.Lock()
    clock_terminate = mp.Event()
    write_event = mp.Event()

    # initialize clock
    frame = mp.Value("i", 0)
    clock_proc = th.Thread(
        target=world_clock,
        kwargs={
            "server_addr": args.server_addr,
            "server_port": args.server_port,
            "lock": lock,
            "write_event": write_event,
            "global_tick": frame,
            "clock_terminate": clock_terminate,
        },
    )

    lock.acquire()
    roi = (50, 50)
    resolution = 0.04
    dataset = Dataset(
        world,
        vehicle_spawn_point=spawn_point + 1,
        sensor_tick=1 / sample_rate,
        roi=roi,
        root=root,
        resolution=resolution,
        scene=spawn_point,
    )
    # print("fin")
    lock.release()
    # print("clock start")
    clock_proc.start()
    # print("started")
    tm.set_synchronous_mode(True)
    # print("tm dync started")

    # dataset.sensor_platform.ego_vehicle.set_autopilot(True, 6006)

    strtree = dataset.map_bridge.str_tree
    sensor_dict = dataset.sensor_dict

    # io_queue = mp.JoinableQueue(200)
    sensor_lock = mp.Lock()
    shapely_lock = mp.Lock()
    step = mp.Value("i", 0)
    # define args for process

    s_dict = dataset.sensor_dict
    threads_terminate = mp.Event()
    threads = []

    lidar_q = mp.Queue()
    # print(lidar_q)
    # print("start lidar")
    lidar_t = th.Thread(
        target=translate_lidar,
        args=(
            sensor_dict["lidars"]["lidar_top"]["queue"],
            lidar_q,
            threads_terminate,
        ),
    )
    lidar_t.start()
    threads.append(lidar_t)
    s_dict["lidars"]["lidar_top"]["queue"] = lidar_q
    imu_q = mp.Queue()
    imu_t = th.Thread(
        target=translate_imu,
        args=(sensor_dict["imu"]["queue"], imu_q, threads_terminate),
    )
    imu_t.start()
    s_dict["imu"]["queue"] = imu_q
    threads = [lidar_t, imu_t]
    for name, cam in sensor_dict["cameras"].items():
        cam_q = mp.Queue()
        cam_t = th.Thread(
            target=translate_camera,
            args=(cam["queue"], cam_q, threads_terminate),
        )
        cam_t.start()
        threads.append(cam_t)
        s_dict["cameras"][name]["queue"] = cam_q

    spawn_points = world.get_map().get_spawn_points()
    # define which spawn points to use

    if args.random_sample:
        selected_scenes_indx = random.sample(
            range(len(spawn_points)), k=args.number_of_scenes
        )
    else:
        selected_scenes_indx = range(
            args.spawn_point, args.spawn_point + args.number_of_scenes
        )
    with tqdm(selected_scenes_indx) as scenes:
        for i, scene_idx in enumerate(scenes):
            scenes.set_description(f"[{args.map}] rendering scene_{scene_idx}")

            # freeze global clock
            lock.acquire()
            args.spawn_point = i
            scene_dir = root / f"scene_{i}"

            with step.get_lock():
                step.value = 0

            dataset.sensor_platform.ego_vehicle.set_autopilot(False, args.tm_port)
            dataset.sensor_platform.teleport(spawn_points[i])
            dataset.sensor_platform.ego_vehicle.set_autopilot(True, args.tm_port)

            # flush all data queues
            # since lidar_top is reference it should be enough to flush one queue
            while not lidar_q.empty():
                d = lidar_q.get(timeout=5.0)

            write_event.set()
            lock.release()

            # better safe than sorry
            lidar_q.get()
            write_event.set()

            main(args)
