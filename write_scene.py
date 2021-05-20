import glob
import os
import sys
import cv2

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
from dataset import Dataset
from shapely import speedups

if speedups.available:
    speedups.enable()
    # print("Speedups enabled")

from pathlib import Path
import yaml
import pickle
import logging
import time

from dataset_utilities.pcd_parser import PCDParser
from dataset_utilities.transformation import Isometry

from tqdm import tqdm, trange

from multiprocessing import JoinableQueue, Queue, Lock, Process, Event
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import multiprocessing

import numpy as np


def to_disk(queue: Queue, deactive: Event):

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
        except:
            continue

        pcd_parser = PCDParser()

        root = data_dict["root"]
        sample_dir = data_dict["sample_dir"]
        print(sample_dir)

        sample_dir.mkdir(exist_ok=True, parents=True)

        sample_dict = {}
        ego_pose = data_dict["ego_pose"]

        sample_dict["ego_pose"] = ego_pose
        sample_dict["sensors"] = {}

        lidars = data_dict["lidars"]
        for name, lidar in lidars.items():
            export_dict = {}
            pcd_parser.addCalibration(
                calibration=Isometry.from_carla_transform(
                    np.asarray(lidar["extrinsic"])
                )
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
            sample_dict["sensors"][lidar.id] = export_dict

        cameras = data_dict["cameras"]
        for name, cam in cameras.items():
            export_dict = {}
            export_file = str(sample_dir / f"{name}.png")
            cv2.imwrite(export_file, cam["data"])
            # cv save file
            export_dict["data"] = export_file.relative_to(root).as_posix()
            export_dict["extrinsic"] = cam["extrinsic"]
            export_dict["intrinsic"] = cam["intrinsic"]
            sample_dict["sensors"][cam] = export_dict

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


def write(
    args,
    lock: Lock,
    end: Event,
    write_event: Event,
    queue_event: Event,
    inital_step: int = 0,
):
    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(logging.DEBUG)
    lock.acquire()
    client = carla.Client(args.server_addr, args.server_port)
    client.set_timeout(20.0)  # seconds

    world = client.get_world()
    lock.release()

    step = inital_step
    town = args.map
    step_delta = args.step_delta

    base = Path(args.base_path)
    root = base / town
    root.mkdir(parents=True, exist_ok=True)

    spawn_point = args.spawn_point
    scene_dir = root / ("scene_%s" % spawn_point)

    # fix sensor rate at 2hz
    sample_rate = 2  # [hz]
    number_of_samples = args.scene_length * sample_rate

    dataset = Dataset(
        world,
        vehicle_spawn_point=spawn_point,
        sensor_tick=1 / sample_rate,
        roi=(50, 50),
        root=root,
        scene=spawn_point,
    )

    tm = client.get_trafficmanager(8000)
    dataset.sensor_platform.ego_vehicle.set_autopilot(True, 8000)

    dataset.process_samples(
        write_event=write_event, number_of_samples=number_of_samples
    )

    end.set()
    write_event.set()
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors()])


def write_scene(args, client=None, world=None, load_world: bool = True):

    client = carla.Client(args.server_addr, args.server_port)
    client.set_timeout(20.0)  # seconds

    if load_world:
        world = client.load_world(args.map)
        world.set_weather(carla.WeatherParameters.ClearNoon)
    else:
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # [world.tick() for _ in trange(50, leave=False)]

    # start writing thread
    lock = Lock()
    end = Event()
    write_event = Event()

    queue_has_space = Event()
    p = Process(
        target=write,
        kwargs={
            "args": args,
            "end": end,
            "write_event": write_event,
            "queue_event": queue_has_space,
            "lock": lock,
            "inital_step": 0,
        },
    )
    p.start()

    number_of_ticks = args.scene_length / 0.05
    number_of_ticks = int(number_of_ticks) + 1

    write_cntr = 0
    write_tld = 50  # at least one write every 20 ticks
    [world.tick() for _ in range(50)]
    tick = 0
    while not end.is_set():
        # tick world
        # don't tick when there has been a write

        lock.acquire()
        world.tick()
        lock.release()
        tick += 1

        if tick % 10 == 0:
            write_event.wait()
            write_event.clear()
        """
        if write_event.is_set():
            write_event.clear()
            write_cntr -= 15  # one write is equivalent to 1 / 0.05 / fps
        else:
            write_cntr += 1

        if write_cntr < write_tld:
            lock.acquire()
            world.tick()
            lock.release()
        else:
            write_event.wait()
        """

    print("join write ")
    p.join()
    print("finished write ")

    # dataset.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors()])
    client = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Town01", help="town/map")
    parser.add_argument(
        "--server_addr",
        type=str,
        default="localhost",
        help="network adress for carla server",
    )
    parser.add_argument(
        "--server_port", type=int, default=2000, help="port for carla server"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/home/dominic/data/carla",
        help="path for data output",
    )

    parser.add_argument(
        "--step_delta",
        type=float,
        default=0.05,
        help="incremental step size for syncronous carla",
    )
    parser.add_argument(
        "--scene_length",
        type=int,
        default=90,
        help="length of recorded scene in seconds",
    )

    parser.add_argument(
        "--spawn_point", type=int, default=0, help="index of the spawn point"
    )

    args = parser.parse_args()
    write_scene(args)
