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

from tqdm import tqdm, trange

from multiprocessing import JoinableQueue, Queue, Lock, Process, Event
from concurrent.futures import ThreadPoolExecutor


def to_disk(queue: Queue):

    while True:
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
        data_dict = queue.get()

        sample_dir = data_dict["sample_dir"]
        root = data_dict["root"]

        sample_dict = {}
        ego_pose = data_dict["ego_pose"]
        sample_dict["ego_pose"] = {
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

        sample_dict["sensors"] = {}
        lidars = data_dict["lidars"]
        for name, lidar in lidars.items():
            export_dict = {}
            export_file = lidar.exportPCD(sample_dir)
            export_dict["data"] = (
                export_file.relative_to(root).with_suffix(".pcd").as_posix()
            )
            export_dict["extrinsic"] = lidar.M.tolist()
            sample_dict["sensors"][lidar.id] = export_dict

        cameras = data_dict["cameras"]
        for name, cam in cameras.items():
            export_dict = {}
            export_file = cam.write_data(sample_dir)
            export_dict["data"] = export_file.relative_to(root).as_posix()
            export_dict["extrinsic"] = cam.M.tolist()
            export_dict["intrinsic"] = cam.K.tolist()
            sample_dict["sensors"][cam.id] = export_dict

        road_boundaries = data_dict["road_boundaries"]["boundaries"]
        road_boundaries_img = data_dict["road_boundaries"]["boundaries_img"]

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
    )

    io_queue = JoinableQueue()
    io_exec = ThreadPoolExecutor(max_workers=4)
    [io_exec.submit(to_disk, io_queue) for i in range(4)]

    tm = client.get_trafficmanager(8000)
    dataset.sensor_platform.ego_vehicle.set_autopilot(True, 8000)

    with trange(number_of_samples, leave=False, smoothing=0, unit="sample") as t_range:
        for step in t_range:
            sample_dir = scene_dir / ("sample_%s" % step)
            sample_dir.mkdir(parents=True, exist_ok=True)

            """
            if dataset.sensor_platform.ego_pose.qsize() < 100:
                queue_event.set()
            else:
                queue_event.clear()
            """

            sample = dataset.get_sample(frame_id=0, include_map=True)
            if sample is False:
                continue
            t_range.set_description(f"sample_{step} / {number_of_samples}")

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
            data_dict = {}

            data_dict["sample_dir"] = sample_dir
            data_dict["root"] = root

            data_dict["road_boundaries"] = {}
            data_dict["road_boundaries"]["image"] = dataset.boundaries_img
            data_dict["road_boundaries"]["boundaries"] = dataset.road_boundaries

            data_dict["ego_pose"] = {
                "rotation": {
                    "roll": dataset.ego_pose.rotation.roll,
                    "pitch": dataset.ego_pose.rotation.pitch,
                    "yaw": dataset.ego_pose.rotation.yaw,
                },
                "location": [
                    dataset.ego_pose.location.x,
                    dataset.ego_pose.location.y,
                    dataset.ego_pose.location.z,
                ],
            }
            data_dict["lidars"] = {}
            for name, lidar in dataset.lidars.items():
                data_dict["lidars"][name] = lidar

            data_dict["cameras"] = {}
            for name, cam in dataset.cameras.items():
                data_dict["cameras"][name] = cam

            io_queue.put(data_dict)

            write_event.set()

    # io_queue.join()
    while io_queue.qsize() > 0:
        pass
    # io_queue.join()
    io_exec.shutdown()
    end.set()
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors()])


def write_scene(args, client=None, world=None):

    client = carla.Client(args.server_addr, args.server_port)
    client.set_timeout(20.0)  # seconds

    world = client.load_world(args.map)
    world.set_weather(carla.WeatherParameters.ClearNoon)

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

    p.join()

    # dataset.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors()])
    client = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="Town01")
    parser.add_argument("--server_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=2000)
    parser.add_argument("--base_path", type=str, default="/home/dominic/data/carla")

    parser.add_argument("--step_delta", type=float, default=0.05)
    parser.add_argument("--scene_length", type=int, default=90)

    parser.add_argument("--spawn_point", type=int, default=0)

    args = parser.parse_args()
    write_scene(args)
