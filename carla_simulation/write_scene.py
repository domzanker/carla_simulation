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

from multiprocessing import Lock, Process, Event


def write(
    args,
    lock: Lock,
    end: Event,
    write_event: Event,
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

    tm = client.get_trafficmanager(8000)
    dataset.sensor_platform.ego_vehicle.set_autopilot(True, 8000)

    with trange(number_of_samples, leave=False, smoothing=0, unit="sample") as t_range:
        for step in t_range:
            sample_dir = scene_dir / ("sample_%s" % step)
            sample_dir.mkdir(parents=True, exist_ok=True)

            sample = dataset.get_sample(frame_id=0, include_map=True)
            if sample is False:
                continue
            t_range.set_description(f"sample_{step} / {number_of_samples}")

            sample_dict = {}
            cv2.imwrite(str(sample_dir / "road_boundary.png"), dataset.boundaries_img)

            sample_dict["ego_pose"] = {
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
            sample_dict["sensors"] = {}
            for name, lidar in dataset.lidars.items():
                export_dict = {}
                export_file = lidar.exportPCD(sample_dir)
                export_dict["data"] = (
                    export_file.relative_to(root).with_suffix(".pcd").as_posix()
                )
                export_dict["extrinsic"] = lidar.M.tolist()
                sample_dict["sensors"][lidar.id] = export_dict

            for name, cam in dataset.cameras.items():
                export_dict = {}
                export_file = cam.write_data(sample_dir)
                export_dict["data"] = export_file.relative_to(root).as_posix()
                export_dict["extrinsic"] = cam.M.tolist()
                export_dict["intrinsic"] = cam.K.tolist()
                sample_dict["sensors"][cam.id] = export_dict

            boundary_file = sample_dir / "road_polygon.pkl"
            with boundary_file.open("wb+") as f:
                pickle.dump(dataset.road_boundaries, f)

            sample_file = sample_dir / "sample.yaml"
            with sample_file.open("w+") as f:
                yaml.safe_dump(sample_dict, f)
            del sample_dict
            write_event.set()

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

    # start writing thread
    lock = Lock()
    end = Event()
    write_event = Event()
    p = Process(
        target=write,
        kwargs={
            "args": args,
            "end": end,
            "write_event": write_event,
            "lock": lock,
            "inital_step": 0,
        },
    )
    p.start()

    number_of_ticks = args.scene_length / 0.05
    number_of_ticks = int(number_of_ticks) + 1

    write_cntr = 0
    write_tld = 50  # at least one write every 20 ticks
    while not end.is_set():
        # tick world
        # don't tick when there has been a write
        if write_event.is_set():
            write_event.clear()
            write_cntr = 0
        else:
            write_cntr += 1

        if write_cntr < write_tld:
            lock.acquire()
            world.tick()
            lock.release()
        else:
            write_event.wait()

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
