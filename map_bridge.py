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

import matplotlib.pyplot as plt
import shapely.geometry
import shapely.ops
import shapely.strtree
import shapely.affinity
from math import cos, sin, radians
from descartes.patch import PolygonPatch
import fiona
import numpy as np

from typing import Union, Tuple
from dataset_utilities.transformation import Isometry

"""
First find out whether we can auitomatically extract map information from the carla map
"""


def plot_line(ax, ob, color=None):
    x, y = ob.xy
    if color is not None:
        ax.plot(x, y, color=color)
    else:
        ax.plot(x, y)


def plot_polygon(ax, polygon, fc="green", ec="black", *args, **kwargs):
    patch = PolygonPatch(polygon, fc=fc, ec=ec, *args, **kwargs)
    ax.add_patch(patch)


class Segment(object):
    def __init__(self, waypoints: Tuple[carla.Waypoint]):
        self.start = (
            waypoints[0].transform.location.x,
            # FIXME
            waypoints[0].transform.location.y,
        )
        self.end = (
            waypoints[1].transform.location.x,
            # FIXME
            waypoints[1].transform.location.y,
        )

        self.width = waypoints[0].lane_width / 2
        self.line = shapely.geometry.LineString([self.start, self.end])

    @property
    def contour(self):
        return self.line.buffer(self.s, cap_style=2)

    @property
    def boundaries(self):
        left = self.line.parallel_offset(self.s, side="left")
        right = self.line.parallel_offset(self.s, side="right")
        return (left, right)


class WaypointBoundaries:
    def __init__(self, waypoint: carla.Waypoint):
        self.waypoint = waypoint
        self.transform = waypoint.transform  # FIXME
        self.position = self.transform.location
        self.yaw = radians(self.transform.rotation.yaw)  # FIXME radians(...)
        self.width = waypoint.lane_width

        ox = -sin(self.yaw)
        oy = cos(self.yaw)
        self.left = (
            self.width / 2 * ox + self.position.x,
            self.width / 2 * oy + self.position.y,
        )
        self.right = (
            -self.width / 2 * ox + self.position.x,
            -self.width / 2 * oy + self.position.y,
        )


class Lane(object):
    def __init__(self, seed_point: carla.Waypoint, discretization_step: float = 0.1):
        self.segments = []
        self.seed = seed_point
        self.id = seed_point.lane_id
        self._init_lane_segments(distance=discretization_step)

    def _init_lane_segments(self, distance: float = 0.1):
        wbs = [WaypointBoundaries(wp) for wp in self.seed.next_until_lane_end(distance)]
        if len(wbs) < 2:

            self.left_segments = shapely.geometry.Point(wbs[0].left)
            self.middle_segments = shapely.geometry.Point(
                [wbs[0].position.x, wbs[0].position.y]
                # [wbs[0].position[0], wbs[0].position[1]] # FIXME
            )
            self.right_segments = shapely.geometry.Point(wbs[0].right)

            self.polygon = shapely.geometry.Polygon(
                [self.left_segments, self.middle_segments, self.right_segments]
            )
        else:
            l, m, r = [], [], []

            for w in wbs:
                l.append(w.left)
                m.append((w.position.x, w.position.y))
                r.append(w.right)

            self.left_segments = shapely.geometry.LineString(l)
            self.middle_segments = shapely.geometry.LineString(m)
            self.right_segments = shapely.geometry.LineString(r)

            poly = self.left_segments.coords[:]
            poly += self.right_segments.coords[::-1]
            self.polygon = shapely.geometry.Polygon(poly)
            """
            fig, ax = plt.subplots()
            p = PolygonPatch(self.polygon)
            x_min, y_min, x_max, y_max = self.polygon.bounds
            ax.set_xlim([x_min - 1, x_max + 1])
            ax.set_ylim([y_min - 1, y_max + 1])
            ax.add_patch(p)
            plt.show()
            """


class MapBridge:
    def __init__(self, world, waypoint_discretization: float = 0.05):
        self.map = world.get_map()
        self.waypoint_discretization = waypoint_discretization

        self.lane_topology = []
        self.lanes = []
        self.lane_polyons = shapely.geometry.MultiPolygon()

        self.str_tree = None

    def load_lane_polygons(self):
        self.lane_topology = self.map.get_topology()
        polys = []
        print("")
        for i, waypoint in enumerate(self.lane_topology, 1):
            print("processing lane %s/%s" % (i, len(self.lane_topology)), end="\r")
            lane = Lane(
                seed_point=waypoint[0], discretization_step=self.waypoint_discretization
            )
            self.lanes.append(lane)

            polys.append(lane.polygon)
        self.lane_polyons = shapely.geometry.MultiPolygon(polys)

        self.str_tree = shapely.strtree.STRtree(self.lane_polyons)

    def get_map_patch(self, box_dims, transform: Union[carla.Transform, Isometry]):
        # get all polygons in a bounding box
        # box_dims = [x, y]
        m_ = np.asarray(transform.get_matrix())
        coefficient_list = np.ravel(m_[:3, :3]).tolist()
        coefficient_list += np.ravel(m_[:3, 3]).tolist()

        query_box = shapely.geometry.box(
            -box_dims[0] / 2, -box_dims[1] / 2, box_dims[0] / 2, box_dims[1] / 2
        )
        query_box = shapely.affinity.affine_transform(query_box, coefficient_list)

        # get all polys in this box
        filtered_polygons = self.str_tree.query(query_box)
        filtered_polygons = [
            p.buffer(self.waypoint_discretization) for p in filtered_polygons
        ]
        union = shapely.ops.unary_union(filtered_polygons)
        union = union.buffer(-self.waypoint_discretization)

        return (query_box, union)

    def plot_polys(self, ax):
        for poly in self.lane_polyons:
            plot_polygon(ax, poly)

    def dump(self):
        pass

    def load(self, path):
        pass


if __name__ == "__main__":
    from shapely import speedups

    if speedups.available:
        speedups.enable()
        print("Speedups enabled")

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)  # seconds

    world = client.load_world("Town03")

    bridge = MapBridge(world, waypoint_discretization=0.05)
    bridge.load_lane_polygons()

    (x_min, y_min, x_max, y_max) = bridge.lane_polyons.bounds
    margin = 2
    fig, ax = plt.subplots(1, 2)
    for a in ax:
        a.set_xlim([x_min - margin, x_max + margin])
        a.set_ylim([y_min - margin, y_max + margin])

    bridge.plot_polys(ax[0])
    b, p = bridge.get_map_patch((30, 30), carla.Transform(carla.Location(x=80, y=-7.5)))
    (x_min, y_min, x_max, y_max) = b.bounds
    ax[1].set_xlim([x_min - margin, x_max + margin])
    ax[1].set_ylim([y_min - margin, y_max + margin])

    plot_polygon(
        ax[0],
        b,
        fc="blue",
        ec="blue",
        alpha=0.5,
    )

    plot_polygon(
        ax[1],
        p,
        fc="blue",
        ec="blue",
        alpha=0.5,
    )

    # union = shapely.ops.unary_union(bridge.lane_polyons)
    # plot_polygon(ax[1], union)

    # plt.scatter(x_, y_, s=1)
    plt.show()
    plt.savefig("fig")
