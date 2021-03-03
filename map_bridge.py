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

import matplotlib.pyplot as plt
import shapely.geometry
import shapely.ops
import shapely.strtree
import shapely.affinity
from math import cos, sin, radians
from descartes.patch import PolygonPatch
import fiona
import numpy as np
import logging

from typing import Union, Tuple
from dataset_utilities.transformation import Isometry


def plot_line(ax, ob, color=None):
    x, y = ob.xy
    if color is not None:
        ax.plot(x, y, color=color)
    else:
        ax.plot(x, y)


def plot_polygon(ax, polygon, fc="green", ec="black", *args, **kwargs):
    if isinstance(polygon, shapely.geometry.Polygon):
        if polygon.is_empty:
            logging.warn("polygon empty")
            return

        patch = PolygonPatch(polygon, fc=fc, ec=ec, *args, **kwargs)
        ax.add_patch(patch)
    else:
        logging.warn(polygon)


class WaypointBoundaries:
    """
    WaypointBoundaries Class defines the road boundaries for each waypoint in the carla road topology.
    """

    def __init__(self, waypoint: carla.Waypoint):
        """
        Initialize a WaypointBoundary from a given carla.Waypoint.
        WaypointBoundaries include left and right borders of the road definition as well as the middle lane.
        """
        self.waypoint = waypoint
        self.transform = waypoint.transform  # FIXME
        self.position = self.transform.location
        self.yaw = radians(self.transform.rotation.yaw)  # FIXME radians(...)
        self.width = waypoint.lane_width

        ox = -sin(self.yaw)
        oy = cos(self.yaw)
        self.middle = (self.position.x, -self.position.y)
        self.left = (
            -self.width / 2 * ox + self.position.x,
            -self.width / 2 * oy + self.position.y,
        )
        self.right = (
            self.width / 2 * ox + self.position.x,
            self.width / 2 * oy + self.position.y,
        )


class Lane(object):
    def __init__(self, seed_point: carla.Waypoint, discretization_step: float = 0.1):
        """
        Lane object describing one lane starting from seed_point with discretization_step.

        Parameters:
        -----------
        seed_point: carla.Waypoint
            Starting point from which the lane originates

        discretization_step: float
            default: 0.1
            Sampling step of the lane.
        """
        self.segments = []
        self.seed = seed_point
        self.id = seed_point.lane_id
        self._init_lane_segments(distance=discretization_step)

    def _init_lane_segments(self, distance: float = 0.1):
        """
        get all waypoints along the current lane

        Parameters:
        -----------
        distance: float
            default: 0.1
            Discretization step along lane spline.
        """
        wbs = [WaypointBoundaries(wp) for wp in self.seed.next_until_lane_end(distance)]
        if len(wbs) < 2:

            self.left_segments = shapely.geometry.Point(wbs[0].left)
            self.middle_segments = shapely.geometry.Point(wbs[0].middle)
            self.right_segments = shapely.geometry.Point(wbs[0].right)

            self.polygon = shapely.geometry.Polygon(
                [self.left_segments, self.middle_segments, self.right_segments]
            )
        else:
            l, m, r = [], [], []

            for w in wbs:

                l.append(w.left)
                m.append((w.middle))
                r.append(w.right)

                # p_l = w.left
                # p_m = w.middle
                # p_r = w.right

            self.left_segments = shapely.geometry.LineString(l)
            self.middle_segments = shapely.geometry.LineString(m)
            self.right_segments = shapely.geometry.LineString(r)

            poly = self.left_segments.coords[:]
            poly += self.right_segments.coords[::-1]
            self.polygon = shapely.geometry.Polygon(poly)


class MapBridge:
    """
    The MapBridge class provides a interface to the carla lane definition
    """

    def __init__(self, world, waypoint_discretization: float = 0.05):
        """
        initialize a new instane of MapBridge

        Parameters:
        -----------
        world: carla.World
            carla world instance
        waypoint_discretization: float
            dicretization step size for polygon.
        """
        self.map = world.get_map()
        self.waypoint_discretization = waypoint_discretization

        self.lane_topology = []
        self.lanes = []
        self.lane_polyons = shapely.geometry.MultiPolygon()

        self.str_tree = None

    def load_lane_polygons(self):
        """
        load lane topology, discretizise along road definition
        initialize the lane polygon and STR tree

        Parameters:
        -----------
        """
        self.lane_topology = self.map.get_topology()
        polys = []
        for i, waypoint in enumerate(self.lane_topology, 1):
            lane = Lane(
                seed_point=waypoint[0], discretization_step=self.waypoint_discretization
            )
            self.lanes.append(lane)
            polys.append(lane.polygon)
            """
            logging.info(
                "processing lane %s/%s" % (i, len(self.lane_topology)),
                end="\r",
            )
            """

        self.lane_polyons = shapely.geometry.MultiPolygon(polys)
        self.str_tree = shapely.strtree.STRtree(self.lane_polyons)

    def get_map_patch(self, box_dims, world_T_veh):
        """
        get polygon within a defined box in vehicle frame

        Parameters:
        -----------
        box_dims: tuple
            width and height of ROI box
        world_T_veh: np.array
            transformation matrix. vehicle in world frame
        """
        # get all polygons in a bounding box
        # box_dims = [x, y]
        # m_ = world_T_veh.matrix
        # m_ = np.array(world_T_veh.get_matrix())
        m_ = world_T_veh
        coefficient_list = np.ravel(m_[:3, :3]).tolist()
        coefficient_list += np.ravel(m_[:3, 3]).tolist()

        # setup shapely box
        query_box = shapely.geometry.box(
            -box_dims[0] / 2, -box_dims[1] / 2, box_dims[0] / 2, box_dims[1] / 2
        )
        # transform box to world frame
        query_box = shapely.affinity.affine_transform(query_box, coefficient_list)

        # get all polys in this box
        filtered_polygons = self.str_tree.query(query_box)
        filtered_polygons = [
            p.buffer(self.waypoint_discretization) for p in filtered_polygons
        ]
        # fill gaps in the polygon by expandinng and then collapsing the polygon by a discretization step
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

    world = client.load_world("Town01")

    bridge = MapBridge(world, waypoint_discretization=0.05)
    bridge.load_lane_polygons()

    (x_min, y_min, x_max, y_max) = bridge.lane_polyons.bounds
    margin = 2
    fig, ax = plt.subplots(1, 2)
    for a in ax:
        a.set_xlim([x_min - margin, x_max + margin])
        a.set_ylim([y_min - margin, y_max + margin])

    bridge.plot_polys(ax[0])
    trafo = carla.Transform(carla.Location(x=160, y=55))

    s = world.get_spectator()
    s.set_transform(trafo)

    iso = Isometry.from_carla_transform(trafo)
    b, p = bridge.get_map_patch((40, 30), np.array(trafo.get_matrix()))
    print("area")
    print(p.area)
    # b, p = bridge.get_map_patch((40, 30), iso.matrix)
    (x_min, y_min, x_max, y_max) = b.bounds
    ax[1].set_xlim([x_min - margin, x_max + margin])
    ax[1].set_ylim([y_min - margin, y_max + margin])
    # ax[1].set_xlim([-20 - margin, 20 + margin])
    # ax[1].set_ylim([-15 - margin, 15 + margin])

    # convert to veh
    m_ = np.array(trafo.get_inverse_matrix())
    coefficient_list = np.ravel(m_[:3, :3]).tolist()
    coefficient_list += np.ravel(m_[:3, 3]).tolist()

    p = shapely.affinity.affine_transform(p, coefficient_list)

    (x_min, y_min, x_max, y_max) = p.bounds
    ax[1].set_xlim([x_min - margin, x_max + margin])
    ax[1].set_ylim([y_min - margin, y_max + margin])
    # print([y for y in p.exterior.coords])
    print([y for y in b.exterior.coords])

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
