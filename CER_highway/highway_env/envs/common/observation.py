from itertools import product
from typing import List, Dict, TYPE_CHECKING, Optional, Union, Tuple
from gym import spaces
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from matplotlib import cm
from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.lane import AbstractLane
from highway_env.utils import distance_to_circle, Vector
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.myhighway_utils import bring_positions, overtake_lane, get_index_list,front_dist,car_dist,front_vehicle
if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def __init__(self, env: 'AbstractEnv', **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):

    """
    An observation class that collects directly what the simulator renders.

    Also stacks the collected frames as in the nature DQN.
    The observation shape is C x W x H.

    Specific keys are expected in the configuration dictionary passed.
    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84)
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion,
        }
    """

    def __init__(self, env: 'AbstractEnv',
                 observation_shape: Tuple[int, int],
                 stack_size: int,
                 weights: List[float],
                 scaling: Optional[float] = None,
                 centering_position: Optional[List[float]] = None,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.shape = (stack_size, ) + self.observation_shape
        self.weights = weights
        self.obs = np.zeros(self.shape)

        # The viewer configuration can be different between this observation and env.render() (typically smaller)
        viewer_config = env.config.copy()
        viewer_config.update({
            "offscreen_rendering": True,
            "screen_width": self.observation_shape[0],
            "screen_height": self.observation_shape[1],
            "scaling": scaling or viewer_config["scaling"],
            "centering_position": centering_position or viewer_config["centering_position"]
        })
        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def space2(self) -> spaces.Space:
        # if self.as_image:
        #     return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        # else:
        #     return spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32)
        s = {
            'map': spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32),
            'decision':spaces.Discrte(1)
        }
        return spaces.Dict(s)

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :] = new_obs
        return self.obs

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()  # H x W x C
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.observe().shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros((3, 3, int(self.horizon * self.env.config["policy_frequency"])))
        grid = compute_ttc_grid(self.env, vehicle=self.observer_vehicle,
                                time_quantization=1/self.env.config["policy_frequency"], horizon=self.horizon)
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0:lf+1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0:vf + 1, :, :]
        return clamped_grid.astype(np.float32)


class KinematicObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = False,
                 observe_intentions: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

    def space(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


class OccupancyGridObservation(ObservationType):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy', 'on_road']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 grid_step: Optional[Tuple[float, float]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 align_to_vehicle_axes: bool = False,
                 clip: bool = True,
                 as_image: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step), dtype=np.int)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Initialize empty data
            self.grid.fill(np.nan)

            # Get nearby traffic data
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df.iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                        if "y" in self.features_range:
                            y = utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer, cell[1], cell[0]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)
                    # self.fill_road_layer_by_cell(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs



    def pos_to_index(self, position: Vector, relative: bool = False) -> Tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position
        return int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),\
               int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1]))

    def index_to_pos(self, index: Tuple[int, int]) -> np.ndarray:

        position = np.array([
            (index[1] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
            (index[0] + 0.5) * self.grid_step[1] + self.grid_size[1, 0]
        ])
        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(-self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(self, layer_index: int, lane_perception_distance: float = 100) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        Here, we iterate over lanes and regularly placed waypoints on these lanes to fill the corresponding cells.
        This approach is faster if the grid is large and the road network is small.

        :param layer_index: index of the layer in the grid
        :param lane_perception_distance: lanes are rendered +/- this distance from vehicle location
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(origin - lane_perception_distance,
                                            origin + lane_perception_distance,
                                            lane_waypoints_spacing).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer_index, cell[1], cell[0]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        In this implementation, we iterate the grid cells and check whether the corresponding world position
        at the center of the cell is onroad/offroad. This approach is faster if the grid is small and the road network large.
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: 'AbstractEnv', scales: List[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float64),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64),
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64),
            ))
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return {
            "observation": np.zeros((len(self.features),)),
            "achieved_goal": np.zeros((len(self.features),)),
            "desired_goal": np.zeros((len(self.features),))
        }

        obs = np.ravel(pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features])
        goal = np.ravel(pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features])
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales
        }
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: 'AbstractEnv', attributes: List[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict({
                attribute: spaces.Box(-np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float64)
                for attribute in self.attributes
            })
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            attribute: getattr(self.env, attribute) for attribute in self.attributes
        }


class MultiAgentObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_config: dict,
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.agents_observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


class TupleObservation(ObservationType):
    def __init__(self,
                 env: 'AbstractEnv',
                 observation_configs: List[dict],
                 **kwargs) -> None:
        super().__init__(env)
        self.observation_types = [observation_factory(self.env, obs_config) for obs_config in observation_configs]

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class ExitObservation(KinematicObservation):

    """Specific to exit_env, observe the distance to the next exit lane as part of a KinematicObservation."""

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind)
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs


class LidarObservation(ObservationType):
    DISTANCE = 0
    SPEED = 1

    def __init__(self, env,
                 cells: int = 16,
                 maximum_range: float = 60,
                 normalize: bool = True,
                 **kwargs):
        super().__init__(env, **kwargs)
        self.cells = cells
        self.maximum_range = maximum_range
        self.normalize = normalize
        self.angle = 2 * np.pi / self.cells
        self.grid = np.ones((self.cells, 1)) * float('inf')
        self.origin = None

    def space(self) -> spaces.Space:
        high = 1 if self.normalize else self.maximum_range
        return spaces.Box(shape=(self.cells, 2), low=-high, high=high, dtype=np.float32)

    def observe(self) -> np.ndarray:
        obs = self.trace(self.observer_vehicle.position, self.observer_vehicle.velocity).copy()
        if self.normalize:
            obs /= self.maximum_range
        return obs

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        self.origin = origin.copy()
        self.grid = np.ones((self.cells, 2)) * self.maximum_range

        for obstacle in self.env.road.vehicles + self.env.road.objects:
            if obstacle is self.observer_vehicle or not obstacle.solid:
                continue
            center_distance = np.linalg.norm(obstacle.position - origin)
            if center_distance > self.maximum_range:
                continue
            center_angle = self.position_to_angle(obstacle.position, origin)
            center_index = self.angle_to_index(center_angle)
            distance = center_distance - obstacle.WIDTH / 2
            if distance <= self.grid[center_index, self.DISTANCE]:
                direction = self.index_to_direction(center_index)
                velocity = (obstacle.velocity - origin_velocity).dot(direction)
                self.grid[center_index, :] = [distance, velocity]

            # Angular sector covered by the obstacle
            corners = utils.rect_corners(obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading)
            angles = [self.position_to_angle(corner, origin) for corner in corners]
            min_angle, max_angle = min(angles), max(angles)
            start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
            if start < end:
                indexes = np.arange(start, end+1)
            else:
                indexes = np.hstack([np.arange(start, self.cells), np.arange(0, end + 1)])

            # Actual distance computation for these sections
            for index in indexes:
                direction = self.index_to_direction(index)
                ray = [origin, origin + self.maximum_range * direction]
                distance = utils.distance_to_rect(ray, corners)
                if distance <= self.grid[index, self.DISTANCE]:
                    velocity = (obstacle.velocity - origin_velocity).dot(direction)
                    self.grid[index, :] = [distance, velocity]
        return self.grid

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        return np.arctan2(position[1] - origin[1], position[0] - origin[0]) + self.angle/2

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])


class PotentialFieldObservation(ObservationType):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy', 'on_road']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]



    def __init__(self,
                 env: 'AbstractEnv',
                 features: Optional[List[str]] = None,
                 grid_size: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 grid_step: Optional[Tuple[float, float]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 align_to_vehicle_axes: bool = False,
                 clip: bool = True,
                 as_image: bool = True,
                 min_ttc_front: float = 5.,
                 min_ttc_rear: float = 3.,
                 stack_size: int = 4,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step), dtype=np.int)
        pf_grid_shape = np.asarray((33,110),dtype=np.int)
        # pf_grid_shape = grid_shape
        # pf_grid_shape[0] = 8*4+1 # 차선폭/grid_step * 차선개수 + 1
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.pf_grid = np.zeros((stack_size, *pf_grid_shape))
        self.action_stack = np.ones(stack_size)
        self.stack_size = stack_size
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image
        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear
        self.ttc = 0
        self.decision = 'IDLE'
        self.target_lane = 0

        self.ref_pre = 0
        # self.shape = (stack_size, ) + tuple(grid_shape)
        #
        # self.shape = (stack_size,) + self.observation_shape
        #
        # self.observation_shape = observation_shape
        # self.shape = (stack_size,) + self.observation_shape
        # self.weights = weights
        # self.obs = np.zeros(self.shape)

    def space2(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=self.pf_grid.shape, low=0, high=255, dtype=np.uint8),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8)
        }
        return spaces.Dict(s)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) :
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Initialize empty data
            self.grid.fill(np.nan)


            # Get nearby traffic data
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            speed = self.observer_vehicle.speed
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df.iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                        if "y" in self.features_range:
                            y = utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        shape = self.shape_to_index((x, y), np.arccos(vehicle["cos_h"]), relative=not self.absolute)
                        if vehicle["y"] == 0:
                            ego_car = True
                        else:
                            ego_car = False
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            if ego_car:
                                self.grid[1, shape[1], shape[0]] = 1 #1+10*vehicle.vx
                            else:
                                self.grid[1, shape[1], shape[0]] = 0.5  # 1+10*vehicle.vx

                            self.grid[layer, cell[1], cell[0]] = vehicle[feature]


                # elif feature == "on_road":
                #     self.fill_road_layer_by_cell(layer)

                elif feature == "potential_field":
                    self.pf_grid[1:] = self.pf_grid[:-1]
                    self.fill_road_layer_by_lanes(layer)
                    idx = np.where(self.grid[1] == self.grid[1])
                    self.grid[layer, idx[0], idx[1]] = self.grid[1,idx[0],idx[1]]
                    idx = np.where(self.grid[layer,:,0]!=10)
                    if len(idx[0]) == self.pf_grid.shape[1]:
                        self.pf_grid[0] = self.grid[layer,idx,:]

            obs = self.grid



            obs = np.nan_to_num(obs).astype(self.space2().dtype)
            pf_obs = np.nan_to_num(self.pf_grid).astype(self.space().dtype)

            if self.clip:
                pf_obs = np.clip(pf_obs, -1, 1)

            if self.as_image:
                pf_obs = ((np.clip(pf_obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            """ 디버깅용 """
            """ 디버깅용 """
            EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                                [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])


            idx = np.where(self.env.global_path[0] <= EGO_POS[0])
            if len(idx[0]) == 0:
                REF_LANE = int(self.env.global_path[1][0])
            else:
                REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
            # if REF_LANE != self.ref_pre:
                # print(REF_LANE)
            # self.ref_pre = REF_LANE

            # if ((self.decision == 'IDLE') & (self.env.time % self.env.config['decision_frequency'] == 0)) \
            #     or ((self.decision != 'IDLE') & (self.env.controlled_vehicles[0].lane_index[2] == self.target_lane)):

            # if self.env.time % self.env.config['decision_frequency'] == 0:
            #     self.decision, self.target_lane = self._decision_module()
            #     self.env.target_lane = self.target_lane
            if self.decision == 'IDLE':
                self.decision, self.target_lane = self._decision_module()
                self.env.target_lane = self.target_lane
            else:
                if abs(EGO_POS[1]-self.target_lane*4) < 4*0.4: #도로 폭의 40%
                    self.decision = 'IDLE'

            if self.decision == 'IDLE':
                action = 1
                self.env.allowed_lane = [self.target_lane]
            elif self.decision == 'LEFT':
                action = 0
                self.env.allowed_lane = [self.target_lane, self.target_lane+1]
            elif self.decision == 'RIGHT':
                action = 2
                self.env.allowed_lane = [self.target_lane, self.target_lane-1]
            self.action_stack[1:] = self.action_stack[:-1]
            self.action_stack[0] = action
            observation = dict(map= pf_obs, decision=self.action_stack)
            # observation = observation.astype(self.space().dtype)

            # print("current : {},  target : {}, global : {},  decision : {}  ".format(int(EGO_POS[3]), self.env.target_lane, REF_LANE, self.decision))
            # if action == 2:
            #     plot_observation(observation['map'])

            return observation

    def shape_to_index(self, position: Vector, heading, relative: bool = False) -> Tuple[int, int]:
        WIDTH = 2
        LENGTH = 5 # 차량 크기 스펙

        # shape_x = np.linspace(position[0] - LENGTH / 2, position[0] + LENGTH / 2, LENGTH / self.grid_step[0])
        # shape_y = np.linspace(position[1] - WIDTH / 2, position[1] + WIDTH / 2, WIDTH / self.grid_step[1])

        shape_x = np.linspace(- LENGTH / 2, LENGTH / 2, int(LENGTH / self.grid_step[0])+1)
        shape_y = np.linspace(- WIDTH / 2, WIDTH / 2, int(WIDTH / self.grid_step[1])+1)

        shape_x, shape_y = np.meshgrid(shape_x,shape_y)

        shape_x = np.ravel(shape_x, order='C')
        shape_y = np.ravel(shape_y, order='C')
        shape = np.array([shape_x, shape_y])
        shape = np.transpose(shape)

        c, s = np.cos(heading), np.sin(heading)
        shape = shape.dot(np.array([[c, -s], [s, c]]))
        shape += np.array([position[0], position[1]])

        if not relative:
            shape -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(self.observer_vehicle.heading)
            shape = shape.dot(np.array([[c, -s], [s, c]]))
        shape = np.floor((shape - np.array([self.grid_size[1, 0],self.grid_size[0,0]])) / self.grid_step)
        shape = shape.astype(np.int_)
        shape = np.transpose(shape)
        idx = np.where((shape[0]>=0) & (shape[0]<self.grid.shape[-1]) & (shape[1]>=0) & (shape[1]<self.grid.shape[-2]))
        # if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:

        return (shape[0][idx], shape[1][idx])


    def pos_to_index(self, position: Vector, relative: bool = False) -> Tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position
        return int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),\
               int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1]))

    def index_to_pos(self, index: Tuple[int, int]) -> np.ndarray:

        position = np.array([
            (index[1] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
            (index[0] + 0.5) * self.grid_step[1] + self.grid_size[1, 0]
        ])
        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(-self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(self, layer_index: int, lane_perception_distance: float = 100) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        Here, we iterate over lanes and regularly placed waypoints on these lanes to fill the corresponding cells.
        This approach is faster if the grid is large and the road network is small.

        :param layer_index: index of the layer in the grid
        :param lane_perception_distance: lanes are rendered +/- this distance from vehicle location
        """
        LANE_WIDTH = 4
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        lane_cell = int(LANE_WIDTH*0.5/self.grid_step[0]) + 1
        self.grid[layer_index, :, :] = 10

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(origin - lane_perception_distance,
                                            origin + lane_perception_distance,
                                            lane_waypoints_spacing).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        cell_lines = np.array([cell[1] - 4 * 0.5 / self.grid_step[0], cell[1] + 4 * 0.5 / self.grid_step[0]])
                        cell_lines = cell_lines.astype(int)

                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.attractive_filed(cell, cell[1], layer_index)
                            for iter in range(1,lane_cell):
                                cell_lines = np.array([cell[1] - iter, cell[1] + iter])
                                cell_lines = cell_lines.astype(int)
                                if cell_lines[0] >= 0 and cell_lines[1] < self.grid.shape[-2]:
                                    self.attractive_filed(cell, cell_lines[0], layer_index)
                                    self.attractive_filed(cell, cell_lines[1], layer_index)
                                elif cell_lines[0] >= 0:
                                    self.attractive_filed(cell, cell_lines[0], layer_index)
                                elif cell_lines[1] < self.grid.shape[-2]:
                                    self.attractive_filed(cell, cell_lines[1], layer_index)


                            # self.grid[layer_index, cell[1], cell[0]] = 0.5
                            # # print('{}'.format(cell_lines[0]))
                            # self.grid[layer_index, cell_lines[0], cell[0]] = 1
                            # self.grid[layer_index, cell_lines[1], cell[0]] = 1


                            # self.grid[layer_index, cell[2], cell[0]] = 1
                            # self.grid[layer_index, cell[3], cell[0]] = 1


                            # print('{}....{}'.format(self.grid_step[0], self.grid_step[0]*10))
                            # self.grid[layer_index, cell[1]+4, cell[0]] = 1
                            # self.grid[layer_index, cell[1] - 4, cell[0]] = 1

                            '''
                            StraightLane.DEFAULT_WIDTH = 4 이므로 
                            '''
    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) infoprint("////rmation

        In this implementation, we iterate the grid cells and check whether the corresponding world position
        at the center of the cell is onroad/offroad. This approach is faster if the grid is small and the road network large.
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1

    def attractive_filed(self, cell, x, layer_index) -> None:
        Sigma = 0.8
        mu = cell[1]
        value = -1*math.exp(-(x-mu)**2 / 2*Sigma**2)
        '''
        grid 값 범위 : -1~1 
        '''
        self.grid[layer_index, x, cell[0]] = value

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle:
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]
                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]

                if front[0] - ego[0] < LENGTH * 1.5:
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]:
                        front_ttc = +1000
                    else:
                        front_ttc = (front[0] - ego[0])/-(front[2] - ego[2])


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])

            if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear):
                lane_change = True
            else:
                lane_change = False

            return front_ttc, rear_ttc, lane_change





    def _keep_lane(self, ego_pos, ref_lane) -> Tuple[str, int]:
        LANE_CNT = self.env.config['lanes_count']
        vehicles = bring_positions(self.env.road.vehicles[1:]) #vehicles : x, y, v, lane id
        front_id = np.where((vehicles[:, 3] == ego_pos[3]) & (vehicles[:, 0] > ego_pos[0]))[0][0]
        front = vehicles[front_id]
        dv = ego_pos[2] - front[2]

        if ego_pos[3] == ref_lane:
            if dv < 0:
                # ttc = -1 # inf
                ref_lane = ego_pos[3]
                decision = 'IDLE'
            else:
                self.ttc = (front[0] - ego_pos[0])/dv

                if self.ttc < self.min_ttc:
                    [decision, ref_lane] = overtake_lane(ego_pos, vehicles, LANE_CNT)
                else:
                    ref_lane = ego_pos[3]
                    decision = 'IDLE'


        return (decision, ref_lane)


    def _overtake_lane(self, ego_pos) -> Tuple[str, int]:
        LANE_CNT = self.env.config['lanes_count']
        vehicles = bring_positions(self.env.road.vehicles[1:]) #vehicles : x, y, v, lane id
        front_id = np.where((vehicles[:, 3] == ego_pos[3]) & (vehicles[:, 0] > ego_pos[0]))[0][0]
        front = vehicles[front_id]
        dv = ego_pos[2] - front[2]
        if dv < 0:
            # ttc = -1 # inf
            ref_lane = ego_pos[3]
            decision = 'IDLE'
        else:
            self.ttc = (front[0] - ego_pos[0])/dv

            if self.ttc < self.min_ttc:
                [decision, ref_lane] = overtake_lane(ego_pos, vehicles, LANE_CNT)
            else:
                ref_lane = ego_pos[3]
                decision = 'IDLE'

        return (decision, ref_lane)


class KinematicDecisionObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 5,
                 min_ttc_rear = 3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8)
        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten



        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
        else:
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.4:  # 도로 폭의 40%
                self.decision = 'IDLE'

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        observation = dict(map=obs.astype(self.space2().dtype), decision=self.action_stack)

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle:
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]
                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]

                if front[0] - ego[0] < LENGTH * 1.5:
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]:
                        front_ttc = +1000
                    else:
                        front_ttc = (front[0] - ego[0])/-(front[2] - ego[2])


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])

            if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear):
                lane_change = True
            else:
                lane_change = False

            return front_ttc, rear_ttc, lane_change


class KinematicDecisionObservation2(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy','cos_h', 'sin_h']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 5,
                 min_ttc_rear = 3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8)
        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten



        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
        else:
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.4:  # 도로 폭의 40%
                self.decision = 'IDLE'

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        observation = dict(map=obs.astype(self.space2().dtype), decision=self.action_stack)
        # observation = dict(map=obs.astype(self.space2().dtype), decision=np.array([action]))

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]

                if front[0] - ego[0] < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)
                        front_ttc = (front[0] - ego[0])/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear):
                    lane_change = True

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change

class NoDecisionObservation1(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy','heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 5,#3,#8,#5,
                 min_ttc_rear = 3,#2,#5,#3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            'center': spaces.Box( low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),
            'ttc': spaces.Box(low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),

        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten




        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))
        self.env.decision = action
        observation = dict(map=obs.astype(self.space2().dtype), center=self.center_stack, ttc=self.ttc_stack)
        # observation = dict(map=obs.astype(self.space2().dtype), decision=np.array([action]))

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                # if ego[0] - rear[0] < LENGTH * 1.5:
                if ego[0] - rear[0] < 20:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change

class MainObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy','heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front =6, #10,#6,#3,#
                 min_ttc_rear = 4,#8,#4,#2,#
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0


    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            'center': spaces.Box( low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8),
            'ttc': spaces.Box(low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),

        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten




        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))
        self.env.decision = action
        # print(action)
        observation = dict(map=obs.astype(self.space2().dtype), center=self.center_stack, decision=self.action_stack, ttc=self.ttc_stack)
        # observation = dict(map=obs.astype(self.space2().dtype), decision=np.array([action]))

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                # if ego[0] - rear[0] < LENGTH * 1.5:
                if ego[0] - rear[0] < 20:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change

class KinematicDecisionObservation4(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy','heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 5,#3,#8,#5,
                 min_ttc_rear = 3,#2,#5,#3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8),

        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten




        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))
        observation = dict(map=obs.astype(self.space2().dtype), decision=self.action_stack)

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                #if ego[0] - rear[0] < LENGTH * 1.5:
                if ego[0] - rear[0] < 20:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change

class DecisionObservation1(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy','heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 5,#3,#8,#5,
                 min_ttc_rear = 3,#2,#5,#3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(1, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'kinematic': spaces.Box(shape=(1, len(self.features)), low=-np.inf, high=np.inf,dtype=np.float32),
            'center': spaces.Box( low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8),
            'ttc': spaces.Box(low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),
        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        # close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
        #                                                  self.env.PERCEPTION_DISTANCE,
        #                                                  count=self.vehicles_count - 1,
        #                                                  see_behind=self.see_behind,
        #                                                  sort=self.order == "sorted")
        # if close_vehicles:
        #     origin = self.observer_vehicle if not self.absolute else None
        #     df = df.append(pd.DataFrame.from_records(
        #         [v.to_dict(origin, observe_intentions=self.observe_intentions)
        #          for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
        #                    ignore_index=True)
        # # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # # Fill missing rows
        # if df.shape[0] < self.vehicles_count:
        #     rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
        #     df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten




        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))
        observation = dict(kinematic=obs.astype(self.space2().dtype),center=self.center_stack, decision=self.action_stack, ttc=self.ttc_stack)
        # observation = dict(map=obs.astype(self.space2().dtype), decision=np.array([action]))

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change
class MPCObservation_v0(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['x', 'y', 'vx', 'vy']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 8,#3,#8,#5,
                 min_ttc_rear = 8,#2,#5,#3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(1, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8),
            # 'surroundings': spaces.Box(shape=(self.vehicles_count-1, 3), low=-np.inf, high=np.inf, dtype=np.float32),
            'surroundings': spaces.Box(shape=(1, 3), low=-np.inf, high=np.inf, dtype=np.float32)
        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        data = self.observer_vehicle.to_dict()
        features = []
        for f in self.features:
            if f in data.keys():
                features.append(f)


        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         # see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        # if close_vehicles:
        #     origin = self.observer_vehicle if not self.absolute else None
        #     df = df.append(pd.DataFrame.from_records(
        #         [v.to_dict(origin, observe_intentions=self.observe_intentions)
        #          for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
        #                    ignore_index=True)
        # # Normalize and clip
        # if self.normalize:
        #     df = self.normalize_obs(df)
        # # Fill missing rows
        # if df.shape[0] < self.vehicles_count:
        #     rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
        #     df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # # Reorder
        # df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten

        # obs_data = np.zeros([self.vehicles_count-1,3])
        p = []
        for v in close_vehicles:
            p.append(pd.DataFrame.from_records([v.to_dict()])[['x', 'y', 'speed']])
        # for i, vehicle in enumerate(p):
        #     obs_data[i] = vehicle.values.copy()


        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])



        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        obs_data = front_vehicle(EGO_POS, close_vehicles, self.target_lane * 4)
        # obs_data[3] = obs_data[0] - EGO_POS[0] - 1.698 * EGO_POS[2]
        obs_data[3] = obs_data[3] = obs_data[0] - self.env.controlled_vehicles[0].position[0]
        self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))

        obs = np.append(obs, self.env.target_lane*4)
        observation = dict(map=obs, decision=self.action_stack, surroundings=obs_data)
        # observation = dict(map=obs.astype(self.space2().dtype), decision=np.array([action]))
        self.env.decision = action
        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change
class MPCObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['x', 'y', 'heading', 'speed', 'target_y']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 5,#3,#8,#5,
                 min_ttc_rear = 3,#2,#5,#3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(1, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            # 'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8),
            # 'surroundings': spaces.Box(shape=(self.vehicles_count-1, 3), low=-np.inf, high=np.inf, dtype=np.float32),
            'surroundings': spaces.Box(shape=(1, 3), low=-np.inf, high=np.inf, dtype=np.float32)
        }
        return spaces.Dict(s)


    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        data = self.observer_vehicle.to_dict()
        features = []
        for f in self.features:
            if f in data.keys():
                features.append(f)


        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         # see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        # if close_vehicles:
        #     origin = self.observer_vehicle if not self.absolute else None
        #     df = df.append(pd.DataFrame.from_records(
        #         [v.to_dict(origin, observe_intentions=self.observe_intentions)
        #          for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
        #                    ignore_index=True)
        # # Normalize and clip
        # if self.normalize:
        #     df = self.normalize_obs(df)
        # # Fill missing rows
        # if df.shape[0] < self.vehicles_count:
        #     rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
        #     df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # # Reorder
        # df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten

        # obs_data = np.zeros([self.vehicles_count-1,3])
        p = []
        for v in close_vehicles:
            p.append(pd.DataFrame.from_records([v.to_dict()])[['x', 'y', 'speed']])
        # for i, vehicle in enumerate(p):
        #     obs_data[i] = vehicle.values.copy()


        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])



        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            # self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        obs_data = front_vehicle(EGO_POS, close_vehicles, self.target_lane * 4)
        # obs_data = front_vehicle(EGO_POS, close_vehicles)
        # obs_data[3] = obs_data[0] - EGO_POS[0] - 1.698 * EGO_POS[2]
        obs_data[3] = obs_data[3] = obs_data[0] - self.env.controlled_vehicles[0].position[0]
        # self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            # self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            # self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            # self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))

        obs = np.append(obs, self.target_lane*4)
        # self.env.target_lane = self.target_lane

        observation = np.concatenate([obs.reshape(-1, 1), obs_data.reshape(-1, 1)], axis=0)
        # 이것 때문에 바깥에 target lane이랑 decision이 바뀐다.
        self.env.decision = action
        self.env.target_lane = self.target_lane
        self.env.observation_type.decision = self.decision
        self.env.observation_type.target_lane = self.target_lane
        if self.target_lane == 2:
            print(self.env.target_lane)
        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""

        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change

class MPCObservation_Simple(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['x', 'y', 'heading', 'speed', 'target_y']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front = 6,#3,#8,#5,
                 min_ttc_rear = 4,#2,#5,#3,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        # self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        # self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    # def space(self) -> spaces.Space:
    #     s = {
    #         'map': spaces.Box(shape=(1, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
    #         # 'decision': spaces.Box( low=np.zeros(4), high=np.ones(4)*2, dtype=np.uint8),
    #         # 'surroundings': spaces.Box(shape=(self.vehicles_count-1, 3), low=-np.inf, high=np.inf, dtype=np.float32),
    #         'surroundings': spaces.Box(shape=(1, 3), low=-np.inf, high=np.inf, dtype=np.float32)
    #     }
    #     return spaces.Dict(s)


    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        data = self.observer_vehicle.to_dict()
        features = []
        for f in self.features:
            if f in data.keys():
                features.append(f)


        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         # see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        # if close_vehicles:
        #     origin = self.observer_vehicle if not self.absolute else None
        #     df = df.append(pd.DataFrame.from_records(
        #         [v.to_dict(origin, observe_intentions=self.observe_intentions)
        #          for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
        #                    ignore_index=True)
        # # Normalize and clip
        # if self.normalize:
        #     df = self.normalize_obs(df)
        # # Fill missing rows
        # if df.shape[0] < self.vehicles_count:
        #     rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
        #     df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # # Reorder
        # df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten

        # obs_data = np.zeros([self.vehicles_count-1,3])
        p = []
        for v in close_vehicles:
            p.append(pd.DataFrame.from_records([v.to_dict()])[['x', 'y', 'speed']])
        # for i, vehicle in enumerate(p):
        #     obs_data[i] = vehicle.values.copy()


        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        # MPC에서 target lane이랑 ref를 또 계산할 필요 없다 .. ?
        #
        # if self.decision == 'IDLE':
        #     From = self.target_lane
        #     self.decision, self.target_lane = self._decision_module()
        #     # self.env.target_lane = self.target_lane
        #     if self.decision != 'IDLE':
        #         # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
        #         if EGO_POS[1] < self.target_lane * 4:
        #             self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
        #         else:
        #             self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
        #         self.cnt = 1
        # else:
        #     self.cnt += 1
        #     if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.40:  # 도로 폭의 25%
        #         self.decision = 'IDLE'
        #
        #
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        obs_data = front_vehicle(EGO_POS, close_vehicles, self.target_lane * 4)
        # obs_data = front_vehicle(EGO_POS, close_vehicles)
        # obs_data[3] = obs_data[0] - EGO_POS[0] - 1.698 * EGO_POS[2]
        obs_data[3] = obs_data[3] = obs_data[0] - self.env.controlled_vehicles[0].position[0]
        # self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            # self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            # self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            # self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))

        obs = np.append(obs, self.env.target_lane*4)

        observation = np.concatenate([obs.reshape(-1, 1), obs_data.reshape(-1, 1)], axis=0)
        # 이것 때문에 바깥에 target lane이랑 decision이 바뀐다.
        # self.env.decision = action
        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                if ego[0] - rear[0] < LENGTH * 1.5:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change



class SimpleObservation(ObservationType):

    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy','heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 5,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = True,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 min_ttc_front =6,
                 min_ttc_rear = 4,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.decision = 'IDLE'
        self.action_stack = np.ones(4)
        self.center_stack = np.zeros(4)
        self.ttc_stack = np.ones(4)

        self.min_ttc_front = min_ttc_front
        self.min_ttc_rear = min_ttc_rear

        self.target_lane = 0
        self.lc_reward = np.zeros(8)
        self.cnt = 0

    def space2(self) -> spaces.Space:
        return spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        s = {
            'map': spaces.Box(shape=(self.vehicles_count, len(self.features)), low=-np.inf, high=np.inf, dtype=np.float32),
            # 'center': spaces.Box( low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),
            'decision': spaces.Box( low=np.zeros(1), high=np.ones(1)*2, dtype=np.uint8),
            # 'ttc': spaces.Box(low=np.zeros(4), high=np.ones(4)*16, dtype=np.float_),

        }
        return spaces.Dict(s)

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(self.observer_vehicle.lane_index)
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [-AbstractLane.DEFAULT_WIDTH * len(side_lanes), AbstractLane.DEFAULT_WIDTH * len(side_lanes)],
                "vx": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED],
                "vy": [-2*Vehicle.MAX_SPEED, 2*Vehicle.MAX_SPEED]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(self.observer_vehicle,
                                                         self.env.PERCEPTION_DISTANCE,
                                                         count=self.vehicles_count - 1,
                                                         see_behind=self.see_behind,
                                                         sort=self.order == "sorted")
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = df.append(pd.DataFrame.from_records(
                [v.to_dict(origin, observe_intentions=self.observe_intentions)
                 for v in close_vehicles[-self.vehicles_count + 1:]])[self.features],
                           ignore_index=True)
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = df.append(pd.DataFrame(data=rows, columns=self.features), ignore_index=True)
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten




        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])

        if self.decision == 'IDLE':
            From = self.target_lane
            self.decision, self.target_lane = self._decision_module()
            self.env.target_lane = self.target_lane
            if self.decision != 'IDLE':
                # self.lc_reward = np.linspace(From*4, self.target_lane*4, 8)
                if EGO_POS[1] < self.target_lane * 4:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, 0.5)
                else:
                    self.lc_reward = np.arange(EGO_POS[1], self.target_lane * 4, -0.5)
                self.cnt = 1
        else:
            self.cnt += 1
            if abs(EGO_POS[1] - self.target_lane * 4) < 4 * 0.25:  # 도로 폭의 25%
                self.decision = 'IDLE'
        # print(self.decision, self.cnt)
        # print(self.lc_reward)

        self.env.lc_reward = self.lc_reward[min(self.cnt, len(self.lc_reward)-1)]

        if self.decision == 'IDLE':
            action = 1
            self.env.allowed_lane = [self.target_lane]
        elif self.decision == 'LEFT':
            action = 0
            self.env.allowed_lane = [self.target_lane, self.target_lane + 1]
        elif self.decision == 'RIGHT':
            action = 2
            self.env.allowed_lane = [self.target_lane, self.target_lane - 1]
        self.action_stack[1:] = self.action_stack[:-1]
        self.action_stack[0] = action

        self.center_stack[1:] = self.center_stack[:-1]
        self.center_stack[0] = self.target_lane - EGO_POS[1]/4
        self.ttc_stack[1:] = self.ttc_stack[:-1]
        front_distance = front_dist(EGO_POS, self.env.road.vehicles[1:])*0.01
        self.ttc_stack[0] = np.clip(front_distance,0,1) # 0~1사이의 값으로 scaling
        # print("{:.2f}, {:.2f}".format(self.ttc_stack[0], front_distance))
        self.env.decision = action
        # print(action)
        # observation = dict(map=obs.astype(self.space2().dtype), center=self.center_stack, decision=self.action_stack, ttc=self.ttc_stack)
        observation = dict(map=obs.astype(self.space2().dtype), decision=np.array([action]))

        return observation

    def _decision_module(self) -> Tuple[str, int]:
        """make decision of the scene : idle / lc left / lc right"""
        """triggered at longer period than env step"""
        """현재 차량의 속도, 레퍼런스 속도, 전방 차량정보, 측방 차량 정보, global path"""
        EGO_POS = np.append(self.env.controlled_vehicles[0].position,
                            [self.env.controlled_vehicles[0].speed, self.env.controlled_vehicles[0].lane_index[2]])
        idx = np.where(self.env.global_path[0] <= EGO_POS[0])

        if len(idx[0]) == 0:
            REF_LANE = int(self.env.global_path[1][0])
        else:
            REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])
        # REF_LANE = int(self.env.global_path[1][int(idx[0][-1])])


        front_ttc, rear_ttc, flag_idle = self.calc_ttc(EGO_POS, EGO_POS[3])
        front_left_ttc, rear_left_ttc, flag_left = self.calc_ttc(EGO_POS, EGO_POS[3]-1)
        front_right_ttc, rear_right_ttc, flag_right = self.calc_ttc(EGO_POS, EGO_POS[3]+1)

        # priority : idle
        if REF_LANE == EGO_POS[3]:
            LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3] + 1, EGO_POS[3]]
            DECISION = ['LEFT', 'RIGHT', 'IDLE']

            if flag_idle or (not flag_idle and not flag_left and not flag_right):
                decision = 'IDLE'
                ref = REF_LANE

            else:
                flag = [flag_left,flag_right]
                ttc = [front_left_ttc, front_right_ttc]

                p = [i for i in range(2) if flag[i] is True]
                if len(p) >= 1:
                    m = np.argmax(get_index_list(ttc,p))
                    m = p[m]
                # elif len(p) == 1:
                #     m = p[0]
                else:
                    m = 2
                decision = DECISION[m]
                ref = LANE_NOM[m]

        #priority : lane change
        else:
            if REF_LANE < EGO_POS[3]:
                flag = [flag_left, flag_idle, flag_right] # 우선 순위
                ttc = [front_left_ttc, front_ttc, front_right_ttc]
                LANE_NOM = [EGO_POS[3] - 1, EGO_POS[3], EGO_POS[3] + 1]
                DECISION = ['LEFT', 'IDLE', 'RIGHT']
            else:
                flag = [flag_right, flag_idle, flag_left]
                ttc = [front_right_ttc, front_ttc, front_left_ttc]
                LANE_NOM = [EGO_POS[3] + 1, EGO_POS[3], EGO_POS[3] - 1]
                DECISION = ['RIGHT', 'IDLE', 'LEFT']

            if flag[0] is True:
                m = 0
            elif flag[1] is True:
                m = 1
            elif flag[2] is True:
                m = 2
            else:
                m = 1


            # p = [i for i in range(3) if flag[i] is True]
            # if len(p) >= 1:
            #     m = np.argmax(get_index_list(ttc, p))
            #     m = p[m]
            # else:
            #     m = 2
            decision = DECISION[m]
            ref = LANE_NOM[m]



        return (decision, ref)




    def calc_ttc(self, ego, lane) -> Tuple[float, float, bool]:
        LANE_CNT = self.env.config['lanes_count']
        LENGTH = self.env.controlled_vehicles[0].LENGTH
        vehicles = bring_positions(self.env.road.vehicles[1:])  # vehicles : x, y, v, lane id

        LANE_KEEP = True if ego[3] == lane else False
        RIGHT = True if ego[3] + 1 == lane else False
        LEFT = True if ego[3] - 1 == lane else False
        dx = 100

        if lane < 0:
            return None, None, False
        elif lane >= LANE_CNT:
            return None, None, False
        else:
            front_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] > ego[0]))[0]
            if len(front_id) == 0:
                front_ttc = +1000
            else:
                front = vehicles[front_id[0]]
                dx = (front[0] - ego[0])
                if dx < LENGTH * 3: #2->3 : 차량 길이를 생각 안함. 계산 되는 차량의 위치는 차량의 중앙점,
                                                    # 3 이면 두 차 사이 거리가 8m (내차(2m), 앞차(2m) 앞 뒤 길이 뺀 값)
                    front_ttc = -1
                else:
                    if front[2] >= ego[2]: # 선행차량의 속도가 1m/s이상 빠르면 안전하다 판단
                        front_ttc = +1000
                    else:
                        dv = min(abs(front[2] - ego[2]), 10)

                        front_ttc = dx/dv


            rear_id = np.where((vehicles[:, 3] == lane) & (vehicles[:, 0] <= ego[0]))[0]
            if len(rear_id) == 0:
                rear_ttc = +1000
            else:
                rear = vehicles[rear_id[-1]]
                # if ego[0] - rear[0] < LENGTH * 1.5:
                if ego[0] - rear[0] < 20:
                    rear_ttc = -1
                else:
                    if rear[2] <= ego[2]:
                        rear_ttc = +1000
                    else:
                        rear_ttc = (ego[0] - rear[0]) / (rear[2] - ego[2])


            if LANE_KEEP:
                if (front_ttc > self.min_ttc_front-1) :
                    lane_keep = True

                else:
                    lane_keep = False

                return front_ttc, rear_ttc, lane_keep
            else:

                if (front_ttc > self.min_ttc_front) & (rear_ttc > self.min_ttc_rear) & (dx >= 25):
                    lane_change = True
                    if LEFT:
                        lc = np.where(self.action_stack == 2)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False
                    if RIGHT:
                        lc = np.where(self.action_stack == 0)  # 최근 4 step안에 오른쪽으로 이동한 경로가 있는지
                        if len(lc[0]) > 0:
                            lane_change = False

                else:
                    lane_change = False


                return front_ttc, rear_ttc, lane_change


def plot_observation(observation):
    ob = observation.shape
    y = np.linspace(0,ob[1]-1,ob[1])
    x = np.linspace(0,ob[2]-1,ob[2])
    X,Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=plt.figaspect(0.25))

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    surf = ax.plot_surface(X, Y, observation[0], rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    surf = ax.plot_surface(X, Y, observation[1], rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    surf = ax.plot_surface(X, Y, observation[2], rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    surf = ax.plot_surface(X, Y, observation[3], rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()
    # time.sleep(0.1)

def observation_factory(env: 'AbstractEnv', config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    elif config["type"] == "PotentialField":
        return PotentialFieldObservation(env, **config)
    elif config["type"] == "KinematicDecision":
        return KinematicDecisionObservation(env, **config)
    elif config["type"] == "KinematicDecision2":
        return KinematicDecisionObservation2(env, **config)
    elif config["type"] == "NoDecision1":
        return NoDecisionObservation1(env, **config)
    elif config["type"] == "FrontDist1":
        return MainObservation(env, **config)

    elif config["type"] == "KinematicDecision4":
        return KinematicDecisionObservation4(env, **config)

    elif config["type"] == "Decision1":
        return DecisionObservation1(env, **config)

    elif config["type"] == "MPCObservation":
        return MPCObservation(env, **config)
    elif config["type"] == "MPCObservation_v0":
        return MPCObservation_v0(env, **config)
    elif config["type"] == "Simple":
        return SimpleObservation(env, **config)

    else:

        raise ValueError("Unknown observation type")
