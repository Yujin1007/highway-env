class MyHighwayEnv_Kinematic(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicDecision", #PotentialField  OccupancyGrid
            },
            "action": {
                "type": "ContinuousAction",

                # "dynamical": True, # 먼저 학습 해 보고 잘되면 나중에 더하
            },
            "policy_frequency": 10,
            "decision_frequency": 10,
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 4000000,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "target_lane_reward": 0.8,
            "direction_reward":0.1,
            "high_speed_reward": 0.2,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "jerk_reward": 0.05,
            "reward_speed_range": [20, 30],
            "offroad_terminal": True,
            "simulation_frequency" : 10
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._create_global_path()
        self.action_pre = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        self.anvel = 0
        self.anvel_pre = 0
        self.trajectory = []
        self.target_lane = self.global_path[1][0]
        self.allowed_lane = [self.target_lane]

        # x = 0
    def _create_global_path(self) -> None:
        """ create global path (reference path)"""
        LANE_LEN = self.controlled_vehicles[0].lane.length
        LANE_WID = self.controlled_vehicles[0].lane.width
        INIT_POS = np.append(self.controlled_vehicles[0].position, self.controlled_vehicles[0].lane_index[2])
        INTERVAL = 300.
        LANE_CNT = self.config['lanes_count']

        inter_x = np.arange(INIT_POS[0], LANE_LEN, INTERVAL)
        global_path = np.array([inter_x, np.zeros(inter_x.size)])
        self.create_global_path(global_path, 0, LANE_CNT, INIT_POS[2])
        self.global_path = global_path

    def create_global_path(self, global_path, idx, lanes, lane_id):
        global_path[1][idx] = lane_id
        idx += 1
        if idx == global_path.size * 0.5:
            return global_path
        else : #0,1,2,3 -> lanes = 4
            if lane_id == 0:
                # global_path[1][idx] = 1
                lane_id = 1
            elif lane_id == lanes-1:
                # global_path[1][idx] = lane_id - 1
                lane_id = lane_id - 1
            else:
                # global_path[1][idx] = random.choice([lane_id-1, lane_id+1])
                # lane_id = global_path[1][idx]
                lane_id = random.choice([lane_id-1, lane_id+1])
            self.create_global_path(global_path, idx, lanes, lane_id)




    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        width = self.road.network.graph['0']['1'][0].width
        self.anvel = action[1] - self.action_pre[1]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])


        jerk_v = abs(action[0] - self.action_pre[0])
        jerk_w = abs(self.anvel - self.anvel_pre)
        scaled_jerk_v = utils.lmap(jerk_v, [0, 10],[1, 0])
        scaled_jerk_w = utils.lmap(jerk_w, [0, np.pi],[1, 0])

        reward_c = self.config["collision_reward"] * self.vehicle.crashed
        reward_t = self.config["target_lane_reward"] * max(-1/(0.5*width)*np.abs(self.vehicle.position[1]-4*self.target_lane)+1,0)
        reward_s = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward_d = self.config["direction_reward"] * (self.obs['decision'][0]-1)*np.sign(action[1])
        reward_jw = self.config["jerk_reward"] * (scaled_jerk_w)
        reward_jv = self.config["jerk_reward"] * scaled_jerk_v
        reward = reward_c + reward_t + reward_s+reward_d + reward_jv + reward_jw
        # print(
        #     "T : {:.2f}, St : {:.2f}, Sp : {:.2f}, Jv : {:.2f}, Jw : {:.2f}".format(reward_t/self.config["target_lane_reward"], reward_d/self.config["direction_reward"],
        #                                                                                             reward_s/self.config["high_speed_reward"], reward_jv/self.config["jerk_reward"],
        #                                                                                             reward_jw/self.config["jerk_reward"]))
        #
        # print("T : {:.2f}, St : {:.2f}, Sp : {:.2f}, Jv : {:.2f}, Jw : {:.2f}, Total : {:.2f}".format(reward_t, reward_d , reward_s, reward_jv , reward_jw, reward))
        # reward = utils.lmap(reward,
        #                   [self.config["collision_reward"],
        #                    self.config["high_speed_reward"] + self.config["right_lane_reward"]],
        #                   [0, 1])

        reward = -1 if not self.vehicle.on_road or self.vehicle.position[0] < self.trajectory[0][0] or \
                       self.controlled_vehicles[0].lane_index[2] not in self.allowed_lane else reward

        self.action_pre = action
        self.anvel_pre = self.anvel

        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""

        if self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road) or \
            self.vehicle.position[0]<self.trajectory[0][0] or \
            self.controlled_vehicles[0].lane_index[2] not in self.allowed_lane:
            pass
        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road) or \
            self.vehicle.position[0]<self.trajectory[0][0] or \
            self.controlled_vehicles[0].lane_index[2] not in self.allowed_lane

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)
