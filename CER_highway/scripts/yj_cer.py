import gym
import torch as th
import torch.nn as nn
import highway_env
from stable_baselines3.td3_ver1.td3_ver1 import TD3_ver1
from stable_baselines3 import TD3
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.td3_ver1.td3_ver2 import TD3_ver2

from stable_baselines3.td3_ver1.td3_ver3 import TD3_ver3
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

# kinematics observation with decision replay buffer


if __name__ == '__main__':
    Train = True
    env = gym.make("myhighway-v5")
    env.configure({
        "observation": {
            "type": "FrontDist1",
            # "type": "NoDecision1",
        },
        "action": {
            "type": "ContinuousAction",
            'steering_range': (-np.pi / 30, np.pi / 30),
            "dynamical": False,  # 먼저 학습 해 보고 잘되면 나중에 더하
        },
        "reward_speed_slope": [0, 2],
        "collision_reward": -10,  # The reward received when colliding with a vehicle.
        "target_lane_reward": 0.55,
        "jerk_v_reward": 0.1,
        "high_speed_reward": 0.35,
        "duration": 1000,  # 속력 20m/s이면 1200m가지 진행 (60sec)
        "random_road": True,
        "stop_speed": 15,
    })
    env.reset()

    # reward shaping from yj_td_decision3.py -> TD3_19, 각도 작게 만들고, lane 2개로 만들어서 앞에 다 막히는 상황

    model = TD3_ver3("MultiInputPolicy",
                env,
                # policy_kwargs=dict(
                #     net_arch=dict(qf=[512, 400, 300], pi=[512, 400, 300]),
                # ),
                verbose=1,
                buffer_size=1_000_000,
                train_freq=(1, "step"),
                # replay_buffer_class=HerReplayBuffer,
                # max_episode_length = 1000,
                learning_rate=1e-4, # actor : 1e-4, critic : 1e-5
                batch_size=100,
                # save_path='/Users/cml/highway-env/scripts/highway_td3_decisionbuffer/new_reward/model{}_{}',
                # save_interval=5e4,
                tensorboard_log="highway_td3_decisionbuffer/")
    if Train:
        # model_path = 'highway_td3_decisionbuffer/TD3_47/model'
        # model = TD3_ver2.load(model_path, env=env)

        model.learn(total_timesteps=int(5e6),tb_log_name='TD3_v2_1step', log_interval=20,)
        model.save("highway_td3_decisionbuffer/new_reward/model")
