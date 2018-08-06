#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from MyFunctions import ppo2_joint as ppo2j
import baselines.ppo2.policies as policies
import gym_remote.exceptions as gre

from retro_contest.local import make

def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    games = ['SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis',
             'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis',
             'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis',
             'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis',
             'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis', 'SonicTheHedgehog-Genesis']
    game_states = ['GreenHillZone.Act1', 'GreenHillZone.Act2', 'GreenHillZone.Act3',
                   'SpringYardZone.Act1', 'SpringYardZone.Act2', 'SpringYardZone.Act3',
                   'StarLightZone.Act1', 'StarLightZone.Act2', 'StarLightZone.Act3',
                   'LabyrinthZone.Act1', 'LabyrinthZone.Act2', 'LabyrinthZone.Act3',
                   'MarbleZone.Act1', 'MarbleZone.Act2', 'MarbleZone.Act3',
                   'ScrapBrainZone.Act1', 'ScrapBrainZone.Act2', 'ScrapBrainZone.Act3']

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2j.learn(policy=policies.CnnPolicy,
                   games=games,
                   game_states=game_states,
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.90,
                   noptepochs=3,
                   log_interval=5,
                   ent_coef=0.01,
                   save_interval=2,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.2,
                   total_timesteps=int(1e7),
                   load_model=False,
                   load_model_path="/home/carl/PycharmProjects/RetroCompetition/Joint_PPO2_Logs/checkpoints/00066",
                   save_model=True,
                   save_model_path="/home/carl/PycharmProjects/RetroCompetition/Joint_PPO2_Logs/v7")


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
