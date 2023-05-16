# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple agent-environment training loop."""
import torch
import tree
import itertools
import time
from typing import Optional
import tensorflow as tf
from acme import core
# Internal imports.
from acme.utils import counting
from acme.utils import loggers
import numpy as np

import dm_env

from misc import gumbel_softmax


class CustomEnvironmentLoop(core.Worker):
    """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, actor)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

    def __init__(
            self,
            environment,
            learner,
            DQN_network,
            summary_writer=None,
            chronics=None,
            max_ffw=None,
            # dn_ffw=None,
            # ep_infos=None,
            counter: counting.Counter = None,
            logger: loggers.Logger = None,
            label: str = 'environment_loop',
    ):
        # Internalize agent and environment.
        self._DQN_network = DQN_network
        self._environment = environment
        self._learner = learner
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(label)
        self.begin_time = time.time()
        # self.dn_ffw = dn_ffw
        # self.ep_infos = ep_infos
        self.chronics=chronics
        self.max_ffw=max_ffw
        self.summary_writer=summary_writer

    def compute_episode_score(self, chronic_id, agent_step, agent_reward, ffw=None):
        min_losses_ratio = 0.7
        ep_marginal_cost = self._environment.gen_cost_per_MW.max()
        if ffw is None:
            ep_do_nothing_reward = self.ep_infos[chronic_id]["donothing_reward"]
            ep_do_nothing_nodisc_reward = self.ep_infos[chronic_id]["donothing_nodisc_reward"]
            ep_dn_played = self.ep_infos[chronic_id]['dn_played']
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])
        else:
            start_idx = 0 if ffw == 0 else ffw * 288 - 2
            end_idx = start_idx + 864
            ep_dn_played, ep_do_nothing_reward, ep_do_nothing_nodisc_reward = self.dn_ffw[(chronic_id, ffw)]
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])[start_idx:end_idx]
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])[start_idx:end_idx]

        # Add cost of non delivered loads for blackout steps
        blackout_loads = ep_loads[agent_step:]
        if len(blackout_loads) > 0:
            blackout_reward = np.sum(blackout_loads) * ep_marginal_cost
            agent_reward += blackout_reward

        # Compute ranges
        worst_reward = np.sum(ep_loads) * ep_marginal_cost
        best_reward = np.sum(ep_losses) * min_losses_ratio
        zero_reward = ep_do_nothing_reward
        zero_blackout = ep_loads[ep_dn_played:]
        zero_reward += np.sum(zero_blackout) * ep_marginal_cost
        nodisc_reward = ep_do_nothing_nodisc_reward

        # Linear interp episode reward to codalab score
        if zero_reward != nodisc_reward:
            # DoNothing agent doesnt complete the scenario
            reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_reward, zero_reward, worst_reward]
            score_range = [100.0, 0.0, -100.0]

        ep_score = np.interp(agent_reward, reward_range, score_range)
        return ep_score

    def test(self, chronics, max_ffw):
        result = {}
        steps, scores = [], []

        if max_ffw == 5:
            chronics = chronics * 5
        for idx, i in enumerate(chronics):
            if max_ffw == 5:
                ffw = idx
            else:
                ffw = int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(max_ffw) if
                                     (i, fw) in self.dn_ffw and self.dn_ffw[(i, fw)][0] >= 10]))

            dn_step = self.dn_ffw[(i, ffw)][0]
            self._environment.seed(59)
            self._environment.set_id(i)
            obs = self._environment.reset()

            if ffw > 0:
                self._environment.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self._environment.step(self._environment.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            result[(i, ffw)] = {}
            while not done:
                observation = self._DQN_network.convert_obs(obs)
                act = self._learner.learner._network(torch.from_numpy(observation).reshape(1, -1))
                action = act.numpy().flatten()
                action = np.argmax(action)
                action_con = self._DQN_network.convert_act(action)
                obs, reward, done, info = self._environment.step(action_con)
                total_reward += reward
                alive_frame += 1
                if alive_frame == 864:
                    done = True

            l2rpn_score = float(self.compute_episode_score(i, alive_frame, total_reward, ffw))
            print(f'[Test Ch{i:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f} ')
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(i, ffw)]["real_reward"] = total_reward
            result[(i, ffw)]["reward"] = l2rpn_score
            result[(i, ffw)]["step"] = alive_frame

        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]['step']
            val_score += result[key]['reward']
            val_rew += result[key]['real_reward']
        stats = {
            'step': val_step / len(chronics),
            'score': val_score / len(chronics),
            'reward': val_rew / len(chronics)
        }
        self._logger.write(stats)
    
    
    def run(self, step: Optional[int] = None):
        """Perform the run loop.

    Run the environment loop for `num_episodes` episodes. Each episode is itself
    a loop which interacts first with the environment to get an observation and
    then give that observation to the agent in order to retrieve an action. Upon
    termination of an episode a new episode will be started. If the number of
    episodes is not given then this will interact with the environment
    infinitely.

    Args:
      num_episodes: number of episodes to run the loop for. If `None` (default),
        runs without limit.
    """

        avg_return = []
        avg_steps=[]
        max_ffw=self.max_ffw
        chronics=self.chronics
        # if max_ffw == 5:
        #     chronics = chronics * 5
        total_step=0
        # chronics=[1,1,1,1,1]
        start_time = time.time()
        counts=None
        for idx, i in enumerate(chronics):
            # Reset any counts and start the environment.
            self._environment.seed(59)
            self._environment.set_id(i)
            episode_steps = 0
            episode_return = 0
            done = False
            timestep = self._environment.reset()
            observation = self._DQN_network.convert_obs(timestep)

            # Run an episode
            while not done:
                # Generate an action from the agent's policy and step the environment.
                actions = self._learner.learner._policy_network(torch.from_numpy(observation).reshape(1, -1))
                action= gumbel_softmax(torch.from_numpy(actions.numpy()), hard=True).detach().numpy().flatten()
                action = np.argmax(action)
                action_con = self._DQN_network.convert_act(action)
                new_obs, reward, done, info = self._environment.step(action_con)
                observation = self._DQN_network.convert_obs(new_obs)
                episode_steps += 1
                if episode_steps==864:
                    done=True
                episode_return += reward
            total_step+=episode_steps
            avg_steps.append(episode_steps)
            avg_return.append(episode_return)

            # Record counts.
            counts = self._counter.increment(episodes=1, steps=episode_steps)

        # Collect the results and combine with counts.
        steps_per_second = total_step / (time.time() - start_time)
        mean_return = np.sum(avg_return) / total_step
        total_time_elapsed=time.time() - self.begin_time
        result = {
            'episode_length': total_step/(len(chronics)),
            'episode_return': avg_return,
            'avg_steps':avg_steps,
            'avg_return':np.mean(avg_return),
            'mean_return': mean_return,
            'steps_per_second': steps_per_second,
            'total_time_elapsed': total_time_elapsed
        }
        with self.summary_writer.as_default():
            # tf.summary.image('episode_return', avg_return, step=step)
            tf.summary.scalar('avg_return', np.mean(avg_return), step=step)
            tf.summary.scalar('avg_return_by_time', np.mean(avg_return), step=int(round(total_time_elapsed)))
            # tf.summary.image('avg_steps', avg_steps, step=step)
        result.update(counts)

        # Log the given results.
        self._logger.write(result)

# Internal class.
