import json
import os
import sys
import torch
import pprint
from collections import OrderedDict
from typing import Mapping, Sequence, Optional
import sys

from acme.wrappers import base
import tree

from absl import app
from absl import flags
import acme
import reverb

from DQN import DQN

try:
    import tensorflow as tf

    _CAN_USE_TENSORFLOW = True
except ImportError:
    _CAN_USE_TENSORFLOW = False
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.agents.tf import d4pg
from acme.tf import networks
from acme.tf.networks import duelling
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import dm_env
import gym
import numpy as np
import sonnet as snt

from multiprocessing import Process
import threading
import argparse

from acme.wrappers import gym_wrapper
from acme.wrappers import atari_wrapper
from acme.tf.networks import base

import bsuite
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

Images = tf.Tensor
QValues = tf.Tensor
Logits = tf.Tensor
Value = tf.Tensor


def get_actor_sigma(sigma_max, actor_id, n_actors):
    sigmas = list(np.arange(0, sigma_max, (sigma_max - 0) / n_actors))
    print(sigmas)
    return sigmas[actor_id - 1]


def Q_compress(W, n):
    assert (n == 8)

    W = W.numpy()
    W_orig = W
    if n >= 32:
        return W
    assert (len(W.shape) <= 2)
    range = np.max(W) - np.min(W)
    d = range / (2 ** (n - 1))
    if d == 0:
        return W
    z = -np.min(W, 0) // d
    W = np.rint(W / d)

    W_q = torch.from_numpy(W).char()
    return d, W_q


def Q_decompress(V, n):
    return V[0] * V[1].float()


def Q(W, n):
    if n >= 32:
        return W
    assert (len(W.shape) <= 2)
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2 ** (n))
    if d == 0:
        return W
    z = -np.min(W, 0) // d
    W = np.rint(W / d)
    W = d * (W)
    return W


def Q_opt(W, n, intervals=100):
    return Q(W, n)
    if n == 32:
        return Q(W, n)
    best = Q(W, n)
    best_err = np.mean(np.abs((best - W)).flatten())
    first_err = best_err
    minW, maxW = np.min(W), np.max(W)
    max_abs = max(abs(minW), abs(maxW))
    for lim in np.arange(0, max_abs, max_abs / intervals):
        W_clipped = np.clip(W, -lim, lim)
        W_clipped_Q = Q(W_clipped, n)
        mse = np.mean(np.abs((W_clipped_Q - W)).flatten())
        if mse < best_err:
            # print("New best err: (%f->%f) at clip %f (W_min=%f, W_max=%f)" % (best_err, mse, lim, minW, maxW))
            best_err = mse
            best = W_clipped_Q
    print("Opted: %f->%f err" % (first_err, best_err))
    return best


def input_size_from_obs_spec(env_spec):
    if hasattr(env_spec, "shape"):
        return int(np.prod(env_spec.shape))
    if type(env_spec) == OrderedDict:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec.values()]))
    try:
        return int(sum([input_size_from_obs_spec(x) for x in env_spec]))
    except:
        assert (0)


def input_from_obs(observation):
    observation = tf2_utils.add_batch_dim(observation)
    observation = tf2_utils.batch_concat(observation)
    return tf2_utils.to_numpy(observation)


# The default settings in this network factory will work well for the
# MountainCarContinuous-v0 task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.


class DQNnetwork(AgentWithConverter):
    def __init__(self, env, action_space):
        # Call parent constructor
        if not _CAN_USE_TENSORFLOW:
            raise RuntimeError("Cannot import tensorflow, this function cannot be used.")
        AgentWithConverter.__init__(self, action_space,
                                    action_space_converter=IdToAct)
        self.env = env
        self.action_space.filter_action(self._filter_action)
        self.action_size = self.action_space.size()
        observation = self.env.reset()
        self.observation_size = self.convert_obs(observation).shape[0]
        # self.obs_space = observation_space
        # self.observation_size = self.obs_space.size_obs()

    # def observation_to_vect(self):
    #     feats_bus_load_pos = self.obs_space['feats_bus_load_pos']
    #     sub_load = env.backend._pglib_sub_infos["load_p"][feats_bus_load_pos] / 20e3  # 获取各个子站点的负载情况
    #     sub_load_vec = sub_load[0].reshape(-1, )  # 将负载数据拼接为向量形式
    #
    #     line_or_status = self.obs_space['line_or_status'].astype(np.float32)
    #     line_connect_vec = line_or_status.reshape(-1, )  # 将输电线状态数据拼接为向量形式
    #
    #     angle = self.obs_space['angle'] / np.pi  # 归一化后的相角数据
    #     angle_vec = angle.reshape(-1, )  # 将相角数据拼接为向量形式
    #
    #     obs_vec = np.concatenate([sub_load_vec, line_connect_vec, angle_vec])

    def _filter_action(self, action):
        MAX_ELEM = 2
        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= MAX_ELEM:
            return True
        return False

    def convert_act(self, action):
        return super().convert_act(action)

    def make_networks(
            self,
            policy_layer_sizes: Sequence[int] = (2048, 2048, 2048),
            critic_layer_sizes: Sequence[int] = (64, 64, 64),
            vmin: float = -150.,
            vmax: float = 150.,
            num_atoms: int = 51,
            placement: str = "CPU",
    ) -> Mapping[str, types.TensorTransformation]:
        """Creates the networks used by the agent."""

        with tf.device(placement):
            # Get total number of action dimensions from action spec.
            num_dimensions = self.action_size

            # Create the shared observation network; here simply a state-less operation.
            observation_network = tf2_utils.batch_concat

            uniform_initializer = tf.initializers.VarianceScaling(
                distribution='uniform', mode='fan_out', scale=0.333)
            policy_network = snt.Sequential([
                snt.nets.MLP(
                    policy_layer_sizes,
                    w_init=uniform_initializer,
                    activation=tf.nn.relu,
                    activate_final=False),
                snt.nets.MLP(
                    [num_dimensions],
                    w_init=uniform_initializer,
                    activate_final=True,
                    activation=tf.nn.tanh)
            ])

            # Create the critic network.
            critic_network = snt.Sequential([
                # The multiplexer concatenates the observations/actions.
                networks.CriticMultiplexer(),
                networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
                networks.DiscreteValuedHead(vmin, vmax, num_atoms),
            ])

            return {
                'policy': policy_network,
                'critic': critic_network,
                'observation': observation_network,
            }

    def my_act(self, transformed_observation, reward, done=False):
        pass

    # def convert_obs(self,observation):
    #     li_vect = []
    #     for el in observation.attr_list_vect:
    #         v = observation._get_array_from_attr_name(el).astype(np.float32)
    #         v_fix = np.nan_to_num(v)
    #         v_norm = np.linalg.norm(v_fix)
    #         if v_norm > 1e6:
    #             v_res = (v_fix / v_norm) * 10.0
    #         else:
    #             v_res = v_fix
    #         li_vect.append(v_res)
    #     return np.concatenate(li_vect)

    def convert_obs(self, observation):
        # 对需要转换的特征进行转换
        # feat1 = np.ravel(observation.gen_p) / self.env.gen_pmax
        # feat2 = np.ravel(observation.gen_q) / self.env.gen_pmax
        # feat3 = np.ravel(observation.gen_v) / np.array([142.1, 142.1, 22.0, 22.0,13.2, 142.1])  # 转换负荷有功
        # feat4 = np.ravel(observation.load_p) / np.array(
        #     [26.7, 112.3, 63.1, 9.2, 13.3, 33.2, 11.1, 4.4, 7.5, 16.5, 17.8])  # 转换负荷无功
        # feat5 = np.ravel(observation.load_q) / np.array(
        #     [18.6, 78.3, 43.4, 6.4, 9.2, 23.4, 7.8, 3.0, 5.2, 11.6, 12.5])  # 转换发电机有功
        # feat6 = np.ravel(observation.load_v) / np.array(
        #     [142.1, 142.1, 142.1, 142.1, 22., 22., 22., 22., 22., 22., 22.])  # 转换发电机电压
        # feat7 = np.ravel(observation.rho) # 转换发电机无功
        # feat8 = np.ravel(observation.line_status)  # 转换支路状态
        # feat9 = np.ravel(observation.timestep_overflow)  # 转换发电机状态
        # feat10 = np.ravel(observation.topo_vect)

        feat1 = np.ravel(observation.gen_p)
        feat2 = np.ravel(observation.p_or)
        feat3 = np.ravel(observation.p_ex)
        feat4 = np.ravel(observation.load_p)
        # feat5 = np.ravel(observation.load_q)
        # feat6 = np.ravel(observation.load_v)
        feat7 = np.ravel(observation.rho)  # 转换发电机无功
        feat8 = np.ravel(observation.line_status)  # 转换支路状态
        feat9 = np.ravel(observation.timestep_overflow)  # 转换发电机状态
        feat10 = np.ravel(observation.topo_vect)
        # 把所有特征向量连接起来生成单个特征向量
        feats = np.concatenate([feat1, feat2, feat3, feat4, feat7, feat8, feat9, feat10]).astype(np.float32)

        return feats


def create_variables(
        network: snt.Module,
        input_spec
) -> Optional[tf.TensorSpec]:
    """Builds the network with dummy inputs to create the necessary variables.

  Args:
    network: Sonnet Module whose variables are to be created.
    input_spec: list of input specs to the network. The length of this list
      should match the number of arguments expected by `network`.

  Returns:
    output_spec: only returns an output spec if the output is a tf.Tensor, else
        it doesn't return anything (None); e.g. if the output is a
        tfp.distributions.Distribution.
  """
    from acme.tf.utils import squeeze_batch_dim, add_batch_dim, zeros_like
    # Create a dummy observation with no batch dimension.
    dummy_input = zeros_like([tf.convert_to_tensor([0.] * input_spec)])

    # If we have an RNNCore the hidden state will be an additional input.
    if isinstance(network, snt.RNNCore):
        initial_state = squeeze_batch_dim(network.initial_state(1))
        dummy_input += [initial_state]

    # Forward pass of the network which will create variables as a side effect.
    dummy_output = network(*add_batch_dim(dummy_input))

    # Evaluate the input signature by converting the dummy input into a
    # TensorSpec. We then save the signature as a property of the network. This is
    # done so that we can later use it when creating snapshots. We do this here
    # because the snapshot code may not have access to the precise form of the
    # inputs.
    input_signature = tree.map_structure(
        lambda t: tf.TensorSpec((None,) + t.shape, t.dtype), dummy_input)
    network._input_signature = input_signature  # pylint: disable=protected-access

    def spec(output):
        # If the output is not a Tensor, return None as spec is ill-defined.
        if not isinstance(output, tf.Tensor):
            return None
        # If this is not a scalar Tensor, make sure to squeeze out the batch dim.
        if tf.rank(output) > 0:
            output = squeeze_batch_dim(output)
        return tf.TensorSpec(output.shape, output.dtype)

    return tree.map_structure(spec, dummy_output)


# def convert_obs(observation):
#     li_vect = []
#     for el in observation.attr_list_vect:
#         v = observation._get_array_from_attr_name(el).astype(np.float32)
#         v_fix = np.nan_to_num(v)
#         v_norm = np.linalg.norm(v_fix)
#         if v_norm > 1e6:
#             v_res = (v_fix / v_norm) * 10.0
#         else:
#             v_res = v_fix
#         li_vect.append(v_res)
#     return np.concatenate(li_vect)


MAX_FFW = {
    'rte_case5_example': 5,
    'l2rpn_case14_sandbox': 26,
    'l2rpn_wcci_2020': 26
}
DATA_SPLIT = {
    'rte_case5_example': ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    'l2rpn_case14_sandbox': (list(range(0, 40 * 26, 40)), list(range(1, 100 * 10 + 1, 100)), []),
    # list(range(2, 100*10+2, 100))),
    'l2rpn_wcci_2020': (
    [17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230,
     301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689],
    list(range(2880, 2890)), [])
}


def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(MAX_FFW[case]):
            with open(os.path.join(path, f'{i}_{j}.json'), 'r', encoding='utf-8') as f:
                a = json.load(f)
                res[(i, j)] = (a['dn_played'], a['donothing_reward'], a['donothing_nodisc_reward'])
            if i >= 2880: break
    return res
