# Internal imports.
import math
import time
import pickle
import traceback
import numpy as np
import gc
from acme import adders
from acme import core
import random
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from typing import Mapping, Sequence
import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import torch
import tree
import sys
from concurrent import futures
from typing import Mapping, Optional, Sequence
from acme import core
import tensorflow as tf
import tree
import msgpack
import torch.nn as nn


import torch.nn.functional as F

from misc import gumbel_softmax, onehot_from_logits


class TorchDQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super(TorchDQN, self).__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.layers = nn.Sequential(
            nn.Linear(self.observation_size, self.observation_size * 2),
            nn.ReLU(),
            nn.Linear(self.observation_size * 2, self.observation_size),
            nn.ReLU(),
            nn.Linear(self.observation_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        self.advantage_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_size),
        )
        self.value_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        advantage = self.advantage_layers(x)
        advantage_mean = torch.mean(advantage, axis=1, keepdim=True)
        advantage = advantage - advantage_mean
        value = self.value_layers(x)
        q_out = value + advantage
        return q_out


class TFToPyTorchVariableClient:
    """A variable client for updating variables from a remote source."""

    def __init__(self,
                 client: core.VariableSource,
                 model,
                 update_period: int = 1):
        self.m = model
        self._call_counter = 0
        self._update_period = update_period
        self._client = client
        self._request = lambda: client.get_variables(None)

        self.updated_callbacks = []

        # Create a single background thread to fetch variables without necessarily
        # blocking the actor.
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._async_request = lambda: self._executor.submit(self._request)

        # Initialize this client's future to None to indicate to the `update()`
        # method that there is no pending/running request.
        self._future: Optional[futures.Future] = None

    def add_updated_callback(self, cb):
        self.updated_callbacks.append(cb)

    def update(self):
        """Periodically updates the variables with the latest copy from the source.
    Unlike `update_and_wait()`, this method makes an asynchronous request for
    variables and returns. Unless the request is immediately fulfilled, the
    variables are only copied _within a subsequent call to_ `update()`, whenever
    the request is fulfilled by the `VariableSource`.
    This stateful update method keeps track of the number of calls to it and,
    every `update_period` call, sends an asynchronous request to its server to
    retrieve the latest variables. It does so as long as there are no existing
    requests.
    If there is an existing fulfilled request when this method is called,
    the resulting variables are immediately copied.
    """

        # Track the number of calls (we only update periodically).
        if self._call_counter < self._update_period:
            self._call_counter += 1

        period_reached: bool = self._call_counter >= self._update_period
        has_active_request: bool = self._future is not None

        if period_reached and not has_active_request:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            # self._future = self._async_request()    # todo uncomment
            self._copy(self._request())
            self._call_counter = 0

        if has_active_request and self._future.done():
            # The active request is done so copy the result and remove the future.
            self._copy(self._future.result())
            self._future: Optional[futures.Future] = None
        else:
            # There is either a pending/running request or we're between update
            # periods, so just carry on.
            return

    def update_and_wait(self):
        """Immediately update and block until we get the result."""
        self._copy(self._request())

    def _copy(self, new_variables: Sequence[Sequence[tf.Variable]]):
        """Copies the new variables to the old ones."""

        for cb in self.updated_callbacks:
            cb(new_variables)


def pytorch_model_load_state_dict(model, new_variables):
    if len(new_variables) == 0:
        return
    pytorch_keys = list(model.state_dict().keys())
    # Switch ordering of weight + bias
    new_pytorch_keys = []
    for i in range(0, len(pytorch_keys), 2):
        new_pytorch_keys.append(pytorch_keys[i + 1])
        new_pytorch_keys.append(pytorch_keys[i])
    pytorch_keys = new_pytorch_keys
    new_state_dict = {k: torch.from_numpy(v.T) for k, v in zip(pytorch_keys, new_variables)}
    model.load_state_dict(new_state_dict)


def pytorch_quantize(m, q):
    assert (q in [8, 16, 32])
    if q == 8:
        return torch.quantization.quantize_dynamic(
            m, {torch.nn.Linear}, dtype=torch.qint8)
    if q == 16:
        return torch.quantization.quantize_dynamic(
            m, {torch.nn.Linear}, dtype=torch.float16)
    if q == 32:
        return m


class PytorchTanhToSpec(torch.nn.Module):
    def __init__(self, action_spec):
        super(PytorchTanhToSpec, self).__init__()
        self._scale = torch.from_numpy(np.array(action_spec.maximum - action_spec.minimum))
        self._offset = torch.from_numpy(np.array(action_spec.minimum))

    def forward(self, x):
        inputs = torch.tanh(x)
        inputs = .5 * (inputs + 1.0)
        outputs = inputs * self._scale + self._offset
        return outputs


class PytorchClippedGaussian(torch.nn.Module):
    def __init__(self, sigma):
        super(PytorchClippedGaussian, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.normal(0, self.sigma, size=x.size())
        x = x + noise
        x = torch.clamp(x, -1.0, 1.0)
        return x


class PytorchClipToSpec(torch.nn.Module):
    def __init__(self, spec):
        super(PytorchClipToSpec, self).__init__()
        self.spec = spec
        self._min = torch.from_numpy(spec.minimum)
        self._max = torch.from_numpy(spec.maximum)

    def forward(self, x):
        return torch.max(torch.min(x, self._max), self._min)


def create_model(input_size, output_size, sigma, policy_layer_sizes=(2048, 2048, 2048)):
    sizes = [input_size] + list(policy_layer_sizes) + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        in_size, out_size = sizes[i], sizes[i + 1]
        layers.append(torch.nn.Linear(in_size, out_size))
        layers.append(torch.nn.ReLU())

    layers = layers[:-1]
    # layers.append(PytorchTanhToSpec(action_spec))
    layers.append(torch.nn.Tanh())
    # layers.append(PytorchClipToSpec(action_spec))

    return torch.nn.Sequential(*layers)


tfd = tfp.distributions


class MyFeedForwardActor(core.Actor):
    """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

    def __init__(
            self,
            policy_network: snt.Module,
            decay_epsilon,
            ini_epsilon=.99,
            final_epsilon=.5,
            adder: Optional[adders.Adder] = None,
            variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self.eps = ini_epsilon
        self.ini_epsilon = ini_epsilon
        self.final_epsilon = final_epsilon
        self.decay_epsilon = decay_epsilon
        self._policy_network = policy_network

    @tf.function
    def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

        return action

    def adaptive_epsilon_decay(self, step):
        ada_div = self.decay_epsilon / 10.0
        step_off = step + ada_div
        ada_eps = self.ini_epsilon * -math.log10((step_off + 1) / (self.decay_epsilon + ada_div))
        ada_eps_up_clip = min(self.ini_epsilon, ada_eps)
        ada_eps_low_clip = max(self.final_epsilon, ada_eps_up_clip)
        self.eps = ada_eps_low_clip

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        # Pass the observation through the policy network.
        batched_observation = tf2_utils.add_batch_dim(observation)
        pred = self._policy_network(batched_observation)
        pred = pred.numpy().flatten()
        if random.random() <= self.eps:
            return np.int32(random.randint(0, pred.shape[0] - 1))
        return np.argmax(pred)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)


class FeedForwardActor(core.Actor):
    """A feed-forward actor.
  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

    def __init__(
            self,
            m,
            decay_epsilon,
            ini_epsilon=.99,
            final_epsilon=.01,
            q=32,
            args=None,
            adder: adders.Adder = None,
            variable_client: tf2_variable_utils.VariableClient = None,
    ):
        """Initializes the actor.
    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._variable_client.add_updated_callback(self.updated)
        self.eps = ini_epsilon
        self.ini_epsilon = ini_epsilon
        self.final_epsilon = final_epsilon
        self.decay_epsilon = decay_epsilon
        self.m = m
        self.q = q
        self.q_m = pytorch_quantize(self.m, self.q)
        self.q_m_state_dict = self.q_m.state_dict()
        self.args = args

    def adaptive_epsilon_decay(self, step):
        ada_div = self.decay_epsilon / 10.0
        step_off = step + ada_div
        ada_eps = self.ini_epsilon * -math.log10((step_off + 1) / (self.decay_epsilon + ada_div))
        ada_eps_up_clip = min(self.ini_epsilon, ada_eps)
        ada_eps_low_clip = max(self.final_epsilon, ada_eps_up_clip)
        self.eps = ada_eps_low_clip

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        pred = self.q_m(torch.from_numpy(observation).reshape(1, -1))
        return gumbel_softmax(pred, hard=True).detach().numpy().flatten()

    def predict_action(self, observation: types.NestedArray) -> types.NestedArray:
        pred = self.q_m(torch.from_numpy(observation).reshape(1, -1))
        pred = pred.detach().numpy().flatten()
        return np.argmax(pred)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(
            self,
            action: types.NestedArray,
            next_timestep: dm_env.TimeStep,
    ):
        if self._adder:
            self._adder.add(action.astype(np.int32), next_timestep)

    def update(self):
        if self._variable_client:
            self._variable_client.update()

    def updated(self, new_variables):
        t1 = time.time()
        try:
            state_dict = new_variables[0]
        except:
            return

        if "id" in state_dict:
            del state_dict["id"]

        if self.args["weight_compress"] != 0:
            t_q_decompress_start = time.time()
            for k, v in state_dict.items():
                state_dict[k] = Q_decompress(state_dict[k], self.args["weight_compress"])
            print("Q_decompress time: %f" % (time.time() - t_q_decompress_start))

        if self.q == 32:
            self.q_m.load_state_dict(state_dict)
        else:
            """
      with torch.no_grad():
        for name, child in self.q_m._modules.items():
          print("Loading: ", child, type(child))
          if type(child) == torch.nn.quantized.dynamic.modules.linear.Linear:
            scale = state_dict[name + ".scale"]
            zero_point = state_dict[name + ".zero_point"]
            packed_params = state_dict[name + "._packed_params._packed_params"]
            packed_params_dumped = state_dict[name + "._packed_params._packed_params.dumped"]
            child.scale = scale
            child.zero_point = zero_point
            #child._packed_params._packed_params = torch.ops.quantized.linear_prepack_fp16(packed_params[0], packed_params[1])
            child._packed_params._packed_params = pickle.loads(packed_params_dumped)        

      """
            self.q_m.load_state_dict(state_dict)
            pass

        print("Load state dict time: %f" % (time.time() - t1))
