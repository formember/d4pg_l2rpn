import os
import random
import traceback
from acme import core
from acme.adders import reverb as adders
from acme.agents.tf import actors
import tree
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from grid2op import make
from grid2op.Action import TopologyChangeAndDispatchAction
from grid2op.Reward import GameplayReward, L2RPNReward, CombinedScaledReward

from MyEnv import makeMyEnv, makeCustomeEnv
from d4pg_args import parser
from utils import get_actor_sigma, DQNnetwork, create_variables, MAX_FFW, DATA_SPLIT, read_ffw_json
import numpy as np
import pickle
import pytorch_actors
import reverb
import sonnet as snt
import sys
import tensorflow as tf
import time
import torch
import trfl
import zlib
torch.set_num_threads(1)
cpus = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpus)


class ExternalVariableSource(core.VariableSource):

    def __init__(self, reverb_table, model_table_name, actor_id, args):
        self.reverb_table = reverb_table
        self.model_table_name = model_table_name
        self.actor_id = actor_id
        self.prev_sample = None
        self.args = args
        self.cached = None
    def get_variables(self, names):
        # if self.cached is not None:
        #  return self.cached

        # Pull from reverb table
        tstart = time.time()
        sample = next(self.reverb_table.sample(self.model_table_name))[0]
        tend = time.time()

        # Decode sample
        d = [x.tobytes() for x in sample.data]

        try:
            if self.args["compress"]:
                d = [zlib.decompress(x) for x in d]
            tdecompress = time.time()
            decoded = [pickle.loads(x) for x in d]
            tdecode = time.time()
            print("Pull time: %f, Decompress/tobytes time: %f, Deserialize time: %f" % (
            tend - tstart, tdecompress - tend, tdecode - tdecompress))
            return decoded
        except:
            traceback.print_exc()
            pass
        return []


class IntraProcessTFToPyTorchVariableSource(core.VariableSource):
    def __init__(self, tf_model):
        self.tf_model = tf_model

    def get_variables(self, name):
        res = [tf2_utils.to_numpy(v) for v in self.tf_model.variables]
        return res


def get_shutdown_status(client, shutdown_table_name):
    sample = next(client.sample(shutdown_table_name))[0]
    return int(sample.data[0])


def get_rapaly_data(client):
    sample = next(client.sample("priority_table"))[0]
    return sample
DATA_DIR="./data"


def chronic_priority(dn_ffw, cid, ffw, step):
    m = 864
    scale = 2.
    diff_coef = 0.05
    d = dn_ffw[(cid, ffw)][0]
    progress = 1 - np.sqrt(step / m)
    difficulty = 1 - np.sqrt(d / m)
    score = (progress + diff_coef * difficulty) * scale
    return score


def actor_main(actor_id, args):
    print("Starting actor %d" % actor_id)

    address = "localhost:%d" % args["port"]
    client = reverb.Client(address)
    actor_device_placement = args["actor_device_placement"]
    actor_device_placement = "%s:0" % (actor_device_placement)

    model_sizes = tuple([int(x) for x in args["model_str"].split(",")])
    env_path = os.path.join(DATA_DIR, args["taskstr"])
    # 创建一个电网环境
    environment_grid,_= makeCustomeEnv(args["taskstr"],args["seed"])
    # Only load 128 steps in ram
    # environment_grid.chronics_handler.set_chunk_size(128)
    DQN_network = DQNnetwork(environment_grid,environment_grid.action_space)
    # 根据电网环境创建网络
    policy_network = DQN_network.make_networks(placement=args["actor_device_placement"],
                                               policy_layer_sizes=model_sizes)["policy"]
    observation_spec=DQN_network.observation_size
    action_spec=DQN_network.action_size
    with tf.device(actor_device_placement):
        create_variables(policy_network, observation_spec)
        # epsilon = tf.Variable(get_actor_sigma(args["sigma"], args["actor_id"], args["n_actors"]), trainable=False,dtype=tf.float32)
        behavior_network = snt.Sequential([
            policy_network
        ])

        # 配置adder
        adder = adders.NStepTransitionAdder(
            # priority_fns={args["replay_table_name"]: lambda x: 1.},
            client=client,
            n_step=args["n_step"],
            discount=args["discount"])

        variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
        variable_client = tf2_variable_utils.VariableClient(
            variable_source, {'policy': policy_network.variables}, update_period=args["actor_update_period"])

        # 创建FeedActor
        actor = pytorch_actors.MyFeedForwardActor(behavior_network,decay_epsilon=(args["num_episodes"]/args["n_actors"])*30,adder=adder, variable_client=variable_client)

        # 创建pytorch actor
        pytorch_adder = adders.NStepTransitionAdder(
            priority_fns={args["replay_table_name"]: lambda x: 1.},
            client=client,
            n_step=args["n_step"],
            discount=args["discount"])

        pytorch_model = pytorch_actors.create_model(observation_spec,
                                                    action_spec,
                                                    args["sigma"],
                                                    policy_layer_sizes=model_sizes)
        pytorch_variable_source = ExternalVariableSource(client, args["model_table_name"], actor_id, args)
        pytorch_variable_client = pytorch_actors.TFToPyTorchVariableClient(
            pytorch_variable_source, pytorch_model, update_period=args["actor_update_period"])
        pytorch_actor = pytorch_actors.FeedForwardActor(pytorch_model,
                                                        decay_epsilon=(args["num_episodes"]/args["n_actors"])*3,
                                                        adder=pytorch_adder,
                                                        variable_client=pytorch_variable_client,
                                                        q=args["quantize"],
                                                        args=args)

    actor = {
        "tensorflow": actor,
        "pytorch": pytorch_actor,
    }[args["inference_mode"]]
    # Main actor loop
    t_start = time.time()
    n_total_steps = 0
    max_ffw=MAX_FFW[args["taskstr"]]
    train_chronics,valid_chronics, test_chronics=DATA_SPLIT[args["taskstr"]]
    # train_chronics_ffw = [(cid, fw) for cid in valid_chronics for fw in range(max_ffw)]
    # dn_json_path = os.path.join(env_path, 'json')
    # dn_ffw = read_ffw_json(dn_json_path, train_chronics + valid_chronics, args["taskstr"])
    # total_chronic_num = len(train_chronics_ffw)
    # chronic_records = [0] * total_chronic_num
    # for i in chronic_records:
    #     cid, fw = train_chronics_ffw[i]
    #     chronic_records[i] = chronic_priority(dn_ffw,cid, fw, 1)
    # dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(chronic_records))
    while True:
        should_shutdown = get_shutdown_status(client, args["shutdown_table_name"])
        sys.stdout.flush()
        if should_shutdown:
            break
        # record_idx = dist.sample().item()
        # chronic_id, ffw = train_chronics_ffw[record_idx]
        environment_grid.set_id(random.choice(train_chronics))
        # _ = environment_grid.chronics_handler.sample_next_chronics()
        timestep = environment_grid.reset()
        current_chronics_id = environment_grid.chronics_handler.get_id()
        print(f"Current chronic ID is {current_chronics_id}")
        observation = DQN_network.convert_obs(observation=timestep)
        # observation=tf.cast(observation,dtype=tf.float32)
        episode_return = 0
        done=False
        # 将experience写入table中
        actor._adder._writer.append(dict(observation=observation,
                             start_of_episode=True),
                        partial_step=True)
        actor._adder._add_first_called = True
        t_start_local = time.time()
        local_steps = 0

        while not done:
            local_steps += 1
            tstart = time.time()

            # 根据actor模型选取action
            if n_total_steps<args["pre_step"]:
                action=np.random.randint(0,action_spec)
            else:
                with tf.device(actor_device_placement):
                    action = actor.select_action(observation)
            action_softmax=np.argmax(action)
            print(action_softmax)
            action_con=DQN_network.convert_act(action_softmax)
            new_obs, reward, done, info = environment_grid.step(action_con)
            if local_steps==864:
                done=True
            observation=DQN_network.convert_obs(new_obs)
            # 将experience写入table中
            if actor._adder._writer.episode_steps >= actor._adder.n_step:
                actor._adder._first_idx += 1
            actor._adder._last_idx += 1
            # if done:
            #     discount=np.float32(1.0)
            # else:
            #     discount=np.float32(0.0)
            discount = np.float32(1.0)
            if not actor._adder._add_first_called:
                raise ValueError('adder.add_first must be called before adder.add.')
            current_step = dict(
                # Observation was passed at the previous add call.
                action=np.int32(action),
                reward=np.float32(reward),
                discount=discount,
                **{}
            )
            actor._adder._writer.append(current_step)
            # Have the agent observe the timestep and let the actor update itself.

            actor._adder._writer.append(
                dict(
                    observation=observation,
                    start_of_episode=False),
                partial_step=True)
            actor._adder._write()
            if done:
                dummy_step = tree.map_structure(np.zeros_like, current_step)
                actor._adder._writer.append(dummy_step)
                actor._adder._write_last()
                actor._adder.reset()

            episode_return +=reward
            n_total_steps += 1
            # actor.adaptive_epsilon_decay(n_total_steps)
            tend = time.time()

            print("Step time: %f" % (tend - tstart))

            # Update the actor
            if n_total_steps * args["n_actors"] >= args["min_replay_size"]:
                actor.update()
            # print(args)
        

        steps_per_second = n_total_steps / (time.time() - t_start)
        local_steps_per_second = local_steps / (time.time() - t_start_local)
        print("Actor %d finished timestep (r=%f) (steps_per_second=%f) (local_steps_per_second=%f)" % (
        actor_id, float(episode_return), steps_per_second, local_steps_per_second))
    print("Actor %d shutting down" % (actor_id))


if __name__ == "__main__":

    args = parser.parse_args()
    os.sched_setaffinity(0, [args.actor_id])
    print(vars(args))
    actor_main(args.actor_id, vars(args))
