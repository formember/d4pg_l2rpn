import datetime
import json
import os
from typing import Optional

from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme.utils import loggers
from grid2op import make
from grid2op.Action import PlayableAction, TopologyChangeAndDispatchAction
from grid2op.Reward import GameplayReward, L2RPNReward, CombinedScaledReward
from lightsim2grid import LightSimBackend
from matplotlib import pyplot as plt
import pandas as pd

from CustomReward import CustomReward
from MyEnv import makeMyEnv, makeCustomeEnv
from d4pg_args import parser
from d4pg_learner import D4PG_learner
import sonnet as snt
import sys
import tensorflow as tf
from concurrent import futures
import trfl
from custom_environment_loop import CustomEnvironmentLoop
from utils import DQNnetwork, read_ffw_json, DATA_SPLIT, MAX_FFW

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
# tf.config.run_functions_eagerly(True)


# For debugging
# tf.debugging.set_log_device_placement(True)

class PeriodicBroadcaster(object):
    """A variable client for updating variables from a remote source."""

    def __init__(self, f, update_period=1):
        self._call_counter = 0
        self._update_period = update_period
        self._request = lambda x: f(x)

        self.updated_callbacks = []

        # Create a single background thread to fetch variables without necessarily
        # blocking the actor.
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._async_request = lambda x: self._executor.submit(self._request, x)

        # Initialize this client's future to None to indicate to the `update()`
        # method that there is no pending/running request.
        self._future: Optional[futures.Future] = None

    def add_updated_callback(self, cb):
        self.updated_callbacks.append(cb)

    def update(self, weights):
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
            self._future = self._async_request(weights)  # todo uncomment
            self._call_counter = 0

        if has_active_request and self._future.done():
            # The active request is done so copy the result and remove the future.
            self._future: Optional[futures.Future] = None
        else:
            # There is either a pending/running request or we're between update
            # periods, so just carry on.
            return


DATA_DIR = './data'

if __name__ == "__main__":
    # os.sched_setaffinity(0, [0])
    args = parser.parse_args()

    print(vars(args))
    train_env_name = args.taskstr + "_train"
    val_env_name = args.taskstr + "_val"
    # 创建电网环境
    env_path = os.path.join(DATA_DIR, args.taskstr)
    chronics_path = os.path.join(env_path, 'chronics')
    dn_json_path = os.path.join(env_path, 'json')
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[args.taskstr]
    env_path = os.path.join(DATA_DIR, args.taskstr)

    environment_grid,environment_grid_val=makeCustomeEnv(args.taskstr,args.seed)
    # Only load 128 steps in ram
    # environment_grid.chronics_handler.set_chunk_size(128)
    #
    # # Register custom reward for training


    model_sizes = tuple([int(x) for x in args.model_str.split(",")])
    DQN_network = DQNnetwork(environment_grid,environment_grid.action_space)
    agent_networks = DQN_network.make_networks(placement=args.learner_device_placement,
                                               policy_layer_sizes=model_sizes)

    logger = loggers.InMemoryLogger()
    # Create D4PG learner
    this_path = os.getcwd()
    log_path = os.path.join(this_path, "dbg_logdir/logs_stdout")
    log_path = os.path.join(log_path, args.taskstr)
    if args.quantize == 8:
        log_path = os.path.join(log_path, "quantize")
    else:
        log_path = os.path.join(log_path, "normal")
    log_path=os.path.join(log_path,f"actor_{args.n_actors}")
    checkpoint_path = os.path.join(this_path, "chechpoint")
    tensorboard_path=os.path.join(log_path,"tensorboard")
    summary_writer = tf.compat.v2.summary.create_file_writer(
        tensorboard_path,
        flush_millis=10000)
    learner = D4PG_learner(observation_spec=DQN_network.observation_size,
                          action_spec=DQN_network.action_size,
                          policy_network=agent_networks['policy'],
                          critic_network=agent_networks['critic'],
                          observation_network=agent_networks['observation'],
                          summary_writer=summary_writer,
                          discount=0.98,
                          logger=logger,
                          priority_exponent=0.7,
                          port=args.port,
                          replay_table_name=args.replay_table_name,
                          model_table_name=args.model_table_name,
                          replay_table_max_times_sampled=args.replay_table_max_times_sampled,
                          max_replay_size=args.replay_table_max_replay_size,
                          min_replay_size=args.min_replay_size,
                          shutdown_table_name=args.shutdown_table_name,
                          device_placement=args.learner_device_placement,
                          batch_size=args.batch_size,
                          broadcaster_table_name=args.broadcaster_table_name,
                          checkpoint_subpath=checkpoint_path)

    # Create the evaluation policy.
    with tf.device(args.learner_device_placement):

        # Create the behavior policy.
        epsilon = tf.Variable(0.00, trainable=False)
        eval_policy = snt.Sequential([
            agent_networks["policy"],
            lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
        ])
        eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
    # eval_env = make_environment(args.taskstr)
    # dn_ffw = read_ffw_json(dn_json_path, train_chronics + valid_chronics, args.taskstr)
    # ep_infos = {}
    # if os.path.exists(dn_json_path):
    #     for i in list(set(train_chronics + valid_chronics)):
    #         with open(os.path.join(dn_json_path, f'{i}.json'), 'r', encoding='utf-8') as f:
    #             ep_infos[i] = json.load(f)
    lable_path=os.path.join(log_path,"environment_loop")

    eval_loop = CustomEnvironmentLoop(environment_grid_val, learner, DQN_network,summary_writer=summary_writer,chronics=valid_chronics,max_ffw=MAX_FFW[args.taskstr],
                                      label=lable_path)

    def broadcast_shutdown(should_shutdown):
        learner.client.insert(should_shutdown, {args.shutdown_table_name: 1.0})


    steps = 0


    def submit_parameters_to_broadcaster(weights):
        if weights is None:
            weights = [tf2_utils.to_numpy(v) for v in learner.learner._policy_network.variables]
        learner.client.insert(weights, {args.broadcaster_table_name: 1.0})


    broadcast_shutdown(0)
    variable_broadcaster = PeriodicBroadcaster(submit_parameters_to_broadcaster)
    # variable_broadcaster.update(None)
    with tf.device(args.learner_device_placement):
        for i in range(args.num_episodes):
            learner.learner.step()
            print("finish one")
            sys.stdout.flush()
            variable_broadcaster.update(None)
            if (i) % 1000 == 0:
                # model_save_str= f"saved_model_{i}"
                # model_save_path = os.path.join(log_path, model_save_str)
                # tf.saved_model.save(learner.learner._network, model_save_path)
                eval_loop.run(i)
                # eval_loop.test(valid_chronics,MAX_FFW[args.taskstr])
            # if (i + 1) % 10000 == 0:
            #     tf.saved_model.save(learner.learner._network,"saved_model")
                # learner._checkpointer.save()
    learner._checkpointer.save()
    df = pd.DataFrame(logger.data)
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    csv_file=f"loss{timestamp}.csv"
    csv_path=os.path.join(log_path,csv_file)
    df.to_csv(csv_path,sep=',',index=True,header=True)
    plt.figure(figsize=(10, 4))
    plt.title('Training loss')
    plt.xlabel('Training episodes')
    plt.ylabel('loss')
    plt.plot(df['loss'])
    png_file=f"loss{timestamp}.png"
    png_path=os.path.join(log_path,png_file)
    plt.savefig(png_path)

    broadcast_shutdown(1)
