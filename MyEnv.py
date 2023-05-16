import os.path
import re

import grid2op
from grid2op.Action import TopologyChangeAndDispatchAction, TopologySetAction, TopologyAndDispatchAction
from grid2op.Reward import L2RPNSandBoxScore, CombinedScaledReward, GameplayReward, L2RPNReward, RedispReward, \
    BridgeReward, CloseToOverflowReward, DistanceReward, LinesReconnectedReward
from lightsim2grid import LightSimBackend

from CustomReward import LossReward, FlowLimitAndBlackoutReward

max_iter = 7 * 24 * 12
def makeMyEnv(taskdir,seed):
    # env = grid2op.make(taskdir, test=True,reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
    #             other_rewards={'loss': LossReward})
    # test_env = grid2op.make(taskdir, test=True,reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
    #             other_rewards={'loss': LossReward})
    #
    # env.deactivate_forecast()
    # test_env.deactivate_forecast()
    # env.seed(seed)
    # test_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    # test_env.parameters.NB_TIMESTEP_RECONNECTION = env.parameters.NB_TIMESTEP_RECONNECTION = 12
    # test_env.parameters.NB_TIMESTEP_COOLDOWN_LINE = env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    # test_env.parameters.NB_TIMESTEP_COOLDOWN_SUB = env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    # test_env.parameters.HARD_OVERFLOW_THRESHOLD = env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    # test_env.seed(59)
    env = grid2op.make(taskdir,
               action_class=TopologyChangeAndDispatchAction,
               reward_class=CombinedScaledReward)

    # Only load 128 steps in ram
    env.chronics_handler.set_chunk_size(128)

    # Register custom reward for training
    try:
        # change of name in grid2op >= 1.2.3
        cr = env._reward_helper.template_reward
    except AttributeError as nm_exc_:
        cr = env.reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    # cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(env.n_line))
    # Initialize custom rewards
    cr.initialize(env)
    # Set reward range to something managable
    cr.set_range(-1.0, 1.0)

    test_env = grid2op.make(taskdir,
               reward_class=RedispReward,
               action_class=TopologyChangeAndDispatchAction,
               other_rewards={
                   "bridge": BridgeReward,
                   "overflow": CloseToOverflowReward,
                   "distance": DistanceReward
               })

    return env,test_env


def makeCustomeEnv(taskdir,seed):
    path="/home/dps/桌面/dis_dqn_l2rpn/data"
    this_path=os.path.join(path,taskdir)
    env = grid2op.make(this_path,reward_class=CombinedScaledReward, backend=LightSimBackend(),action_class=TopologyAndDispatchAction)
    test_env = grid2op.make(this_path,reward_class=CombinedScaledReward, backend=LightSimBackend(),action_class=TopologyAndDispatchAction)

    # Register custom reward for training
    try:
        # change of name in grid2op >= 1.2.3
        cr = env._reward_helper.template_reward
    except AttributeError as nm_exc_:
        cr = env.reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    # cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(env.n_line))
    # cr.addReward("l2rpn", L2RPNReward(), 1)
    # Initialize custom rewards
    cr.initialize(env)
    # Set reward range to something managable
    cr.set_range(-1.0, 1.0)


    # Register custom reward for training
    try:
        # change of name in grid2op >= 1.2.3
        cr = test_env._reward_helper.template_reward
    except AttributeError as nm_exc_:
        cr = test_env.reward_helper.template_reward
    # cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 1.0)
    # cr.addReward("recolines", LinesReconnectedReward(), 1.0)
    cr.addReward("l2rpn", L2RPNReward(), 2.0 / float(env.n_line))
    # cr.addReward("l2rpn", L2RPNReward(), 1)
    # Initialize custom rewards
    cr.initialize(test_env)
    # Set reward range to something managable
    cr.set_range(-1.0, 1.0)

    env.deactivate_forecast()
    test_env.deactivate_forecast()
    test_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    test_env.parameters.NB_TIMESTEP_RECONNECTION = env.parameters.NB_TIMESTEP_RECONNECTION = 12
    test_env.parameters.NB_TIMESTEP_COOLDOWN_LINE = env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    test_env.parameters.NB_TIMESTEP_COOLDOWN_SUB = env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    test_env.parameters.HARD_OVERFLOW_THRESHOLD = env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    # test_env.seed(59)
    return env,test_env

    # env = grid2op.make('rte_case14_realistic', reward_class=FlowLimitAndBlackoutReward, backend=LightSimBackend(),
    #                    action_class=TopologySetAction)
    # test_env = grid2op.make('rte_case14_realistic', reward_class=FlowLimitAndBlackoutReward, backend=LightSimBackend(),
    #                    action_class=TopologySetAction)
    # test_env.seed(59)
    # return env,test_env


