import os

import numpy as np
from grid2op.Reward import BaseReward


class CustomReward(BaseReward):
    def __init__(self, logger=None):
        """
        Initializes :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`

        """
        BaseReward.__init__(self, logger=logger)
        self.reward_min = 0.
        self.reward_max = 1.
        self._min_rho = 0.90
        self._max_rho = 2.0

        # parameters init with the environment
        self._max_redisp = None
        self._1_max_redisp = None
        self._is_renew_ = None
        self._1_max_redisp_act = None
        self._nb_renew = None

    def initialize(self, env):
        self._max_redisp = np.maximum(env.gen_pmax - env.gen_pmin, 0.)
        self._max_redisp += 1
        self._1_max_redisp = 1.0 / self._max_redisp / env.n_gen
        self._is_renew_ = env.gen_renewable
        self._1_max_redisp_act = np.maximum(np.maximum(env.gen_max_ramp_up, env.gen_max_ramp_down), 1.0)
        self._1_max_redisp_act = 1.0 / self._1_max_redisp_act / np.sum(env.gen_redispatchable)
        self._nb_renew = np.sum(self._is_renew_)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            res = np.sqrt(env.nb_time_step / env.max_episode_duration())
            print(f"{os.path.split(env.chronics_handler.get_id())[-1]}: {env.nb_time_step = }, reward : {res:.3f}")
            if env.nb_time_step <= 5:
                print(f"reason game over: {env.infos['exception']}")
            # episode is over => 2 cases
            # if env.nb_time_step == env.max_episode_duration():
            #     return self.reward_max
            # else:
            #     return self.reward_min
            return res

        if is_illegal or is_ambiguous or has_error:
            return self.reward_min
        # penalize the dispatch
        obs = env.get_obs()
        score_redisp_state = 0.
        # score_redisp_state = np.sum(np.abs(obs.target_dispatch) * self._1_max_redisp)
        score_redisp_action = np.sum(np.abs(action.redispatch) * self._1_max_redisp_act)
        score_redisp = 0.5 * (score_redisp_state + score_redisp_action)

        # penalize the curtailment
        score_curtail_state = 0.
        # score_curtail_state = np.sum(obs.curtailment_mw * self._1_max_redisp)
        curt_act = action.curtail
        score_curtail_action = np.sum(curt_act[curt_act != -1.0]) / self._nb_renew
        score_curtail = 0.5 * (score_curtail_state + score_curtail_action)

        # rate the actions
        score_action = 0.5 * (np.sqrt(score_redisp) + np.sqrt(score_curtail))

        # score the "state" of the grid
        # tmp_state = np.minimum(np.maximum(obs.rho, self._min_rho), self._max_rho)
        # tmp_state -= self._min_rho
        # tmp_state /= (self._max_rho - self._min_rho) * env.n_line
        # score_state = np.sqrt(np.sqrt(np.sum(tmp_state)))
        score_state = 0.

        # score close to goal
        score_goal = 0.
        # score_goal = env.nb_time_step / env.max_episode_duration()
        # score_goal = 1.0

        # score too much redisp
        res = score_goal * (1.0 - 0.5 * (score_action + score_state))
        return score_goal * res

class LossReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
           if is_illegal or is_ambiguous:
               return self.reward_illegal
           elif is_done:
               return self.reward_min
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        reward = (load_p.sum() / gen_p.sum() * 10. - 9.) * 0.1 # avg ~ 0.01
        return reward


from grid2op.dtypes import dt_float

class FlowLimitAndBlackoutReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = dt_float(-env.backend.n_line)
        self.reward_max = dt_float(env.backend.n_line)
        self.reward_none = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            relative_flow = self.__get_lines_capacity_usage(env) # (flow/max capacity)
            limit_diff = dt_float(1.0) - relative_flow
            reward = np.sum((limit_diff**2) * np.sign(limit_diff))
        elif is_done and not has_error:
            reward = self.reward_max # completed!
        elif has_error:
            reward = self.reward_min # blackout or divergence
        else:
            reward = self.reward_none
        return reward

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-5  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
        return relative_flow