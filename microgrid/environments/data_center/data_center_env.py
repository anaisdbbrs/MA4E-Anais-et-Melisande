from typing import Optional, Union, Tuple

import gym
import numpy as np
from gym import spaces
from gym.core import ObsType, ActType

from microgrid.assets.data_center import DataCenter
import datetime


class DataCenterEnv(gym.Env):
    def __init__(self, data_center_config: dict, nb_pdt=24, seed: Optional[int] = None):
        self.nb_pdt = nb_pdt
        self.data_center = DataCenter(**data_center_config)

        self.observation_space = spaces.Dict(
            {
                'datetime': spaces.Space[datetime.datetime]((), np.datetime64, seed),
                'manager_signal': spaces.Box(low=-np.inf, high=np.inf, shape=(nb_pdt,)),
                'consumption_prevision': spaces.Box(low=0.0, high=np.inf, shape=(nb_pdt,)),
                'hotwater_price_prevision': spaces.Box(low=0.0, high=np.inf, shape=(nb_pdt,)),
            }
        )
        # the action is alpha_t
        self.action_space = spaces.Box(low=0, high=1.0, shape=(nb_pdt,))
        self.now = None
        self.delta_t = None
        # constant problem data
        self.Tcom = self.data_center.Tcom
        self.Tr = self.data_center.Tr
        self.rho = self.data_center.rho
        self.COP_HP = self.Tcom/(self.Tcom - self.Tr) * self.rho
        self.EER = self.data_center.EER
        self.COP_CS = self.EER + 1
        self.max_transfert = self.data_center.max_transfert

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        max_alpha = self.data_center.get_max_alpha_t(self.now, self.delta_t)
        effective_alpha = max(0, min(action[0], max_alpha))
        self.now += self.delta_t
        return self._step_common(effective_alpha, 1 if action[0] != effective_alpha else 0)

    def try_step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        max_alpha = self.data_center.get_max_alpha_t(self.now, self.delta_t)
        effective_alpha = max(0, min(action[0], max_alpha))
        return self._step_common(effective_alpha, 1 if action[0] != effective_alpha else 0)

    def _step_common(self, effective_alpha, penalties) -> Tuple[ObsType, float, bool, dict]:
        state = {
            'datetime': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'consumption_prevision': self.data_center.get_conso_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
            'hotwater_price_prevision': self.data_center.get_prices_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        reward = 0 if penalties == 0 else -1e5
        return state, reward, False, {'reward': reward, 'penalties': penalties, 'effective_action': effective_alpha, 'datetime': self.now}

    def reset(self, *args, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None)\
            -> Union[ObsType, Tuple[ObsType, dict]]:
        self.now, self.delta_t = tuple(args)
        self.data_center.reset()
        state = {
            'datetime': self.now,
            'manager_signal': np.zeros(self.nb_pdt),
            'consumption_prevision': self.data_center.get_conso_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
            'hotwater_price_prevision': self.data_center.get_prices_prevision([self.now + i * self.delta_t for i in range(self.nb_pdt)]),
        }
        return state

    def get_consumption(self, state: ObsType,  action: ActType) -> np.ndarray:
        dt = self.delta_t / datetime.timedelta(hours=1)
        lit = state['consumption_prevision']
        hr = lit * self.COP_CS / self.EER
        lhp = hr / ((self.COP_HP - 1) * dt)
        return lit + action * lhp

    def render(self, mode="human"):
        pass

