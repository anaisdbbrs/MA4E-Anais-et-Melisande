import copy
from collections import defaultdict

import numpy as np
import datetime

from microgrid.agents.charging_station_agent import ChargingStationAgent
from microgrid.agents.industrial_agent import IndustrialAgent
from microgrid.agents.solar_farm_agent import SolarFarmAgent
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
from microgrid.environments.industrial.industrial_env import IndustrialEnv
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
from matplotlib import pyplot as plt


class Manager:
    def __init__(self,
                 agents: dict,
                 start: datetime.datetime = datetime.datetime.now(),
                 delta_t: datetime.timedelta = datetime.timedelta(minutes=30),
                 horizon: datetime.timedelta = datetime.timedelta(days=1),
                 simulation_horizon: datetime.timedelta = datetime.timedelta(days=1),
                 max_iterations: int = 10,
                 ):
        self.nb_agents = len(agents)
        self.agents = agents
        self.envs = [agent.env for _, agent in agents.items()]
        self.nb_pdt = horizon // delta_t
        self.start = start.replace(minute=0, second=0, microsecond=0)
        self.horizon = horizon
        self.simulation_horizon = simulation_horizon
        self.delta_t = delta_t
        self.iteration = 0
        self.max_iterations = max_iterations
        self.previous_consumptions = np.zeros(self.nb_pdt)
        self.data_bank = defaultdict(lambda : defaultdict(dict))

    def init_envs(self):
        # resetting all environments
        agents_data = {}
        for name, agent in self.agents.items():
            env = agent.env
            agent_state = env.reset(self.start, self.delta_t)
            agents_data[name] = \
                {
                    'state': agent_state,
                }
        return agents_data

    def run(self):
        agents_data = self.init_envs()
        signal = np.zeros(self.nb_pdt)
        self.data_bank['initial_state'] = copy.deepcopy(agents_data)
        N = self.simulation_horizon // self.delta_t
        for pdt in range(N):
            now = self.start + pdt * self.delta_t
            # We loop until convergence or max iterations
            agents_data, signal = self.loop(now, agents_data, signal)
            # we apply the last action to the environment
            agents_data = self.apply_all_agents_actions(now, agents_data)
            # we update the data bank
            self.data_bank[now].update(copy.deepcopy(agents_data))
            # we update the signal for the next time step
            signal = self.adapt_signal_for_next_timestep(signal)
            # we update the current_state with next_state
            for name, agent in self.agents.items():
                agents_data[name]['state'] = agents_data[name]['next_state'].copy()

    def loop(self,
             now: datetime.datetime,
             agents_data: dict,
             signal: np.ndarray) -> tuple:
        iteration = 0
        signal = signal.copy()
        outputs = {}
        while iteration < self.max_iterations:
            outputs = self.try_all_agents_with_signal(now, signal, agents_data)
            self.data_bank[now]['__cold'][iteration] = copy.deepcopy(outputs)
            if self.has_converged(outputs):
                break
            signal = self.update_signal(signal, outputs)
            iteration += 1
        return outputs, signal

    def try_all_agents_with_signal(self, now: datetime.datetime, signal: np.ndarray, agents_data: dict):
        outputs = {}
        for name, agent in self.agents.items():
            env = agent.env
            data = agents_data[name]
            agent_state = data['state'].copy()
            agent_state['datetime'] = now
            agent_state['manager_signal'] = signal
            agent_action = agent.take_decision(
                agent_state,
                previous_state=data.get('state', None),
                previous_action=data.get('action', None),
                previous_reward=data.get('reward', None),
            )
            agent_new_state, reward, _, info = env.try_step(agent_action)
            consumption = env.get_consumption(agent_state, info['effective_action'])
            outputs[name] = \
                {
                    'signal': signal,
                    'state': agent_state,
                    'action': agent_action,
                    'reward': reward,
                    'info': info,
                    'consumption': consumption,
                }
        outputs = self.update_reward(now, outputs)
        return outputs

    def apply_all_agents_actions(self, now: datetime.datetime, agents_data: dict):
        outputs = {}
        for name, agent in self.agents.items():
            env = agent.env
            data = agents_data[name]
            agent_state = data['state']
            agent_action = data['action']
            agent_new_state, reward, _, info = env.step(agent_action)
            consumption = env.get_consumption(agent_state, info['effective_action'])
            outputs[name] = \
                {
                    'signal': data['signal'],
                    'state': agent_state,
                    'next_state': agent_new_state,
                    'action': info['effective_action'],
                    'reward': reward,
                    'info': info,
                    'consumption': consumption,
                }
        self.update_reward(now, outputs)
        return outputs


    def has_converged(self, agents_data):
        # TODO: check if converged
        current_consumptions = np.array([a['consumption'] for a in agents_data.values()]).sum(axis=0).squeeze()
        if self.previous_consumptions is None:
            self.previous_consumptions = current_consumptions
            return False
        res = np.linalg.norm(self.previous_consumptions - current_consumptions) < self.eps
        self.previous_consumptions = current_consumptions
        return res

    def update_signal(self, signal, agents_data):
        # TODO: update signal based on previous signal and agents_data
        current_consumptions = np.array([a['consumption'] for a in agents_data.values()]).sum(axis=0).squeeze()
        lpv = agents_data["ferme"]["state"]["pv_previsions"]
        return (current_consumptions-lpv)/lpv


    def update_reward(self, now: datetime.datetime, agents_data: dict):
        # TODO: update rewards based on previous rewards and agents_data
        # you should take into account the collective consumption to determine the reward
        total_consumption = sum([a['consumption'][0] for a in agents_data.values()])
        for name, agent in self.agents.items():
            data = agents_data[name]
            data['reward'] += total_consumption * data['signal'][0]
        return agents_data

    def adapt_signal_for_next_timestep(self, signal):
        return np.roll(signal, -1)

    def plots(self):
        plt.figure()
        names = self.agents.keys()
        T = sorted(list(filter(lambda x: isinstance(x, datetime.datetime), self.data_bank.keys())))
        consumption = [
            sum([self.data_bank[t][n]['consumption'][0] for n in names]) for t in T
        ]
        plt.plot(T, consumption, label='microgrid consumption')
        for name in names:
            consumption = [
                [self.data_bank[t][name]['consumption'][0]] for t in T
            ]
            reward = sum(self.data_bank[t][name]['reward'] for t in T)
            plt.plot(T, consumption, label=f'{name} (reward: {reward:.2f})')
        plt.legend()
        plt.show()



class MyManager(Manager):
    def __init__(self, *args, **kwargs):
        Manager.__init__(self, *args, **kwargs)
        self.eps = 1e-2
        self.previous_consumptions = None

    def has_converged(self, agents_data):
        current_consumptions = np.array([a['consumption'] for a in agents_data.values()]).sum(axis=0).squeeze()
        if self.previous_consumptions is None:
            self.previous_consumptions = current_consumptions
            return False
        res = np.linalg.norm(self.previous_consumptions - current_consumptions) < self.eps
        self.previous_consumptions = current_consumptions
        return res

    def update_signal(self, signal, agents_data):
        current_consumptions = np.array([a['consumption'] for a in agents_data.values()]).sum(axis=0).squeeze()
        return signal + current_consumptions * 0.1

    def update_reward(self, now: datetime.datetime, agents_data: dict):
        total_consumption = sum([a['consumption'][0] for a in agents_data.values()])
        N = len(self.agents)
        for name, agent in self.agents.items():
            data = agents_data[name]
            data['reward'] += total_consumption * data['signal'][0] / N
        return agents_data

    def adapt_signal_for_next_timestep(self, signal):
        return np.roll(signal, -1)


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    solar_farm_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'pv': {
            'surface': 100,
            'location': "enpc",  # or (lat, long) in float
            'tilt': 30,  # in degree
            'azimuth': 180,  # in degree from North
            'tracking': None,  # None, 'horizontal', 'dual'
        }
    }
    station_config = {
        'pmax': 40,
        'evs': [
            {
                'capacity': 50,
                'pmax': 3,
            },
            {
                'capacity': 50,
                'pmax': 22,
            }
        ]
    }
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    agents = {
        'ferme': SolarFarmAgent(SolarFarmEnv(solar_farm_config=solar_farm_config, nb_pdt=N)),
        'evs': ChargingStationAgent(ChargingStationEnv(station_config=station_config, nb_pdt=N)),
        'industrie': IndustrialAgent(IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)),
    }
    manager = MyManager(agents,
                        delta_t=delta_t,
                        horizon=time_horizon,
                        simulation_horizon=datetime.timedelta(days=1),
                        max_iterations=10,
                        )
    manager.init_envs()
    manager.run()
    manager.plots()

    print(manager)