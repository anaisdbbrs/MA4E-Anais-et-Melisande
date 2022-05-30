import logging
import os
import copy
from collections import defaultdict

import numpy as np
import datetime

import pandas as pd
import tqdm

from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
from microgrid.environments.data_center.data_center_env import DataCenterEnv
from microgrid.environments.industrial.industrial_env import IndustrialEnv
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
from matplotlib import pyplot as plt
from create_ppt_summary_of_run import PptSynthesis, set_to_multiple_scenarios_format
from calc_output_metrics import subselec_dict_based_on_lastlevel_keys, suppress_last_key_in_per_actor_bills, \
    calc_microgrid_collective_metrics, calc_two_metrics_tradeoff_last_iter, calc_per_actor_bills, \
    get_best_team_per_region, get_improvement_traj
from config import get_configs
from write import save_load_profiles,save_perf_metrics


class Manager:
    def __init__(self,
                 agents: dict,
                 start: datetime.datetime = datetime.datetime(2022, 5, 16, 0, 0, 0, 0),
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
        for pdt in tqdm.trange(N):
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


    def generate_load_profile(self, mg_team_name='champions'):
        iter_idx = 1
        dates = list(sorted(filter(lambda x: isinstance(x, datetime.datetime), self.data_bank.keys())))
        agents = list(filter(lambda x: not x.startswith('__'), self.data_bank[dates[0]].keys()))
        pv_prof = [self.data_bank[date]['ferme']['state']['pv_prevision'][0] for date in dates]
        load_profiles = {mg_team_name:
                             {iter_idx: {agent: np.array([self.data_bank[date][agent]['consumption'][0] for date in dates])
                                         for agent in agents
                                         }
                              }
                         }
        return load_profiles, dates, pv_prof

    def generate_summary_ppt(self, load_profiles, dates, pv_prof):
        # update dict. to fit with the multiple scenarios case
        fixed_scenario = (1, 1, "grand_nord", 1)
        load_profiles = set_to_multiple_scenarios_format(dict_wo_scenarios=load_profiles, fixed_scenario=fixed_scenario)

        # calculate microgrid profile, max power and collective metrics
        contracted_p_tariffs = {6: 123.6, 9: 151.32, 12: 177.24, 15: 201.36,
                                18: 223.68, 24: 274.68, 30: 299.52, 36: 337.56}

        # calculate per-actor bill
        n_t = len(dates)
        # TODO: update signal from run
        signal = np.random.rand(n_t)
        purchase_price = 0.10 + 0.1 * np.random.rand(n_t)
        sale_price = 0.05 + 0.1 * np.random.rand(n_t)

        delta_t_s = self.delta_t.total_seconds()
        per_actor_bills = calc_per_actor_bills(load_profiles=load_profiles, purchase_price=purchase_price,
                                               sale_price=sale_price, mg_price_signal=signal, delta_t_s=delta_t_s)

        microgrid_prof, microgrid_pmax, collective_metrics = \
            calc_microgrid_collective_metrics(load_profiles=load_profiles, contracted_p_tariffs=contracted_p_tariffs,
                                              emission_rates=50 * np.ones(n_t), delta_t_s=delta_t_s)

        # get external (real) bills
        per_actor_bills_external = subselec_dict_based_on_lastlevel_keys(my_dict=copy.deepcopy(per_actor_bills),
                                                                         last_level_selected_keys=["external"])
        per_actor_bills_external = suppress_last_key_in_per_actor_bills(per_actor_bills=per_actor_bills_external,
                                                                        last_key="external")
        # and the internal ones, used for coord. into the microgrid
        per_actor_bills_internal = subselec_dict_based_on_lastlevel_keys(my_dict=copy.deepcopy(per_actor_bills),
                                                                         last_level_selected_keys=["internal"])
        per_actor_bills_internal = suppress_last_key_in_per_actor_bills(per_actor_bills=per_actor_bills_internal,
                                                                        last_key="internal")

        # calculate cost, autonomy tradeoff
        aggreg_operations = {"cost": sum, "autonomy_score": np.mean}
        cost_autonomy_tradeoff = calc_two_metrics_tradeoff_last_iter(per_actor_bills=per_actor_bills_external,
                                                                     collective_metrics=collective_metrics,
                                                                     metric_1="cost",
                                                                     metric_2="autonomy_score",
                                                                     aggreg_operations=aggreg_operations)

        # calculate cost, CO2 emissions tradeoff
        aggreg_operations = {"cost": sum, "co2_emis": np.mean}
        cost_co2emis_tradeoff = calc_two_metrics_tradeoff_last_iter(per_actor_bills=per_actor_bills_external,
                                                                    collective_metrics=collective_metrics,
                                                                    metric_1="cost",
                                                                    metric_2="co2_emis",
                                                                    aggreg_operations=aggreg_operations)

        # Get best team per region
        coll_metrics_weights = {"pmax_cost": 1 / 365, "autonomy_score": 1,
                                "mg_transfo_aging": 0, "n_disj": 0, "co2_emis": 1}
        team_scores, best_teams_per_region, coll_metrics_names = \
            get_best_team_per_region(per_actor_bills=per_actor_bills_external, collective_metrics=collective_metrics,
                                     coll_metrics_weights=coll_metrics_weights)

        current_dir = os.getcwd()
        result_dir = os.path.join(current_dir, "run_synthesis")
        date_of_run = datetime.datetime.now()
        idx_run = 1
        coord_method = "price_decomposition"
        regions_map_file = os.path.join(current_dir, "images", "pv_regions_no_names.png")

        ppt_synthesis = PptSynthesis(result_dir=result_dir, date_of_run=date_of_run, idx_run=idx_run,
                                     optim_period=pd.date_range(dates[0], dates[-1], freq=f"{int(delta_t_s)}s"),
                                     coord_method=coord_method, regions_map_file=regions_map_file)

        # get "improvement trajectory"
        list_of_run_dates = [datetime.datetime.strptime(elt[4:], "%Y-%m-%d_%H%M") \
                             for elt in os.listdir(os.path.join(current_dir, "run_synthesis")) \
                             if (os.path.isdir(elt) and elt.startswith("run_"))]
        scores_traj = get_improvement_traj(current_dir, list_of_run_dates,
                                           list(team_scores))
        save_load_profiles(load_profiles, "pir", os.path.join(result_dir, f"profilConsos_{date_of_run.strftime('%Y-%m-%d_%H%M')}"))
        save_perf_metrics(collective_metrics,per_actor_bills_external,"pir",os.path.join(result_dir, f"PerfMetrics_{date_of_run.strftime('%Y-%m-%d_%H%M')}"))
        ppt_synthesis.create_summary_of_run_ppt(pv_prof=pv_prof, load_profiles=load_profiles,
                                                microgrid_prof=microgrid_prof, microgrid_pmax=microgrid_pmax,
                                                per_actor_bills_internal=per_actor_bills_internal,
                                                cost_autonomy_tradeoff=cost_autonomy_tradeoff,
                                                cost_co2emis_tradeoff=cost_co2emis_tradeoff, team_scores=team_scores,
                                                best_teams_per_region=best_teams_per_region, scores_traj=scores_traj)


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
    time_horizon = datetime.timedelta(days=1)  # taille de l'horizon glissant
    N = time_horizon // delta_t

    seed = 1234
    configs = get_configs(seed)

    import importlib
    load_profiles = {}
    dates = None
    pv_prof = None

    #teams = ['super_microgrid', 'les_grosses_sacoches', 'les_kssos', 'pir', 'microgrid_autonome', 'smart_grid']
    teams = ['pir']
    for team in teams:
        modSF = importlib.import_module(f'microgrid.agents.solar_farm_agent')
        modCS = importlib.import_module(f'microgrid.agents.charging_station_agent')
        modI = importlib.import_module(f'microgrid.agents.industrial_agent')
        modDC = importlib.import_module(f'microgrid.agents.data_center_agent')
        agents = {
            'ferme': modSF.SolarFarmAgent(SolarFarmEnv(solar_farm_config=configs['solar_farm_config'], nb_pdt=N)),
            'evs': modCS.ChargingStationAgent(ChargingStationEnv(station_config=configs['station_config'], nb_pdt=N)),
            'industrie': modI.IndustrialAgent(IndustrialEnv(industrial_config=configs['industrial_config'], nb_pdt=N)),
            'datacenter': modDC.DataCenterAgent(DataCenterEnv(data_center_config=configs['data_center_config'], nb_pdt=N)),
        }
        manager = MyManager(agents,
                            delta_t=delta_t,
                            horizon=time_horizon,
                            simulation_horizon=datetime.timedelta(days=1), # durée de la glissade
                            max_iterations=10, # nombre d'iterations de convergence des prix
                            )
        manager.run()
        #manager.plots()
        ld, dates, pv_prof = manager.generate_load_profile(team)
        load_profiles.update(ld)
    manager.generate_summary_ppt(load_profiles, dates, pv_prof)
