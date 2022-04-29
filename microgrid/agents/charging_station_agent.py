import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
import pulp
import numpy as np
from collections import defaultdict

class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env
        self.nb_pdt = 96

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        rho = self.env.evs[0].battery.efficiency
        self.env.evs[0].battery.capacity
        plugged_prevision = state.get('is_plugged_prevision')
        soc = state.get("soc")
        manager_signal = state.get("manager_signal")
        date_time = state.get("datetime")
        H = datetime.timedelta(hours=1)

        lp = pulp.LpProblem("charging_station", pulp.LpMinimize)

        l_charge = defaultdict(dict)
        l_decharge = defaultdict(dict)
        alpha = defaultdict(dict)
        a = defaultdict(dict)
        ## Recherche des temps de départ et d'arrivée
        tdep = {}
        tarr = {}

        plugged_prevision_diff = plugged_prevision.copy()
        for k in range(self.env.nb_evs):
            diff = np.diff(plugged_prevision_diff[k])
            for t in range(self.nb_pdt - 1):
                if diff[t] == -1:
                    tdep[k] = t + 1
                if diff[t] == 1:
                    tarr[k] = t + 1

        for t in range(self.nb_pdt):
            for k in range(self.env.nb_evs):
                # Definition des variables
                var_name = "l_charge" + str(t) + str(k)
                l_charge[k][t] = pulp.LpVariable(var_name, 0, evs_config[k].get("pmax"))
                var_name = "l_decharge" + str(t) + str(k)
                l_decharge[k][t] = pulp.LpVariable(var_name, -evs_config[k].get("pmax"), 0)
                var_name2 = "alpha" + str(t) + str(k)
                alpha[k][t] = pulp.LpVariable(var_name2, cat="Binary")

                # Defintion des contraintes
                if t == 0:
                    a[k][t] = 0 + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (delta_t / H)
                elif t <= tdep[k] or t >= tarr[k]:
                    a[k][t] = a[k][t - 1] + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (delta_t / H)

                const_name = "a<=C" + str(t) + str(k)
                lp += a[k][t] <= self.env.evs[k].battery.capacity, const_name
                const_name = "a>=0" + str(t) + str(k)
                lp += a[k][t] >= 0, const_name

                const_name = "lcharge<=pmax*alpha" + str(t) + str(k)
                lp += l_charge[k][t] <= self.env.evs[k].battery.pmax * alpha[k][t], const_name
                const_name = "ldecharge>=-pmax(1-alpha)" + str(t) + str(k)
                lp += l_decharge[k][t] >= -self.env.evs[k].battery.pmax * (1 - alpha[k][t]), const_name

            const_name = "total_l4<=40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(self.env.nb_evs)]) <= 40, const_name

            const_name = "total_l4>=-40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(self.env.nb_evs)]) >= -40, const_name

        for k in range(self.env.nb_evs):
            const_name = "charge_depart-25%" + str(k)
            lp += 0.25 * self.env.evs[k].battery.pmax <= l_charge[k][tdep[k]] + l_decharge[k][tdep[k]], const_name

            const_name = "retourdelavoiture" + str(k)
            lp += a[k][tarr[k]] == a[k][tdep[k]] - 4, const_name

            # Creation de la fonction objectif
        lp.setObjective(pulp.lpSum(
            [(l_charge[k][t] + l_decharge[k][t]) * manager_signal[t] * (delta_t / H) for k in range(self.env.nb_evs) for
             t in range(self.nb_pdt)]))

        lp.solve()
        # a = self.env.action_space.sample()
        # return a

        results = np.zeros((self.env.nb_evs, self.nb_pdt))
        for k in range(self.env.nb_evs):
            for t in range(self.nb_pdt):
                results[k][t] = l_charge[k][t].value() + l_decharge[k][t].value()

        return np.array(results)


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    evs_config = [
        {
            'capacity': 50,
            'pmax': 3,
        },
        {
            'capacity': 50,
            'pmax': 22,
        }
    ]
    station_config = {
        'pmax': 40,
        'evs_config': evs_config
    }
    env = ChargingStationEnv(station_config=station_config, nb_pdt=N)
    agent = ChargingStationAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))