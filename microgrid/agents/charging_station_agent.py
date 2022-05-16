import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
import pulp
import numpy as np
from collections import defaultdict

class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env
        self.nb_pdt = env.nb_pdt

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

        diff = []
        for k in range(self.env.nb_evs):
            print(plugged_prevision_diff[k])
            diff.append([0] + np.diff(plugged_prevision_diff[k]).tolist())


            #Initialisation de IS_PLUGGED
            if plugged_prevision[k][0] == 1:
                IS_PLUGGED = True
            else:
                IS_PLUGGED = False
        print(tdep)
        print(tarr)

        for k in range(self.env.nb_evs):
            tdep = None
            tarr = None
            soc_ = soc[k]
            for t in (range(self.nb_pdt)):
                old_is_plugged = IS_PLUGGED

                if diff[k][t] == 1:
                    IS_PLUGGED = True
                elif diff[k][t] == -1 :
                    IS_PLUGGED = False


                # Definition des variables
                var_name = "l_charge" + str(t) + str(k)
                l_charge[k][t] = pulp.LpVariable(var_name, 0, evs_config[k].get("pmax"))
                var_name = "l_decharge" + str(t) + str(k)
                l_decharge[k][t] = pulp.LpVariable(var_name, -evs_config[k].get("pmax"), 0)
                var_name2 = "alpha" + str(t) + str(k)
                alpha[k][t] = pulp.LpVariable(var_name2, cat="Binary")

                # Defintion des contraintes
                """bool1 = tdep[k]<=tarr[k] and (t < tdep[k] or t >= tarr[k])
                print(bool1)
                bool2 = tdep[k]>tarr[k] and (t>=tarr[k] and t<tdep[k])
                print(bool2)"""

                #if t == 0:
                    #a[k][t] = 0 + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        #self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (delta_t / H)
                    #soc_ = soc

                if IS_PLUGGED :
                    soc_ = soc_ + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (delta_t / H)
                    const_name = "a<=C" + str(t) + str(k)
                    lp += soc_ <= self.env.evs[k].battery.capacity, const_name
                    const_name = "a>=0" + str(t) + str(k)
                    lp += soc_ >= 0, const_name


                const_name = "lcharge<=pmax*alpha" + str(t) + str(k)
                lp += l_charge[k][t] <= self.env.evs[k].battery.pmax * alpha[k][t], const_name
                const_name = "ldecharge>=-pmax(1-alpha)" + str(t) + str(k)
                lp += l_decharge[k][t] >= -self.env.evs[k].battery.pmax * (1 - alpha[k][t]), const_name

                is_gone = True

                if old_is_plugged != IS_PLUGGED and not IS_PLUGGED and is_gone :
                    tdep = t
                    const_name = "charge_depart-25%" + str(k)
                    lp += 0.25 * self.env.evs[k].battery.pmax <= l_charge[k][tdep] + l_decharge[k][tdep], const_name
                    soc_tdep = soc_
                    is_gone = False

                if old_is_plugged != IS_PLUGGED and IS_PLUGGED:
                    if tdep is not None :
                        tarr = t
                        const_name = "retourdelavoiture" + str(k)
                        lp += soc_ == soc_tdep - 4, const_name
                    else:  # Cas particulier, y en a t'il d'autres?
                        if tarr is not None:
                            const_name = "retourdelavoiture" + str(k)
                            lp += soc_ == soc - 4, const_name


        for t in (range(self.nb_pdt)):
            const_name = "total_l4<=40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(self.env.nb_evs)]) <= 40, const_name

            const_name = "total_l4>=-40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(self.env.nb_evs)]) >= -40, const_name


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
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
    ]
    station_config = {
        'pmax': 40,
        'evs': evs_config
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