import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
import pulp
import numpy as np
from collections import defaultdict
from microgrid.check_feasibility import check_charging_station_feasibility

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
        #soc = state.get("soc")
        soc = [5,10,10,10]
        manager_signal = state.get("manager_signal")
        date_time = state.get("datetime")
        H = datetime.timedelta(hours=1)
        nb_evs = 1
        #self.env.nb_evs


        lp = pulp.LpProblem("charging_station", pulp.LpMinimize)

        l_charge = defaultdict(dict)
        l_decharge = defaultdict(dict)
        alpha = defaultdict(dict)
        a = defaultdict(dict)
        ## Recherche des temps de départ et d'arrivée
        tdep_dico = {}
        tarr_dico = {}

        plugged_prevision_diff = plugged_prevision.copy()

        diff = []
        for k in range(nb_evs):
            print(plugged_prevision_diff[k])
            diff.append(np.diff(plugged_prevision_diff[k]).tolist() + [0])


            #Initialisation de IS_PLUGGED
            if plugged_prevision[k][0] == 1:
                IS_PLUGGED = True
            else:
                IS_PLUGGED = False

            #Initialisation de NEXT_IS_PLUGGED
            if plugged_prevision[k][1] == 1:
                NEXT_IS_PLUGGED = True
            else:
                NEXT_IS_PLUGGED = False


        for k in range(nb_evs):
            tdep = None
            tarrmoins1 = None
            soc_ = soc[k]
            is_gone = True
            for t in (range(self.nb_pdt)):

                # Definition des variables
                var_name = "l_charge" + str(t) + str(k)
                l_charge[k][t] = pulp.LpVariable(var_name, 0, self.env.evs_config[k].get("pmax"))
                var_name = "l_decharge" + str(t) + str(k)
                l_decharge[k][t] = pulp.LpVariable(var_name, -self.env.evs_config[k].get("pmax"), 0)
                var_name2 = "alpha" + str(t) + str(k)
                alpha[k][t] = pulp.LpVariable(var_name2, cat="Binary")

                var_name = "a" + str(t) + str(k)
                a[k][t] = pulp.LpVariable(var_name, 0, self.env.evs[k].battery.capacity)


                # Defintion des contraintes
                """bool1 = tdep[k]<=tarr[k] and (t < tdep[k] or t >= tarr[k])
                print(bool1)
                bool2 = tdep[k]>tarr[k] and (t>=tarr[k] and t<tdep[k])
                print(bool2)"""

                if t == 0:
                    const_name = "soc initial"
                    a[k][t] == soc[k], const_name


                if IS_PLUGGED :
                    soc_ = soc_ + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (self.env.delta_t / H)
                    const_name = "a<=C" + str(t) + str(k)
                    lp += soc_ <= self.env.evs[k].battery.capacity, const_name
                    const_name = "a>=0" + str(t) + str(k)
                    lp += soc_ >= 0, const_name

                '''if IS_PLUGGED and t>0:
                    const_name = "maj de a" + str(t) + str(k)
                    lp+=a[k][t] == a[k][t-1] + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (self.env.delta_t / H), const_name

                elif not IS_PLUGGED and t>0:
                    const_name = "maj de a" + str(t) + str(k)
                    lp+=a[k][t] == a[k][t-1], const_name
                const_name = "a<=C" + str(t) + str(k)
                lp += a[k][t] <= self.env.evs[k].battery.capacity, const_name
                const_name = "a>=0" + str(t) + str(k)
                lp += a[k][t] >= 0, const_name'''





                if IS_PLUGGED:
                    const_name = "lcharge<=pmax*alpha" + str(t) + str(k)
                    lp += l_charge[k][t] <= self.env.evs[k].battery.pmax * alpha[k][t], const_name
                    const_name = "ldecharge>=-pmax(1-alpha)" + str(t) + str(k)
                    lp += l_decharge[k][t] >= -self.env.evs[k].battery.pmax * (1 - alpha[k][t]), const_name

                else:
                    const_name = "lcharge=0" + str(t) + str(k)
                    lp += l_charge[k][t] ==0, const_name
                    const_name = "ldecharge=0" + str(t) + str(k)
                    lp += l_decharge[k][t] ==0, const_name


                # SI Départ
                if NEXT_IS_PLUGGED != IS_PLUGGED and IS_PLUGGED and is_gone :
                    tdep = t
                    tdep_dico[k] = tdep
                    const_name = "charge_depart_25%" + str(k)
                    lp += 0.25 * self.env.evs[k].battery.capacity <= soc_, const_name
                    soc_tdep = soc_
                    is_gone = False

                '''if NEXT_IS_PLUGGED != IS_PLUGGED and IS_PLUGGED and is_gone:
                    tdep = t+1
                    tdep_dico[k] = tdep
                    const_name = "charge_depart_25%" + str(k)
                    lp += 0.25 * self.env.evs[k].battery.capacity <= a[k][t], const_name'''


                # SI arrivée
                if NEXT_IS_PLUGGED != IS_PLUGGED and not IS_PLUGGED:
                    if tdep is not None :
                        tarrmoins1 = t
                        tarr_dico[k] = tarrmoins1 +1
                        const_name = "retourdelavoiture" + str(k) + str(t)
                        lp += soc_ == soc_tdep - 4, const_name
                    #else:  # Cas particulier, y en a t'il d'autres
                        #const_name = "retourdelavoiture" + str(k) + str(t)
                        #lp += soc_ == soc[k] - 4, const_name
                '''if NEXT_IS_PLUGGED != IS_PLUGGED and not IS_PLUGGED:
                    if tdep is not None:
                        tarrmoins1 = t
                        tarr_dico[k] = tarrmoins1 + 1
                        const_name = "retourdelavoiture" + str(k) + str(t)
                        lp += a[k][t] == a[k][tdep] - 4, const_name'''

                #Calcul du NEXT_IS_PLUGGED
                IS_PLUGGED = NEXT_IS_PLUGGED
                if t<self.nb_pdt-3:
                    if diff[k][t+2] == 1:
                        NEXT_IS_PLUGGED = True
                    elif diff[k][t+2] == -1 :
                        NEXT_IS_PLUGGED = False




        for t in (range(self.nb_pdt)):
            const_name = "total_l4<=40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(nb_evs)]) <= 40, const_name

            const_name = "total_l4>=-40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(nb_evs)]) >= -40, const_name


            # Creation de la fonction objectif
        lp.setObjective(pulp.lpSum(
            [(l_charge[k][t] + l_decharge[k][t]) * manager_signal[t] * (self.env.delta_t / H) for k in range(nb_evs) for
             t in range(self.nb_pdt)]))

        lp.solve()
        # a = self.env.action_space.sample()
        # return a

        results = np.zeros((nb_evs, self.nb_pdt))
        for k in range(nb_evs):
            for t in range(self.nb_pdt):
                results[k][t] = l_charge[k][t].value() + l_decharge[k][t].value()

        liste_soc = np.zeros((nb_evs,self.nb_pdt))
        for k in range(nb_evs):
            #liste_soc[k][0] = state.get("soc")[k]
            liste_soc[k][0] = 5

        for k in range(nb_evs):
            for t in range(1,self.nb_pdt):
                liste_soc[k][t] = liste_soc[k][t-1] + (self.env.evs[k].battery.efficiency * l_charge[k][t].value() + 1 / (self.env.evs[k].battery.efficiency) * l_decharge[k][t].value()) * (self.env.delta_t / H)
                if k in tarr_dico and t == tarr_dico[k]:
                    liste_soc[k][t] -= 4
        print(liste_soc)

        print("l_charge")
        l_chargetab = np.zeros((nb_evs, self.nb_pdt))
        for k in range(nb_evs):
            for t in range(self.nb_pdt):
                l_chargetab[k][t] = l_charge[k][t].value()
        print(l_chargetab)

        print("l_decharge")
        l_dechargetab = np.zeros((nb_evs, self.nb_pdt))
        for k in range(nb_evs):
            for t in range(self.nb_pdt):
                l_dechargetab[k][t] = l_decharge[k][t].value()
        print(l_dechargetab)

        print(tdep_dico )
        print(tarr_dico)
        print(results.shape)

        for k in range(nb_evs):
            if k not in tdep_dico:
                tdep_dico[k] = 47
            if k not in tarr_dico:
                tarr_dico[k] = 0

        print(np.array(tdep_dico.values()))
        print(np.array(tarr_dico.values()))
        print(type(np.array(tarr_dico.values())))
        i_ev = 0
        t_ev_dep = np.array(list(tdep_dico.values()))
        t_ev_arr = np.array(list(tarr_dico.values()))
        print(t_ev_arr[i_ev])
        print(results[i_ev, t_ev_dep[i_ev]+1:t_ev_arr[i_ev]-1])

        lp.writeLP(f"myoptipb_{datetime.datetime.now()}.lp" )
        cs_dep_soc_penalty, infeas_score, n_infeas_by_type, detailed_infeas_list = \
            check_charging_station_feasibility(results, 1,0, t_ev_dep,
        t_ev_arr, {"normal": self.env.evs[0].battery.pmax, "fast": 22},
        40*np.ones(1), 10*np.ones(1), self.env.evs[0].battery.efficiency,
        self.env.evs[0].battery.efficiency, 48,
        1800, 1000,
        40)

        print(detailed_infeas_list)
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

