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

        ################################################Récupération des paramètres de l'environnement###########################################

        # rendement (opération de charge et décharge)
        rho = self.env.evs[0].battery.efficiency

        # capacité batterie d'un ev (kWh)
        capacity = self.env.evs[0].battery.capacity

        # état de charge initial
        # soc = state.get("soc")
        soc = [5, 10, 10, 10]

        nb_evs = 1  # test... mettre: self.env.nb_evs

        # pour le calcul des tdep et tarr
        plugged_prevision = state.get('is_plugged_prevision')

        # signal prix et date (début du run)
        manager_signal = state.get("manager_signal")
        date_time = state.get("datetime")
        H = datetime.timedelta(hours=1)

        ####################################################Déclaration du problème##########################################################
        lp = pulp.LpProblem("charging_station", pulp.LpMinimize)

        # puissance de charge et de décharge de la batterie (positive et négative)
        l_charge = defaultdict(dict)
        l_decharge = defaultdict(dict)
        alpha = defaultdict(dict)  # booléen: soit on charge, soit on décharge, pas en simultané!

        # état de charge de la batterie (dictionnaire de dictionnaires)
        a = defaultdict(dict)

        ####################################Recherche des temps de départ et d'arrivée######################################################
        # ces dicos nous disent si pour l'ev k à t, il y a un départ, une arrivée..., avec _date: il récupère la date pour chaque ev
        tdep_dico = {}
        tarr_dico = {}
        tdeo_dico_date = {}
        tarr_dico_date = {}

        ##pas utile en fait? ##
        plugged_prevision_diff = plugged_prevision.copy()
        diff = []
        for k in range(nb_evs):
            print(plugged_prevision_diff[k])
            diff.append(np.diff(plugged_prevision_diff[k]).tolist() + [0])
            ##pas utile en fait? ##

            # Initialisation de IS_PLUGGED
            if plugged_prevision[k][0] == 1:
                IS_PLUGGED = True
            else:
                IS_PLUGGED = False

            # Initialisation de NEXT_IS_PLUGGED
            if plugged_prevision[k][1] == 1:
                NEXT_IS_PLUGGED = True
            else:
                NEXT_IS_PLUGGED = False

        for k in range(nb_evs):

            tdep = None
            tarrmoins1 = None
            # soc initial de l'ev k
            soc_ = soc[k]

            is_gone = False

            for t in (range(self.nb_pdt)):

                #############################Définition des variables (voir signification au-dessus)#############################################
                var_name = "l_charge" + str(t) + str(k)
                l_charge[k][t] = pulp.LpVariable(var_name, 0, self.env.evs_config[k].get("pmax"))

                var_name = "l_decharge" + str(t) + str(k)
                l_decharge[k][t] = pulp.LpVariable(var_name, -self.env.evs_config[k].get("pmax"), 0)

                var_name = "alpha" + str(t) + str(k)
                alpha[k][t] = pulp.LpVariable(var_name, cat="Binary")

                # inutile: soc est calculé à partir de variables... (lcharge, ldecharge...)
                var_name = "a" + str(t) + str(k)
                a[k][t] = pulp.LpVariable(var_name, 0, self.env.evs[k].battery.capacity)

                #####################################Définition des contraintes###########################################################

                # !!!!!!!!!!!!!!!! OB : je ne vois pas cette contrainte dans votre .lp... bizarre
                if t == 0:
                    const_name = "soc initial"
                    soc_ = soc[k]
                    a[k][t] == soc_, const_name

                # On détecte un DEPART (=débranché) au temps t+1
                if NEXT_IS_PLUGGED != IS_PLUGGED and IS_PLUGGED:
                    tdepmoins1 = t
                    tdep_dico[k][t + 1] = 1
                    tdeo_dico_date[k] = tdepmoins1 + 1

                    # Contrainte: Batterie chargée au départ, elle s'applique au dernier "1" avant le "0" du départ: instant AVANT le départ (le "0")
                    # les variables lcharge et ldecharge s'optimiseront en ayant en tête cette contrainte (donc pas besoin d'amende)
                    const_name = "charge_depart_25%" + str(k)
                    lp += 0.25 * self.env.evs[k].battery.capacity <= soc_, const_name
                    # lp += 0.25 * self.env.evs[k].battery.capacity <= a[k][t], const_name c'est meiux???
                    soc_tdep = soc_  # garder le soc de départ en mémoire (en fait, c'est déjà géré par la dynamique: faire appel à a[k][tdep]
                    # ou encore à a[k][t] ligne 139 pour la contrainte "retour de la voiture"

                    is_gone = True

                """focus sur "is_gone": variable booléeenne utile dans le cas d'une journée de type 1 classique: la voiture part (1 départ) et 
                revient (1 arrivée), is_gone = true dit que la voiture est partie, on est à t tq tdep<=t<tarr"""
                """ne sert à rien? si ça sert... mais ça se répète avec tdep_dico"""

                # On détecte une ARRIVEE (=branché) au temps t+1, là à t: on n'est pas branché
                if NEXT_IS_PLUGGED != IS_PLUGGED and not IS_PLUGGED:
                    # à t, l'ev k est encore débranché
                    tarrmoins1 = t
                    tarr_dico[k][t + 1] = 1  # le véhicule arrive
                    tarr_dico_date[k] = tarrmoins1 + 1

                const_name = "lcharge=0" + str(t) + str(k)
                lp += l_charge[k][t] == 0, const_name
                const_name = "ldecharge=0" + str(t) + str(k)
                lp += l_decharge[k][t] == 0, const_name

                # l'ev est déjà parti dans la journée (à partir de "now"), cas de la journée classique
                # on gère le "-4", pour les autres types de journée, on suppose que le "-4" est géré par l'environnement?
                # dynamique de la batterie
                if is_gone:  # le côté chronologique de l'algorithme? pas seulement écriture de contraintes?

                    const_name = "retourdelavoiture" + str(k) + str(t)
                    # lp += soc_ == soc_tdep - 4, const_name, ligne en dessous c'est mieux?
                    soc_ = soc_tdep - 4
                    lp += a[k][t] == soc_, const_name

                # Au pas de temps t, l'ev k est branché
                if IS_PLUGGED:
                    # Charge ou bien décharge...
                    const_name = "lcharge<=pmax*alpha" + str(t) + str(k)
                    lp += l_charge[k][t] <= self.env.evs[k].battery.pmax * alpha[k][t], const_name
                    const_name = "ldecharge>=-pmax(1-alpha)" + str(t) + str(k)
                    lp += l_decharge[k][t] >= -self.env.evs[k].battery.pmax * (1 - alpha[k][t]), const_name

                    # MAJ soc_ (scalaire), stockage dans a[k][t]
                    soc_ = soc_ + (self.env.evs[k].battery.efficiency * l_charge[k][t] + 1 / (
                        self.env.evs[k].battery.efficiency) * l_decharge[k][t]) * (self.env.delta_t / H)
                    a[k][t] = soc_

                    # Contrainte état de charge de la batterie
                    const_name = "a<=C" + str(t) + str(k)
                    lp += a[k][t] <= self.env.evs[k].battery.capacity, const_name
                    const_name = "a>=0" + str(t) + str(k)
                    lp += a[k][t] >= 0, const_name

                # Ou bien au pas de temps t, l'ev k est débranché ( = parti), donc charge et décharge nulles et le temps suivant n'est pas une arrivée
                # si le temps suivant est une arrivée, la MAJ de soc_ est différente: voir "on détecte une ARRIVEE"
                if not IS_PLUGGED and tarr_dico[k][t + 1] is None and t < self.nb_pdt - 1:
                    const_name = "lcharge=0" + str(t) + str(k)
                    lp += l_charge[k][t] == 0, const_name
                    const_name = "ldecharge=0" + str(t) + str(k)
                    lp += l_decharge[k][t] == 0, const_name

                    # dynamique de la batterie
                    a[k][t] = soc_

                # Calcul du NEXT_IS_PLUGGED
                # !!!!!!!!!!!!!!!!! OB : ici je pense qu'il y a une erreur d'indice car de base np.diff donne comme élément [0] le statut pour t=1
                # (décalage pris avec l'opération de diff) -> prendre donc seulement diff[k][t+1]... et s'arrêter à nb_pdt-2
                """" à t: IS_PLUGGED, pour t+1: NEXT_IS_PLUGGED, puis MAJ: t+1 et t+2"""
                IS_PLUGGED = NEXT_IS_PLUGGED
                """if t<self.nb_pdt-3:
                    if diff[k][t+2] == 1:
                        NEXT_IS_PLUGGED = True
                    elif diff[k][t+2] == -1 :
                        NEXT_IS_PLUGGED = False"""
                # c'est plutôt avec plugged_prevision?
                if t < self.nb_pdt - 2:
                    if plugged_prevision[k][t + 1] == 1:
                        NEXT_IS_PLUGGED = True
                    else:
                        NEXT_IS_PLUGGED = False
                # convention, condition aux bords pour le derbier pas de temps
                # NEXT_IS_PLUGGED ne change pas... (la voiture à t = 48 sera considérée dans l'était de t = 47

        # Contrainte de la puissance de charge aggrégée
        for t in (range(self.nb_pdt)):
            const_name = "total_l4<=40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(nb_evs)]) <= 40, const_name

            const_name = "total_l4>=-40" + str(t) + str(k)
            lp += pulp.lpSum([l_charge[k][t] + l_decharge[k][t] for k in range(nb_evs)]) >= -40, const_name

            ##############################################Creation de la fonction objectif######################################################
        lp.setObjective(pulp.lpSum(
            [(l_charge[k][t] + l_decharge[k][t]) * manager_signal[t] * (self.env.delta_t / H) for k in range(nb_evs) for
             t in range(self.nb_pdt)]))

        ###############################################Résolution###########################################################################
        lp.solve()
        # a = self.env.action_space.sample()
        # return a

        #######################################################Affichage des valeurs des variables#############################################"
        results = np.zeros((nb_evs, self.nb_pdt))
        for k in range(nb_evs):
            for t in range(self.nb_pdt):
                results[k][t] = l_charge[k][t].value() + l_decharge[k][t].value()

        liste_soc = np.zeros((nb_evs, self.nb_pdt))
        for k in range(nb_evs):
            # liste_soc[k][0] = state.get("soc")[k]
            liste_soc[k][0] = 5

        for k in range(nb_evs):
            for t in range(1, self.nb_pdt):
                liste_soc[k][t] = liste_soc[k][t - 1] + (self.env.evs[k].battery.efficiency * l_charge[k][t].value() + 1 / (
                    self.env.evs[k].battery.efficiency) * l_decharge[k][t].value()) * (self.env.delta_t / H)
                if k in tarr_dico_date and t == tarr_dico_date[k]:
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

        print(tdep_dico_date)
        print(tarr_dico_date)

        for k in range(nb_evs):
            if k not in tdep_dico_date:
                tdep_dico_date[k] = 47
            if k not in tarr_dico_date:
                tarr_dico_date[k] = 0

        i_ev = 0
        t_ev_dep = np.array(list(tdep_dico_date.values()))
        t_ev_arr = np.array(list(tarr_dico_date.values()))

        lp.writeLP(f"myoptipb_{datetime.datetime.now()}.lp")
        cs_dep_soc_penalty, infeas_score, n_infeas_by_type, detailed_infeas_list = \
            check_charging_station_feasibility(results, 1, 0, t_ev_dep, t_ev_arr,
                                               {"normal": self.env.evs[0].battery.pmax, "fast": 22},
                                               40 * np.ones(1), 10 * np.ones(1), self.env.evs[0].battery.efficiency,
                                               self.env.evs[0].battery.efficiency, 48,
                                               1800, 1000, 40)

        print(detailed_infeas_list)
        return np.array(results)


####################################################################################################################################################


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
            'git stapmax': 3,
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
    for i in range(N * 2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))


