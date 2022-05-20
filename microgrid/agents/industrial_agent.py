import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv
import numpy as np
import pulp

class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env
        self.nb_pdt = env.nb_pdt

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        consumption_prevision = state.get("consumption_prevision")
        soc = state.get("soc")
        manager_signal = state.get("manager_signal")
        date_time = state.get("datetime")
        H = datetime.timedelta(hours=1)
        pmax = self.env.battery.pmax
        efficiency = self.env.battery.efficiency
        capacity = self.env.battery.capacity

        lp = pulp.LpProblem("industrial", pulp.LpMinimize)

        l_charge = {}
        l_decharge = {}
        alpha = {}
        a = {}
        for t in range(self.nb_pdt):

            # Definition des variables
            var_name = "l_charge" + str(t)
            l_charge[t] = pulp.LpVariable(var_name, 0, pmax)
            var_name = "l_decharge" + str(t)
            l_decharge[t] = pulp.LpVariable(var_name, -pmax, 0)
            var_name2 = "alpha" + str(t)
            alpha[t] = pulp.LpVariable(var_name2, cat="Binary")

            # Defintion des contraintes
            if t == 0:
                a[t] = 0 + (efficiency * l_charge[t] + 1 / efficiency *
                            l_decharge[t]) * (self.env.delta_t / H)
            else:
                a[t] = a[t - 1] + (
                            efficiency * l_charge[t] + 1 / efficiency *
                            l_decharge[t]) * (self.env.delta_t / H)

            const_name = "a<=C" + str(t)
            lp += a[t] <= capacity, const_name
            const_name = "a>=0" + str(t)
            lp += a[t] >= 0, const_name

            const_name = "lcharge<= pmax*alpha" + str(t)
            lp += l_charge[t] <= pmax * alpha[t], const_name
            const_name = "ldecharge >= -pmax(1-alpha)" + str(t)
            lp += l_decharge[t] >= -pmax * (1 - alpha[t]), const_name

            # Creation de la fonction objectif
        lp.setObjective(pulp.lpSum(
            [(consumption_prevision[t] + l_charge[t] + l_decharge[t]) * manager_signal[t] * (self.env.delta_t / H) for t in
             range(self.nb_pdt)]))

        lp.solve()
        # a = self.env.action_space.sample()
        # return a

        results = [0] * self.nb_pdt
        for t in range(self.nb_pdt):
            results[t] = l_charge[t].value() + l_decharge[t].value()

        return np.array(results)


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
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
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
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