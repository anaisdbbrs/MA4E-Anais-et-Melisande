import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv
import pulp
import numpy as np

class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env
        self.nb_pdt = env.nb_pdt

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):

        lnf = {}
        lhp = {}
        hdc = {}
        hr = {}
        alpha={}
        p_vente = 10

        COP_HP = self.env.COP_HP
        EER = self.env.EER
        COP_CS = self.env.COP_CS
        max_transfert = self.env.max_transfert
        manager_signal = state.get("manager_signal")


        H = datetime.timedelta(hours=1)
        deltat = delta_t/H
        li = state.get("consumption_prevision")

        lp = pulp.LpProblem("data_center", pulp.LpMinimize)


        for t in range(self.nb_pdt):
            var_name = "alpha" + str(t)
            alpha[t] = pulp.LpVariable(var_name, 0, 1)
            lnf[t] = (1+1/(EER*deltat))*li[t]
            hr[t] = COP_CS/EER*li[t]
            lhp[t] = alpha[t]*hr[t]/((COP_HP-1)*deltat)
            hdc[t] = COP_HP*deltat*lhp[t]

            const_name = "hdc<=max_transfert" + str(t)
            lp+= hdc[t] <= max_transfert, const_name

        lp.setObjective(pulp.lpSum([(lnf[t]+lhp[t])*manager_signal[t]-hdc[t]*p_vente for t in range (self.nb_pdt) ]))

        lp.solve()

        return np.array(alpha.values())



if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    data_center_config = {
        'scenario': 10,
    }
    env = DataCenterEnv(data_center_config, nb_pdt=N)
    agent = DataCenterAgent(env)
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