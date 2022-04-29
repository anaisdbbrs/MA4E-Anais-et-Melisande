# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from microgrid.agents.solar_farm_agent import SolarFarmAgent
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    battery_config = {
        'capacity': 100,
        'efficiency': 0.95,
        'pmax': 25,
    }
    pv_config = {
        'surface': 100,
        'location': "enpc",  # or (lat, long) in float
        'tilt': 30,  # in degree
        'azimuth': 180,  # in degree from North
        'tracking': None,  # None, 'horizontal', 'dual'
    }
    env = SolarFarmEnv(battery_config=battery_config, pv_config=pv_config, nb_pdt=24 * 96)
    agent = SolarFarmAgent(env)
    microgrid = MicroGridEnv({'solar_farm': (env, agent)}, nb_pdt=24 * 4)
    obs = microgrid.reset()
    print(obs)
    next_obs, reward, done, info = microgrid.step({'solar_farm': np.zeros(24*4)})
    print(next_obs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
