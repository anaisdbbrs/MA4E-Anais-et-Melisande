import datetime
import os
import pandas as pd
import numpy as np
from random import randint


class DataCenter:
    @staticmethod
    def random():
        return DataCenter(scenario=randint(1, 30))

    def __init__(self,
                 scenario=1,
                 Tcom=60,
                 Tr=35,
                 rho=0.5,
                 EER=4,
                 max_transfert=10,
                 ):
        self.scenario = scenario
        self.hotwater_scenario = 1
        self.Tcom = Tcom + 273.15
        self.Tr = Tr + 273.15
        self.rho = rho
        self.COP_HP = self.Tcom/(self.Tcom - self.Tr) * self.rho
        self.EER = EER
        self.COP_CS = self.EER + 1
        self.max_transfert = max_transfert
        self.data = pd.read_csv(os.path.join(os.path.dirname(__file__), f'scenarios/data_center/data.csv'), delimiter=';')
        self.prices = pd.read_csv(os.path.join(os.path.dirname(__file__), f'scenarios/data_center/hotwater_prices.csv'), delimiter=';')

    def reset(self):
        self.scenario = randint(1, 10)

    def get_power(self, when: datetime.datetime, start: datetime.datetime):
        pdt = (when - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(minutes=30) + 1
        scenario = self.scenario
        if pdt % 241 == 0:
            scenario = scenario % 10 + 1
            pdt = pdt % 241 + 1
        __a = self.data.loc[self.data['scenario'] == scenario]
        __b = __a.loc[__a['time_slot'] == pdt, ['cons (kW)']]
        return __b.values[0][0]

    def get_price(self, when: datetime.datetime, start: datetime.datetime):
        pdt = (when - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(minutes=30) + 1
        scenario = self.hotwater_scenario
        if pdt % 337 == 0:
            scenario = scenario % 1 + 1
            pdt = pdt % 337 + 1
        __a = self.data.loc[self.data['scenario'] == scenario]
        __b = __a.loc[__a['time_slot'] == pdt, ['cons (kW)']]
        return __b.values[0][0]

    def get_conso_prevision(self, datetimes: [datetime.datetime]):
        res = []
        if len(datetimes) > 0:
            start = datetimes[0]
            res = np.array([self.get_power(x, start) for x in datetimes])
            pdt = (start - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(minutes=30)
            if pdt == 24 * 2 - 1:
                self.scenario = self.scenario % 10 + 1
        return res

    def get_prices_prevision(self, datetimes: [datetime.datetime]):
        res = []
        if len(datetimes) > 0:
            start = datetimes[0]
            res = np.array([self.get_price(x, start) for x in datetimes])
            pdt = (start - datetime.datetime.fromordinal(start.date().toordinal())) // datetime.timedelta(minutes=30)
            if pdt == 24 * 2 - 1:
                self.scenario = self.scenario % 10 + 1
        return res

    def get_max_alpha_t(self, when: datetime.datetime, delta_t):
        dt = delta_t / datetime.timedelta(hours=1)
        lit = self.get_power(when, when)
        hr = lit * self.COP_CS / self.EER
        lhp = hr / ((self.COP_HP - 1) * dt)
        hdc = lhp * self.COP_HP * dt
        return self.max_transfert / hdc


if __name__ == '__main__':
    b = DataCenter()
    now = datetime.datetime.now()
    dt = datetime.timedelta(minutes=30)
    print(b.get_power(now, now))
    print(b.get_conso_prevision([now + i * dt for i in range(48)]))
