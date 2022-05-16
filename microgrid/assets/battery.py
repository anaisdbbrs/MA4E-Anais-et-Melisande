import datetime
import enum
from random import uniform


class BatteryState(enum.IntEnum):
    """
    Enum for the battery states
    """
    OK = 0
    OVERCHARGED = 1
    UNDERCHARGED = 2
    OVERPOWERED = 4
    PMAX_EXCEEDED = 8


class Battery:
    def __init__(self, capacity=100, pmax=100, pmin=None, efficiency=0.95, initial_soc=lambda: uniform(0, 0)):
        self.capacity = capacity
        self.pmax = pmax
        self.pmin = pmin if pmin is not None else -pmax
        self.efficiency = efficiency
        self.initial_soc = initial_soc
        self.soc = initial_soc() * capacity

    def reset(self):
        self.soc = self.initial_soc() * self.capacity

    def power_with_efficiency(self, power, forward=True):
        if (forward and power > 0) or (not forward and power < 0):
            return power * self.efficiency
        return power / self.efficiency

    def check_power(self, power, delta_t=datetime.timedelta(minutes=30)):
        state = BatteryState.OK
        H = datetime.timedelta(hours=1)
        effective_power = power
        if power > self.pmax or power < self.pmin:
            effective_power = self.pmax if power > 0 else self.pmin
            state += BatteryState.OVERPOWERED
        dc_power = self.power_with_efficiency(power)
        next_soc = self.soc + dc_power * (delta_t / H)
        if next_soc > self.capacity:
            dc_pmax = (self.capacity - self.soc) / (delta_t / H)
            state += BatteryState.OVERCHARGED
            ac_pmax = self.power_with_efficiency(dc_pmax, forward=False)
            effective_power = min(effective_power, ac_pmax)
        elif next_soc < 0:
            dc_pmin = - self.soc / (delta_t / H)
            state += BatteryState.UNDERCHARGED
            ac_pmin = self.power_with_efficiency(dc_pmin, forward=False)
            effective_power = max(effective_power, ac_pmin)
        return effective_power, state

    def charge(self, power, delta_t=datetime.timedelta(minutes=30)):
        effective_power, state = self.check_power(power, delta_t)
        H = datetime.timedelta(hours=1)
        self.soc += self.power_with_efficiency(effective_power) * (delta_t / H)
        return self.soc, effective_power, state
