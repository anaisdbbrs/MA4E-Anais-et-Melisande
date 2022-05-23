import random

def get_configs(seed):
    random.seed(seed)
    solar_farm_config = {
        'battery': {
            'capacity': 30,
            'efficiency': 0.95,
            'pmax': 10,
            'pmin': -10,
        },
        'pv': {
            'surface': 100,
            'location': "enpc",  # or (lat, long) in float
            'tilt': 30,  # in degree
            'azimuth': 180,  # in degree from North
            'tracking': None,  # None, 'horizontal', 'dual'
        }
    }
    station_config = {
        'pmax': 40,
        'evs': [
            {
                'ev': 1,
                'day': random.randint(60, 110),
                'capacity': 40,
                'pmax': 22.,
                'pmin': -22.,
            },
            {
                'ev': 2,
                'day': random.randint(60, 110),
                'capacity': 40,
                'pmax': 22.,
                'pmin': -22.,
            },
            {
                'ev': 3,
                'day': random.randint(60, 110),
                'capacity': 40,
                'pmax': 3.,
                'pmin': -3.,
            },
            {
                'ev': 4,
                'day': random.randint(60, 110),
                'capacity': 40,
                'pmax': 3.,
                'pmin': -3.,
            },
        ]
    }
    industrial_config = {
        'battery': {
            'capacity': 60,
            'efficiency': 0.95,
            'pmax': 10,
            'pmin': -10.,
        },
        'building': {
            'site': 2,
            'scenario': random.randint(2, 10)
        }
    }
    data_center_config = {
        'scenario': random.randint(2, 8)
    }
    return {
        'solar_farm_config': solar_farm_config,
        'station_config': station_config,
        'industrial_config': industrial_config,
        'data_center_config': data_center_config
    }
