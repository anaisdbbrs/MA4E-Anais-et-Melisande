import numpy as np
import pandas as pd
from calc_output_metrics import save_df_to_csv

def save_load_profiles(load_profiles: dict, team_name: str, filename: str):
    """

    Args:
        :param load_profiles: dictionary with keys 1. Industrial Cons. scenario;
        2. Data Center scenario; 3. PV scenario; 4. EV scenario; 5. microgrid team name;
        6. Iteration; 7. actor type (N.B. charging_station_1, charging_station_2...
        if multiple charging stations in this microgrid) and values the associated
        load profile (kW)
        :param filename: file in which the load profiles have to be saved

    Returns:

    """

    first_ic_scen = list(load_profiles.keys())[0]
    first_dc_scen = list(load_profiles[first_ic_scen].keys())[0]
    first_pv_scen = list(load_profiles[first_ic_scen][first_dc_scen].keys())[0]
    first_ev_scen = list(load_profiles[first_ic_scen][first_dc_scen][first_pv_scen].keys())[0]
    last_iter = max(load_profiles[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name].keys())

    all_agents = list(load_profiles[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name][last_iter])

    n_t = len(load_profiles[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name][last_iter][all_agents[0]])

    dict_to_csv = {"agent": [], "consumption (kW)": []}

    for agent in all_agents:
        dict_to_csv["agent"].extend(n_t * [agent])
        dict_to_csv["consumption (kW)"].extend(list(load_profiles[first_ic_scen][first_dc_scen][first_pv_scen]
                                                    [first_ev_scen][team_name][last_iter][agent]))

    dict_to_csv["time_slot"] = list(np.tile(np.arange(n_t) + 1, len(all_agents)))

    df_to_csv = pd.DataFrame(dict_to_csv)

    col_order = ["agent", "time_slot", "consumption (kW)"]

    save_df_to_csv(df=df_to_csv, fields_tb_rounded=["consumption (kW)"], col_order=col_order, filename=filename)

def save_perf_metrics(collective_metrics: dict, per_actor_bills_external: dict, team_name: str, filename: str):

    first_ic_scen = list(per_actor_bills_external.keys())[0]
    first_dc_scen = list(per_actor_bills_external[first_ic_scen].keys())[0]
    first_pv_scen = list(per_actor_bills_external[first_ic_scen][first_dc_scen].keys())[0]
    first_ev_scen = list(per_actor_bills_external[first_ic_scen][first_dc_scen][first_pv_scen].keys())[0]
    last_iter = max(per_actor_bills_external[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name].keys())

    all_agents = list(per_actor_bills_external[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name][last_iter])

    dict_to_csv = {"bill": 0}

    for elt_col in ["pmax_cost", "autonomy_score", "co2_emis"]:
        dict_to_csv[elt_col] = collective_metrics[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name][last_iter][elt_col]

    for agent in all_agents:
        dict_to_csv["bill"] += per_actor_bills_external[first_ic_scen][first_dc_scen][first_pv_scen][first_ev_scen][team_name][last_iter][agent]

    for elt in dict_to_csv:
        dict_to_csv[elt] = [dict_to_csv[elt]]

    df_to_csv = pd.DataFrame(dict_to_csv)

    col_order = ["bill", "pmax_cost", "autonomy_score", "co2_emis"]

    save_df_to_csv(df=df_to_csv, fields_tb_rounded=["bill", "pmax_cost", "autonomy_score", "co2_emis"],
                   col_order=col_order, filename=filename)


if __name__ =="__main__":
    load_profiles = {1:
                         {1:
                              {'grand_nord':
                                   {1:
                                        {'champions':
                                             {1:
                                                  {'ferme': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                             -0.21, -1.31, -2.67, -4.16, -5.69, -7.20, -8.63, -9.94,
                                                             -11.09, -12.05, -12.80, -13.30, -13.57]),
                                                   'evs': np.array([17.83, 22.24, 24.97, 40.00, 35.14, 11.71, 24.67, 1.30, 14.61,
                                                           2.10, 5.24, 1.86, 4.12, 5.14, 2.63, 3.61, 1.94, 4.47, 3.97,
                                                           1.29, 2.79, 2.85, 2.67, 4.28]),
                                                   'industrie': np.array([51.51, 61.24, 53.53, 47.86, 48.50, 58.21, 51.73, 40.96,
                                                                 58.68, 51.46, 49.14, 48.13, 54.49, 47.25, 54.65, 49.06,
                                                                 51.79, 50.18, 54.30, 49.15, 55.42, 60.26, 53.66, 48.93]),
                                                   'datacenter': np.array([2.29, 19.72, -0.78, 1.79, 15.69, 8.96, 9.17, -3.34, 14.19,
                                                                  8.92, -1.98, 15.56, 3.54, 2.47, 3.35, -0.54, 4.50, 5.10,
                                                                  -2.17, 18.34, 5.30, 3.78, 9.54, 5.81])
                                                   }
                                              },
                                        'pir':
                                             {1:
                                                  {'ferme': 0.5*np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                                                      -0.21, -1.31, -2.67, -4.16, -5.69, -7.20, -8.63,
                                                                      -9.94,
                                                                      -11.09, -12.05, -12.80, -13.30, -13.57]),
                                                   'evs': 0.5*np.array(
                                                       [17.83, 22.24, 24.97, 40.00, 35.14, 11.71, 24.67, 1.30, 14.61,
                                                        2.10, 5.24, 1.86, 4.12, 5.14, 2.63, 3.61, 1.94, 4.47, 3.97,
                                                        1.29, 2.79, 2.85, 2.67, 4.28]),
                                                   'industrie': 0.5*np.array(
                                                       [51.51, 61.24, 53.53, 47.86, 48.50, 58.21, 51.73, 40.96,
                                                        58.68, 51.46, 49.14, 48.13, 54.49, 47.25, 54.65, 49.06,
                                                        51.79, 50.18, 54.30, 49.15, 55.42, 60.26, 53.66, 48.93]),
                                                   'datacenter': 0.5*np.array(
                                                       [2.29, 19.72, -0.78, 1.79, 15.69, 8.96, 9.17, -3.34, 14.19,
                                                        8.92, -1.98, 15.56, 3.54, 2.47, 3.35, -0.54, 4.50, 5.10,
                                                        -2.17, 18.34, 5.30, 3.78, 9.54, 5.81])
                                                   }
                                              }
                                         }
                                    }
                               }
                          }
                     }

    save_load_profiles(load_profiles=load_profiles, team_name="pir", filename="agent_loads")

    collective_metrics = {1:
                              {1:
                                   {'grand_nord':
                                        {1:
                                             {'pir':
                                                  {1:
                                                       {"pmax_cost": 1,
                                                        "autonomy_score": 2,
                                                        "co2_emis": 3
                                                        }
                                                   }
                                              }
                                         }
                                    }
                               }
                          }

    per_actor_bills_external = {1:
                                    {1:
                                         {'grand_nord':
                                              {1:
                                                   {'pir':
                                                        {1:
                                                             {'ferme': 1,
                                                              'evs': 2,
                                                              'industrie': 3,
                                                              'datacenter': 4
                                                              }
                                                         }
                                                    }
                                               }
                                          }
                                     }
                                }

    save_perf_metrics(collective_metrics=collective_metrics, per_actor_bills_external=per_actor_bills_external,
                      team_name="pir", filename="perf_metrics")
