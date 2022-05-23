import copy

from calc_output_metrics import subselec_dict_based_on_lastlevel_keys, suppress_last_key_in_per_actor_bills, \
    calc_per_actor_bills, calc_microgrid_collective_metrics,  calc_two_metrics_tradeoff_last_iter, \
    get_best_team_per_region, get_improvement_traj

if __name__ =="__main__":
    import os
    import numpy as np
    import datetime

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
    contracted_p_tariffs = {6: 123.6, 9: 151.32, 12: 177.24, 15: 201.36, 18: 223.68, 24: 274.68, 30: 299.52, 36: 337.56}
    delta_t_s = 1800

    # calculate per-actor bill
    n_t = len(load_profiles[1][1]["grand_nord"][1]["champions"][1]["ferme"])
    purchase_price = 0.10 + 0.1 * np.random.rand(n_t)
    sale_price = 0.05 + 0.1 * np.random.rand(n_t)
    mg_price_signal = np.random.rand(n_t)

    per_actor_bills = calc_per_actor_bills(load_profiles=load_profiles, purchase_price=purchase_price,
                                           sale_price=sale_price, mg_price_signal=mg_price_signal, delta_t_s=delta_t_s)

    microgrid_prof, microgrid_pmax, collective_metrics = \
        calc_microgrid_collective_metrics(load_profiles=load_profiles, contracted_p_tariffs=contracted_p_tariffs,
                                          emission_rates=50*np.ones(n_t), delta_t_s=delta_t_s)

    # calculate cost, autonomy tradeoff
    aggreg_operations = {"cost": sum, "autonomy_score": np.mean}
    # get external (real) bills
    per_actor_bills_external = subselec_dict_based_on_lastlevel_keys(my_dict=copy.deepcopy(per_actor_bills),
                                                                     last_level_selected_keys=["external"])
    per_actor_bills_external = suppress_last_key_in_per_actor_bills(per_actor_bills=per_actor_bills_external,
                                                                    last_key="external")
    # and the internal ones, used for coord. into the microgrid
    per_actor_bills_internal = subselec_dict_based_on_lastlevel_keys(my_dict=copy.deepcopy(per_actor_bills),
                                                                     last_level_selected_keys=["internal"])
    per_actor_bills_internal = suppress_last_key_in_per_actor_bills(per_actor_bills=per_actor_bills_internal,
                                                                    last_key="internal")

    cost_autonomy_tradeoff = calc_two_metrics_tradeoff_last_iter(per_actor_bills=per_actor_bills_external,
                                                                 collective_metrics=collective_metrics, metric_1="cost",
                                                                 metric_2="autonomy_score", aggreg_operations=aggreg_operations)

    # calculate cost, CO2 emissions tradeoff
    aggreg_operations = {"cost": sum, "co2_emis": np.mean}
    cost_co2emis_tradeoff = calc_two_metrics_tradeoff_last_iter(per_actor_bills=per_actor_bills_external,
                                                                collective_metrics=collective_metrics, metric_1="cost",
                                                                metric_2="co2_emis", aggreg_operations=aggreg_operations)

    # Get best team per region
    coll_metrics_weights = {"pmax_cost": 1 / 365, "autonomy_score": 1,
                            "mg_transfo_aging": 0, "n_disj": 0, "co2_emis": 1}
    team_scores, best_teams_per_region, coll_metrics_names = \
        get_best_team_per_region(per_actor_bills=per_actor_bills_external, collective_metrics=collective_metrics,
                                 coll_metrics_weights=coll_metrics_weights)

    current_dir = os.getcwd()
    result_dir = os.path.join(current_dir, "run_synthesis")
    date_of_run = datetime.datetime.now()
    idx_run = 1
    coord_method = "price_decomposition"
    regions_map_file = os.path.join(current_dir, "images", "pv_regions_no_names.png")

    from create_ppt_summary_of_run import PptSynthesis
    import pandas as pd
    init_date = datetime.datetime(2022,5,17)
    dates = [init_date + datetime.timedelta(seconds=i*delta_t_s) for i in range(n_t)]
    ppt_synthesis = PptSynthesis(result_dir=result_dir, date_of_run=date_of_run, idx_run=idx_run,
                                 optim_period=pd.date_range(dates[0], dates[-1], freq=f"{delta_t_s}s"),
                                 coord_method=coord_method, regions_map_file=regions_map_file)

    # get "improvement trajectory"
    list_of_run_dates = [datetime.datetime.strptime(elt[4:], "%Y-%m-%d_%H%M") \
                         for elt in os.listdir(os.path.join(current_dir, "run_synthesis")) \
                         if (os.path.isdir(elt) and elt.startswith("run_"))]
    scores_traj = get_improvement_traj(current_dir, list_of_run_dates,
                                       list(team_scores))

    pv_prof = np.random.rand(n_t)
    ppt_synthesis.create_summary_of_run_ppt(pv_prof=pv_prof, load_profiles=load_profiles,
                                            microgrid_prof=microgrid_prof, microgrid_pmax=microgrid_pmax,
                                            per_actor_bills_internal=per_actor_bills_internal,
                                            cost_autonomy_tradeoff=cost_autonomy_tradeoff,
                                            cost_co2emis_tradeoff=cost_co2emis_tradeoff, team_scores=team_scores,
                                            best_teams_per_region=best_teams_per_region, scores_traj=scores_traj)