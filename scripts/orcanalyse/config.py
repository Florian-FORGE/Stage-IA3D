EXTREMUM_HEATMAP = {"Matrix" : (-0.2, 3), 
                    "RealMatrix" : (-0.95, 1.8), 
                    "OrcaMatrix" : (-0.75, 1.5)}

COLOR_CHART = {"insulation_count" : "blue", 
               "PC1" : "green", 
               "insulation_correl" : "red"}

WHICH_MATRIX = {"Matrix" : {"count" : "_obs_o_exp", 
                            "correl" : "_obs"}, 
                "OrcaMatrix" : {"count" : "obs_o_exp", 
                                "correl" : "obs"}, 
                "RealMatrix" : {"count" : "log_obs_o_exp", 
                                "correl" : "obs"}}

SMOOTH_MATRIX  = {"_get_insulation_score" : {"bool" : False, 
                                             "val" : {"RealMatrix" : 5, 
                                                      "OrcaMatrix" : 2}}, 
                  "correl_mat" : {"bool" : False, 
                                  "val" : {"RealMatrix" : 2, 
                                           "OrcaMatrix" : 1}}}

TLVs_HEATMAP = {"Matrix" : [0.2, 0.5], 
                "RealMatrix" : [0.4, 0.6], 
                "OrcaMatrix" : [0.2, 0.5]}

SUPERPOSED_PARAMETERS = {"correl_mat" : {"alpha" : {"wtd" : 0.75, 
                                                    "rdm" : 0.25}, 
                                         "color" : {"wtd" : "black",
                                                    "rdm" : "#CC9966"}},
                         "scores_regression" : {"alpha" : {"wtd" : 0.75, 
                                                           "rdm" : 0.25}, 
                                                "color" : {"insulation_count" : {"wtd" : "blue",
                                                                                "rdm" : "#6633FF"},
                                                            "PC1" : {"wtd" : "green",
                                                                     "rdm" : "#33FF66"},
                                                            "insulation_correl" : {"wtd" : "red",
                                                                                   "rdm" : "#FF6633"}}}}