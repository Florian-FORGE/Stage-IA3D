EXTREMUM_HEATMAP = {"Matrix" : (-0.2, 3), "RealMatrix" : (-0.95, 1.8), "OrcaMatrix" : (-0.75, 1.5)}

COLOR_CHART = {"insulation_count" : "blue", "PC1" : "green", "insulation_correl" : "red"}

WHICH_MATRIX = {"Matrix" : {"count" : "_obs_o_exp", "correl" : "_obs"}, 
                "OrcaMatrix" : {"count" : "obs_o_exp", "correl" : "obs"}, 
                "RealMatrix" : {"count" : "log_obs_o_exp", "correl" : "obs"}}

SMOOTH_MATRIX  = {"_get_insulation_score" : {"bool" : True, "val" : {"RealMatrix" : 5, "OrcaMatrix" : 2}}, 
                  "correl_mat" : {"bool" : True, "val" : {"RealMatrix" : 2, "OrcaMatrix" : 1}}}