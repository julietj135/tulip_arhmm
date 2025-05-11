

import os
from run import *
import sys
import analysis as an
import numpy as np
import keypointmoseq as kpms

###################### SET UP ############################################

# body = "g"
# body_name = "gait"
# body_config_path = "training/gait_config.yml"
# body = "r"
# body_name = "righthand"
body = "l"
body_name = "lefthand"
# body = "lr"
# body_name = "leftright"
body_config_path = "training/hand_config.yml"

kpms_config = lambda: load_config(body_config_path,True)
body_config = lambda: update_body_config(body)

print(body_config())
# print(kpms_config())

###################### WORK ############################################

# # FOR PREPROCESSING VIDEOS
# for sub in [1,2,5,10]:
#     preprocess_videos(window=100,id=sub,segment=0,**body_config())

# FOR KAPPA SCAN
# an.kappa_scan([1000,10000,100000,1000000], "training/outputs/1.kappa_scans/{}".format(body_name), "MODEL")

# FOR RUNNING MODEL ENSEMBLES ON SUBSET
prefix, seed, save_dir = "MODEL", int(sys.argv[1]),  "training/outputs/2.model_ensembles/{}".format(body_name)

# model, model_name = run_model(prefix, 
#           seed, 
#           kpms_config,
#           body_config,
#           save_dir,
#           **body_config())
model_name = "MODEL-"+str(seed)
# apply_model(save_dir,
#               model_name,
#               kpms_config,
#               body_config,
#               **body_config())

# FOR CLASSIFICATION
# an.model_ensemble(save_dir, prefix, [0,1,2,3,4,5])

# an.get_sequences(save_dir,
#                   prefix,
#                   1,
#                   0,
#                   body_config,
#                   **body_config()
#                   )

# an.run_classifier(save_dir, 
#                 model_name, # if want specific model, just place full name here instead of prefix
#                 **body_config())


# FOR GRID MOVIES AND SYLLABLE STUFF
an.get_movies(save_dir,
                    model_name,
                    body_config,
                    kpms_config,
                    False, # grid
                    False, # full
                    True, # traj
                    ["sub1.0","sub5.0"],
                    **body_config())
                    
# an.get_transition_matrix(save_dir, model_name, 0.00)

# an.get_syllable_quantitative_main(save_dir, model_name)

# an.get_syllable_quantitative_all(save_dir,
#                               model_name,
#                               )

# an.plot_syllable_heel(save_dir, 
#                    model_name, 
#                    body_config,
#                    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#                    body)

# an.plot_syllable_fingers(save_dir, 
#                    model_name, 
#                    body_config,
#                    [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
#                    body)


# FOR FIGURE MAKING
# data_dict = an.quickly_get_data(**body_config())

# # median durations for gait kappa: 11, 17, 45, 46
# save_dir = "training/outputs/1.kappa_scans/gait"
# model_name = "MODEL-100000"
# model, data, metadata, current_iter = kpms.load_checkpoint(save_dir, model_name)
# an.plot_progress(model,
#     data,
#     "{}/{}/checkpoint.h5".format(save_dir,model_name),
#     project_dir=save_dir,
#     model_name=model_name,
#     savefig=True)

# an.plot_state_seq_ensemble(save_dir, prefix, [0,1,2,3,4,5], 2)
# an.plot_state_seq_ensemble(save_dir, prefix, [0,1,2,3,4,5], 10)
# an.plot_state_seq_subjects(save_dir,model_name)

# for syllables
# coordinates = an.quickly_get_data(**body_config())
# anterior_idxs, posterior_idxs = kpms_config()['anterior_idxs'], kpms_config()['posterior_idxs']
# model, data, metadata, current_iter = kpms.load_checkpoint(save_dir, model_name)
# results = kpms.extract_results(model, metadata, save_dir, model_name, data, data, anterior_idxs, posterior_idxs)
# print(coordinates["stand"].keys())
# an.plot_syllable(coordinates["stand"], results, save_dir, model_name, projection_planes=['yz'], post=10, **kpms_config())

# an.plot_numsub_in_syllable(save_dir, model_name)
