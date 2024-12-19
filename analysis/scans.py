# for model ensemble scans, kappa scans, state sequences

import keypointmoseq as kpms
import matplotlib.pyplot as plt
import numpy as np
import os
from analysis.results import *
import h5py

def kappa_scan(kappas, save_dir, prefix,):
    kpms.plot_kappa_scan(kappas, save_dir, prefix)
    plt.savefig(save_dir+"/kappas.pdf")
    
def plot_state_seq_ensemble(save_dir, prefix, model_ids, index):
    model_names = ['MODEL-{}'.format(i) for i in model_ids]
    all_zs = []
    for model_name in model_names:
        model, data, metadata, current_iter = kpms.load_checkpoint(save_dir, model_name)
        z = np.array(model["states"]["z"])
        all_zs.append(z[index])
    
    all_zs = np.vstack(all_zs)
    plt.figure(figsize=(5,3))
    plt.imshow(all_zs, interpolation='nearest',aspect='auto')
    plt.ylabel("models")
    plt.xlabel("frames")
    plt.yticks([])
    plt.xticks([])
    plt.savefig("{}/figures/state_seq_ensemble.pdf".format(save_dir))
        
    
def model_ensemble(save_dir, prefix, model_ids):
    model_names = ['MODEL-{}'.format(i) for i in model_ids]
    print(model_names)
    
    # calculate eml scores
    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(save_dir, model_names)
    best_model = model_names[np.argmax(eml_scores)]
    print(f"Best model: {best_model}")
    
    # plot eml scores
    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)
    if not os.path.isdir(save_dir+"/figures/"):
        os.mkdir(save_dir+"/figures/")
    plt.savefig(save_dir+"/figures/model_emlscores.pdf")
    
    # create confusion matrix
    name_1, name_2 = '0','1'
    model_name_1 = 'MODEL-'+name_1
    model_name_2 = 'MODEL-'+name_2

    results_1 = kpms.load_results(save_dir, model_name_1)
    results_2 = kpms.load_results(save_dir, model_name_2)

    fig, ax = kpms.plot_confusion_matrix(results_1, results_2, min_frequency=0.0)
    ax.set_xlabel("Model {}".format(name_2))
    ax.set_ylabel("Model {}".format(name_1))
    plt.savefig(save_dir+"/figures/model_confusion_{}-{}.pdf".format(name_1,name_2))
    
def plot_states(save_dir, 
                model_name, 
                name, index, 
                start, 
                window_size, 
                ax):
    model_dir = os.path.join(save_dir, model_name)
    
    if not os.path.isdir(model_dir+"/figures/"):
        os.mkdir(model_dir+"/figures/")
    
    model, data, metadata, current_iter = kpms.load_checkpoint(save_dir, model_name)
    mask = np.array(data["mask"])
    window_size = int(min(window_size, mask[index].sum() - 1))
    print("Shape of mask: ", mask.shape, " same as shape of 'z'")
        
    sample_state_history = []
    with h5py.File(f"{model_dir}/checkpoint.h5", "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])
        
        for i in saved_iterations:
            z = f[f"model_snapshots/{i}/states/z"][()]
            sample_state_history.append(z[index, start : start + window_size])
    print("The mask for this segment/subject is {} frames".format(mask[index].sum()))
    ax.imshow(sample_state_history, cmap = plt.cm.jet, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Iterations")
    ax.set_title("State sequence history for {}, {}".format(name, model_name))
    yticks = [int(y) for y in ax.get_yticks() if y <= len(saved_iterations) and y > 0]
    xticks = [int(x) for x in ax.get_xticks() if x <= start + window_size and x >= 0]
    yticklabels = saved_iterations[yticks]
    xticklabels = np.array(xticks)+np.ones(len(xticks))*start
    xticklabels = xticklabels.astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

def get_num_frames(body_config):
    data_dict = quickly_get_data(**body_config())
    num_frames = {}
    for name in data_dict["rot"].keys():
        num_frames[name] = data_dict["rot"][name].shape[0]
    return num_frames
    
def get_sequences(save_dir,
                  prefix,
                  subject,
                  segment,
                  body_config,
                  augment,
                  num_windows,
                  **kwargs
                  ):
    # get model names
    model_names=[]
    for subdir, dirs, files in os.walk(save_dir):
        if len(subdir.split("/")[-1])>0 and prefix in subdir.split("/")[-1]:
            model_names.append(subdir.split("/")[-1])
    model_names.sort()
    
    # get number of frames per subject window
    frame_nums = get_num_frames(body_config)
    
    # find index of data according 
    ids_order = [1,10,11,12,2,3,4,5,6,7,8,9]
    index_of_subject = ids_order.index(subject)

    # get number of frames for this subject and specific segment
    if augment:
        index = num_windows*index_of_subject + segment
    else:
        index = subject-1
    key = list(frame_nums.keys())[index]
    num_frames = frame_nums[key]

    # plot sequences for all models in one graph
    fig, axs = plt.subplots(len(model_names), 1, figsize=(10,20))  # Adjust figsize as needed
    for i, model_name in enumerate(model_names):
        plot_states(save_dir, model_name, subject, index, 0, num_frames, axs[i])
    fig.tight_layout()
  
    print("saving in ", save_dir + "/figures/state_sequence_{}.pdf".format(key))
    fig.savefig(save_dir + "/figures/state_sequence_{}.pdf".format(key))