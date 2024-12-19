import yaml
from preprocessing.data import *
import keypointmoseq as kpms
import jax.numpy as jnp
from preprocessing.dappy import read, preprocess, vis
from keypointmoseq.jaxmoseq.models.keypoint_slds.alignment import align_egocentric

def load_config(path,get_idxs=False):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    
    if get_idxs:
        config["anterior_idxs"] = jnp.array(
            [config["use_bodyparts"].index(bp) for bp in config["anterior_bodyparts"]]
        )
        config["posterior_idxs"] = jnp.array(
            [config["use_bodyparts"].index(bp) for bp in config["posterior_bodyparts"]]
        )
    return config
    
def update_config(path, **kwargs):
    config = load_config(path)
    config.update(kwargs)
    return config

def update_body_config(body):
    if body == "l":
        return update_config("run_config.yml",
                      body=body,
                      body_name="lefthand",
                      num_keypoints=21,
                      ar_only_kappa=1000.0,
                      full_model_kappa=100.0,
                      num_windows=12,
                      forward_indices=[0,2,17],
                      target_direction=[1,0,0],
                      config_path="training/hand_config.yml",
                      thresh=2,
                      combine=False)
    elif body == "lr":
        return update_config("run_config.yml",
                      body=body,
                      body_name="leftright",
                      num_keypoints=21,
                      ar_only_kappa=1000.0,
                      full_model_kappa=100.0,
                      num_windows=12,
                      forward_indices=[0,2,17],
                      target_direction=[1,0,0],
                      config_path="training/hand_config.yml",
                      thresh=2,
                      combine=True)
    elif body == "r":
        return update_config("run_config.yml",
                      body=body,
                      body_name="righthand",
                      num_keypoints=21,
                      ar_only_kappa=1000.0,
                      full_model_kappa=100.0,
                      num_windows=12,
                      forward_indices=[0,2,17],
                      target_direction=[1,0,0],
                      config_path="training/hand_config.yml",
                      thresh=2,
                      combine=False)
    else:
        return update_config("run_config.yml",
                      body=body,
                      num_keypoints=26,
                      ar_only_kappa=10000.0,
                      full_model_kappa=1000.0,
                      num_windows=8,
                      forward_indices=[3,4,19],
                      target_direction=[1,0,0],
                      config_path="training/gait_config.yml",
                      thresh=1,
                      combine=False)

def get_config_path(body):
    if body == "g":
        skeleton_config_path = "preprocessing/configs/gait_skeleton.yaml"
    elif body == "r":
        skeleton_config_path = "preprocessing/configs/righthand_skeleton.yaml"
    else:
        skeleton_config_path = "preprocessing/configs/lefthand_skeleton.yaml"
    return skeleton_config_path

def preprocess_videos(body, 
                      num_windows, 
                      forward_indices, 
                      target_direction,
                      body_name,
                      window=100,
                      id=1,
                      segment=0,
                      combine=False,
                      **kwargs):
    connectivity = read.connectivity_config(get_config_path(body))
    print(target_direction)
    if not combine:
        data = get_augmented_data([id], num_windows, body, forward_indices, target_direction)
        name = "sub{}.{}".format(id,segment)
    else:
        data = get_left_right_data([id], num_windows, body, forward_indices, target_direction)
        name = "{}sub{}.{}".format("r",id,segment)
    num_frames = data["raw"][name].shape[0]
    
    Y_aligned, v, h = align_egocentric(jax.device_put(data["raw"][name]), [17], [0,2])

    # vis.pose.grid3D(
    #             np.concatenate((data["raw"][name], data["stand"][name]), axis=0), 
    #             connectivity,
    #             frames=[0, num_frames],
    #             centered=False,
    #             labels=["Raw", "My Processing (palm)"],
    #             title=None,
    #             fps=80,
    #             N_FRAMES=window,
    #             VID_NAME="{}_meanheading.mp4".format(name),
    #             SAVE_ROOT="preprocessing/outputs/check_heading/{}/".format(body_name),
    #         )
    
    vis.pose.grid3D(
                np.concatenate((data["raw"][name], data["stand"][name], Y_aligned), axis=0), 
                connectivity,
                frames=[0, num_frames, num_frames*2],
                centered=False,
                labels=["Raw", "My Processing (palm)", "Keypoint Moseq (palm)"],
                title=None,
                fps=80,
                N_FRAMES=window,
                VID_NAME="{}_palmforward.mp4".format(name),
                SAVE_ROOT="preprocessing/outputs/check_heading/{}/".format(body_name),
            )

def get_pca(kpms_config, 
            ids, 
            num_windows, 
            body, 
            forward_indices, 
            target_direction, 
            augment,
            combine,
            config_path,
            **kwargs):
    print("combine: ", combine)
    if combine:
        print("getting left right data")
        data_dict = get_left_right_data(ids, num_windows, body, forward_indices, target_direction)
    elif augment:
        print("getting augmented data")
        data_dict = get_augmented_data(ids, num_windows, body, forward_indices, target_direction)
    else:
        print("getting full subject data")
        data_dict = get_subject_data(ids, num_windows, body, forward_indices, target_direction)
        
    data, metadata = kpms.format_data(data_dict["stand"], data_dict["conf"], **kpms_config())
    
    # fit pca
    print("completing pca...")
    pca = kpms.fit_pca(**data, **kpms_config())
    kpms.save_pca(pca, "training")
    cs = np.cumsum(pca.explained_variance_ratio_)
    latentdim = 0
    if cs[-1] < 0.9:
        latentdim = len(cs)
        print(
            f"All components together only explain {cs[-1]*100}% of variance."
        )
    else:
        latentdim = (cs>0.9).nonzero()[0].min()+1
        print(
            f">={0.9*100}% of variance exlained by {latentdim} components."
        )
    if latentdim == 1:
        latentdim = 2
    kpms.update_config(config_path, latent_dim=int(latentdim))

    return data, metadata, pca

def run_model(prefix, 
              seed, 
              kpms_config,
              body_config,
              save_dir,
              ar_only_kappa,
              full_model_kappa,
              num_ar_iters,
              num_full_iters,
              body,
              **kwargs):
    data, metadata, pca = get_pca(kpms_config, **body_config())
    
    print("Fitting model with seed: ", seed)
    model_name = f'{prefix}-{seed}'
    
    if body == "g":
        fix_heading = True
        neglect_location = True
    else:
        fix_heading = False
        neglect_location = False
    
    print(kpms_config())
        
    model = kpms.init_model(
        data, 
        pca=pca, 
        **kpms_config(), 
        seed=jax.random.PRNGKey(seed),
    )

    # stage 1: fit the model with AR only
    model = kpms.update_hypparams(model, kappa=ar_only_kappa)
    model = kpms.fit_model(
        model,
        data,
        metadata,
        save_dir,
        model_name,
        verbose=False,
        ar_only=True,
        fix_heading=fix_heading, 
        neglect_location=neglect_location,
        num_iters=num_ar_iters
    )[0]
    
    # stage 2: fit the full model
    model = kpms.update_hypparams(model, kappa=full_model_kappa)
    kpms.fit_model(
        model,
        data,
        metadata,
        save_dir,
        model_name,
        verbose=False,
        ar_only=False,
        fix_heading=fix_heading, 
        neglect_location=neglect_location,
        start_iter=num_ar_iters,
        num_iters=num_full_iters
    )

    kpms.reindex_syllables_in_checkpoint(save_dir, model_name)
    model, data, metadata, current_iter = kpms.load_checkpoint(save_dir, model_name)
    
    anterior_idxs, posterior_idxs = kpms_config()['anterior_idxs'], kpms_config()['posterior_idxs']
    
    results = kpms.extract_results(model, metadata, save_dir, model_name, data, data, anterior_idxs, posterior_idxs)
    kpms.save_results_as_csv(results, save_dir, model_name)
    
    return model, model_name
    
def apply_model( 
              save_dir,
              model_name,
              kpms_config,
              body_config,
              body,
              ids,
              num_windows,
              num_full_iters,
              forward_indices,
              target_direction,
              combine,
              augment,
              **kwargs):
    
    train_data, metadata, pca = get_pca(kpms_config, **body_config())
    left_over = [i for i in range(1,16) if i not in ids]
    print("leftover patients: ", left_over)
    
    if combine:
        data_dict = get_left_right_data(left_over, num_windows, body, forward_indices, target_direction)
    elif augment:
        data_dict = get_augmented_data(left_over, num_windows, body, forward_indices, target_direction)
    else:
        data_dict = get_subject_data(left_over, num_windows, body, forward_indices, target_direction)
        
    data, metadata = kpms.format_data(data_dict["stand"], data_dict["conf"], **kpms_config())
    
    model = kpms.load_checkpoint(save_dir, model_name)[0]
    results = kpms.apply_model(model, data, metadata, save_dir, model_name, num_iters=num_full_iters, **kpms_config())
    kpms.save_results_as_csv(results, save_dir, model_name)
    
    
    
    