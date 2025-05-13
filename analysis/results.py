import cv2
import keypointmoseq as kpms
from keypointmoseq.jaxmoseq.utils import get_durations, get_frequencies
from preprocessing.data import *
from keypointmoseq.io import load_results, _get_path
from keypointmoseq.util import *
import tqdm as tqdm
import imageio
from run import *
import yaml
import h5py
from matplotlib.colors import LinearSegmentedColormap

def quickly_get_data(ids,
                     body,
                     num_windows,
                     forward_indices,
                     target_direction,
                     combine,
                     augment,
                     **kwargs):
    """
    Retrieves data based on specified preprocessing type: left-right combined data, augmented data, or full subject data.

    Parameters
    ----------
    ids : list
        List of subject identifiers.
    body : str
        Specifies the body part or configuration (e.g., "lr").
    num_windows : int
        Number of temporal windows per subject.
    forward_indices : list
        Indices used for forward alignment.
    target_direction : str
        Direction to align or process movement (e.g., "forward").
    combine : bool
        If True, returns left-right combined data.
    augment : bool
        If True, returns augmented data.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    dict
        Dictionary containing processed data keyed by subject.
    """
    if combine:
        print("getting left right data")
        data_dict = get_left_right_data(ids, num_windows, body, forward_indices, target_direction)
    elif augment:
        print("getting augmented data")
        data_dict = get_augmented_data(ids, num_windows, body, forward_indices, target_direction)
    else:
        print("getting full subject data")
        data_dict = get_subject_data(ids, num_windows, body, forward_indices, target_direction)
    
    return data_dict

def make_index_csv(save_dir,
                   og_index_df,
                   data,
                   augment,
                   ids,
                   body,
                   num_windows,
                   **kwargs):
    """
    Creates a new index CSV file mapping each sample to its group label (HT or PD) after augmentation.

    Parameters
    ----------
    save_dir : str
        Directory to save the index file.
    og_index_df : pd.DataFrame
        Original index DataFrame containing subject names and group labels.
    data : dict
        Dictionary of processed data.
    augment : bool
        If True, assumes augmentation and expands labels accordingly.
    ids : list
        List of subject identifiers.
    body : str
        Type of body data ("lr" for left-right).
    num_windows : int
        Number of windows per subject.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        New index DataFrame with sample names, group, and stage labels.
    """
    new_labels = []
    new_stages = []
    new_stages_cat2 = []
    new_stages_cat3 = []
    
    if augment and body != "lr":
        for id in ids:
            for l in range(num_windows): 
                newlabel = og_index_df.group[og_index_df.name == "sub{}".format(id)].tolist()[0]
                newstage = og_index_df.stage[og_index_df.name == "sub{}".format(id)].tolist()[0]
                stage2 = og_index_df.stage_2[og_index_df.name == "sub{}".format(id)].tolist()[0]
                stage3 = og_index_df.stage_3[og_index_df.name == "sub{}".format(id)].tolist()[0]
                new_labels.append(newlabel)
                new_stages.append(newstage)
                new_stages_cat2.append(stage2)
                new_stages_cat3.append(stage3)
                
    if body == "lr":
        print("index file should have already been made with 12 windows per sub")
        return og_index_df
                
    new_labels_df = {"name":data.keys(),
                                 "group":new_labels, "stage":new_stages, "stage_2":new_stages_cat2, "stage_3":new_stages_cat3}
    df = pd.DataFrame(new_labels_df)
    df.to_csv(save_dir+"/index.csv",index=False)
    print("Saved new index csv at "+save_dir+"/index.csv")
    return df

def plot_numsub_in_syllable(save_dir, model_name):
    """
    Plots the number of subjects (total, healthy, and diseased) that used each syllable.

    Parameters
    ----------
    save_dir : str
        Directory containing results and where plots will be saved.
    model_name : str
        Name of the trained model.

    Saves
    -----
    counts.png, counts_combined.png : Bar plots of syllable usage statistics.
    """
    results = kpms.load_results(save_dir, model_name)
    z = []
    lengths = []

    for subject in results.keys():
        seq = results[subject]["syllable"]
        z.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)
    z_array = np.full((len(z), max_len), -1)
    for i, seq in enumerate(z):
        z_array[i, :len(seq)] = seq
    print(len(z_array))
    print(z_array)
    index_df = pd.read_csv(os.path.join(save_dir,'index.csv'))

    counts, pd_counts, ht_counts = {}, {}, {}
    for i in range(np.max(z_array)+1):
        counts[i], pd_counts[i], ht_counts[i] = 0,0,0
    for i in range(len(z_array)):
        sylls = set(z_array[i])
        diagnosis = index_df.group[i]
        for syll in sylls:
            if syll > -1:
                counts[syll] += 1
                if diagnosis == "HT":
                    ht_counts[syll] += 1
                else:
                    pd_counts[syll] += 1
    
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.bar(np.arange(np.max(z_array)+1), counts.values(), color='black')
    plt.xlabel("syllable")
    plt.ylabel("total number of subjects")
    plt.subplot(1,3,2)
    plt.bar(np.arange(np.max(z_array)+1), ht_counts.values(), color='green')
    plt.xlabel("syllable")
    plt.ylabel("total number of healthy subjects")
    plt.subplot(1,3,3)
    plt.bar(np.arange(np.max(z_array)+1), pd_counts.values(), color='salmon')
    plt.xlabel("syllable")
    plt.ylabel("total number of diseased subjects")
    plt.tight_layout()
    plt.savefig(save_dir+"/"+model_name+"/figures/counts.png")

    # Another plot
    plt.figure(figsize=(5,5))
    print("HT counts: ", list(ht_counts.values()))
    print("PD counts: ", list(pd_counts.values()))
    plt.bar(np.arange(np.max(z_array)+1), list(ht_counts.values()), label='UPDRS < 2', color='green')
    plt.bar(np.arange(np.max(z_array)+1), list(pd_counts.values()), bottom=list(ht_counts.values()), label='UPDRS >= 2', color='salmon')
    plt.ylabel('Number of Augmented Subjects')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+"/"+model_name+"/figures/counts_combined.png")

def plot_numsub_in_syllable_3stages(save_dir, model_name):
    """
    Plots the number of subjects (total, healthy, and diseased) that used each syllable.

    Parameters
    ----------
    save_dir : str
        Directory containing results and where plots will be saved.
    model_name : str
        Name of the trained model.

    Saves
    -----
    counts.png, counts_combined.png : Bar plots of syllable usage statistics.
    """
    results = kpms.load_results(save_dir, model_name)
    z = []
    lengths = []

    for subject in results.keys():
        seq = results[subject]["syllable"]
        z.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)
    z_array = np.full((len(z), max_len), -1)
    for i, seq in enumerate(z):
        z_array[i, :len(seq)] = seq
    index_df = pd.read_csv(os.path.join(save_dir,'index.csv'))

    counts, counts1, counts2, counts3 = {}, {}, {}, {}
    for i in range(np.max(z_array)+1):
        counts[i], counts1[i], counts2[i], counts3[i] = 0,0,0,0
    for i in range(len(z_array)):
        sylls = set(z_array[i])
        months = index_df.stage[i]
        for syll in sylls:
            if syll > -1:
                counts[syll] += 1
                if months < 24:
                    counts1[syll] += 1
                elif (months >= 24) and (months < 60):
                    counts2[syll] += 1
                else:
                    counts3[syll] += 1

    # Another plot
    plt.figure(figsize=(5,5))
    plt.bar(np.arange(np.max(z_array)+1), list(counts1.values()), label='0-2 yrs', color='green')
    plt.bar(np.arange(np.max(z_array)+1), list(counts2.values()), label='2-5 yrs', color='orange')
    plt.bar(np.arange(np.max(z_array)+1), list(counts3.values()), label='5+ yrs', color='salmon')
    plt.ylabel('Number of Augmented Subjects')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+"/"+model_name+"/figures/counts_combined_3stages.png")
    
def plot_numsub_in_syllable_2stages(save_dir, model_name):
    """
    Plots the number of subjects (total, healthy, and diseased) that used each syllable.

    Parameters
    ----------
    save_dir : str
        Directory containing results and where plots will be saved.
    model_name : str
        Name of the trained model.

    Saves
    -----
    counts.png, counts_combined.png : Bar plots of syllable usage statistics.
    """
    results = kpms.load_results(save_dir, model_name)
    z = []
    lengths = []

    for subject in results.keys():
        seq = results[subject]["syllable"]
        z.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)
    z_array = np.full((len(z), max_len), -1)
    for i, seq in enumerate(z):
        z_array[i, :len(seq)] = seq
    index_df = pd.read_csv(os.path.join(save_dir,'index.csv'))

    counts, counts1, counts2 = {}, {}, {}
    for i in range(np.max(z_array)+1):
        counts[i], counts1[i], counts2[i] = 0,0,0
    for i in range(len(z_array)):
        sylls = set(z_array[i])
        months = index_df.stage[i]
        for syll in sylls:
            if syll > -1:
                counts[syll] += 1
                if months < 24:
                    counts1[syll] += 1
                else:
                    counts2[syll] += 1

    # Another plot
    plt.figure(figsize=(5,5))
    plt.bar(np.arange(np.max(z_array)+1), list(counts1.values()), label='<2 yrs', color='green')
    plt.bar(np.arange(np.max(z_array)+1), list(counts2.values()), label='>=2 yrs', color='salmon')
    plt.ylabel('Number of Augmented Subjects')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir+"/"+model_name+"/figures/counts_combined_2stages.png")

def plot_syllable(
    coordinates,
    results,
    save_dir,
    model_name,
    output_dir=None,
    pre=0,
    post=15,
    min_frequency=0.000,
    min_duration=3,
    skeleton=[],
    bodyparts=None,
    use_bodyparts=None,
    plot_options={},
    get_limits_pctl=0,
    padding={"left": 0.1, "right": 0.1, "top": 0.2, "bottom": 0.2},
    lims=None,
    projection_planes=["xy", "xz"],
    density_sample=False,
    sampling_options={"n_neighbors": 1},
    **kwargs
):
    """
    Plots typical trajectories for each syllable in 2D projections.

    Parameters
    ----------
    coordinates : np.ndarray
        Pose or keypoint data.
    results : dict
        Model output containing syllables.
    save_dir : str
        Root directory for saving plots.
    model_name : str
        Name of the model used for labeling.
    output_dir : str, optional
        Path to save trajectory plots (defaults to model-based path).
    pre : int
        Frames before syllable onset to include.
    post : int
        Frames after syllable onset to include.
    min_frequency : float
        Minimum frequency threshold for a syllable to be included.
    min_duration : int
        Minimum duration for a syllable to be considered.
    skeleton : list
        Skeleton structure for visualization.
    bodyparts : list
        All body parts available in the data.
    use_bodyparts : list
        Subset of bodyparts to include.
    plot_options : dict
        Options for the plot appearance.
    get_limits_pctl : float
        Percentile for calculating axis limits.
    padding : dict
        Padding for plot limits.
    lims : np.ndarray, optional
        Global limits for trajectory plots.
    projection_planes : list of str
        Planes to project ('xy', 'xz', 'yz').
    density_sample : bool
        Whether to sample based on density.
    sampling_options : dict
        Options for sampling (e.g., neighbors).
    **kwargs : dict
        Additional arguments.

    Saves
    -----
    One PDF plot per syllable per projection.
    """
    
    edges = [] if len(skeleton) == 0 else kpms.get_edges(use_bodyparts, skeleton)
    output_dir = "{}/{}/trajectory_plots/".format(save_dir,model_name)

    typical_trajectories = kpms.get_typical_trajectories(
        coordinates,
        results,
        pre,
        post,
        min_frequency,
        min_duration,
        bodyparts,
        use_bodyparts,
        density_sample,
        sampling_options,
    )

    syllable_ixs = sorted(typical_trajectories.keys())
    print(syllable_ixs)
    titles = [f"syllable{s}" for s in syllable_ixs]
    Xs = np.stack([typical_trajectories[s] for s in syllable_ixs])

    projection_planes = [
        "".join(sorted(plane.lower())) for plane in projection_planes
    ]
    assert set(projection_planes) <= set(["xy", "yz", "xz"]), fill(
        "`projection_planes` must be a subset of `['xy','yz','xz']`"
    )
    if lims is not None:
        assert lims.shape == (2, 3), fill(
            "`lims` must be None or an ndarray of shape (2,3) when plotting 3D data"
        )
    all_Xs, all_lims, suffixes = [], [], []
    for plane in projection_planes:
        use_dims = {"xy": [0, 1], "yz": [1, 2], "xz": [0, 2]}[plane]
        all_Xs.append(Xs[..., use_dims])
        suffixes.append("." + plane)
        if lims is None:
            all_lims.append(kpms.get_limits(all_Xs[-1], pctl=get_limits_pctl, **padding))
        else:
            all_lims.append(lims[..., use_dims])
    
    for Xs_2D, lims, suffix in zip(all_Xs, all_lims, suffixes):
        # individual plots
        desc = "Generating trajectory plots"
        for title, X in tqdm.tqdm(
            zip(titles, Xs_2D), desc=desc, total=len(titles), ncols=72
        ):
            fig, ax, rasters = kpms.plot_trajectories(
                [],
                X[None],
                lims,
                edges=edges,
                return_rasters=False,
                plot_width=2,
                **plot_options,
            )

            plt.savefig(os.path.join(output_dir, f"{title}{suffix}.pdf"))
    

def plot_state_seq_subjects(save_dir,model_name):
    """
    Visualizes the state (syllable) sequences of the first two windows per subject as a heatmap.

    Parameters
    ----------
    save_dir : str
        Directory containing model results.
    model_name : str
        Name of the model.

    Saves
    -----
    state_seq_subjects.pdf : Heatmap of state sequences.
    """
    results = kpms.load_results(save_dir, model_name)
    z = []
    lengths = []
    counter = 0
    if "gait" in save_dir:
        window = 8
        colors = ['salmon','lightseagreen','orangered','darkturquoise','deepskyblue','rosybrown','orange']
    else:
        window = 12
        if "righthand" in save_dir:
            colors = ['rosybrown','indianred','darkturquoise','deepskyblue','salmon','orangered','slateblue','lightseagreen']        
        else:
            colors = ['seagreen', 'deepskyblue', 'rosybrown', 'salmon', 'darkturquoise', 'orangered', 'indianred','slateblue']
    for subject in results.keys():
        seq = results[subject]["syllable"]
        if counter % window in [0,1]:
            z.append(seq)
            lengths.append(len(seq))
        counter += 1
    
    for seq in range(len(z)):
        if len(z[seq]) > min(lengths):
            z[seq] = z[seq][:min(lengths)]
        
    z = np.array(z)
    
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))

    plt.figure(figsize=(5,3))
    plt.imshow(z, interpolation='nearest',aspect='auto',cmap=cmap)
    plt.ylabel("subject (first two segments)")
    plt.xlabel("frames")
    plt.yticks([])
    plt.xticks([])
    plt.savefig("{}/{}/figures/state_seq_subjects.pdf".format(save_dir,model_name))
    

def plot_progress(
    model,
    data,
    checkpoint_path,
    project_dir=None,
    model_name=None,
    path=None,
    savefig=True,
    fig_size=None,
    window_size=600,
    min_frequency=0.001,
    min_histogram_length=10,
):
    """
    Plots progress metrics during training: frequency distribution, duration histogram, and median duration over iterations.

    Parameters
    ----------
    model : dict
        Final model containing 'states' with syllable sequences.
    data : dict
        Data dictionary with 'mask' used for training.
    checkpoint_path : str
        Path to the HDF5 file containing model snapshots.
    project_dir : str, optional
        Directory to save plots.
    model_name : str, optional
        Name of the model (used in file naming).
    path : str, optional
        Specific path to save the figure.
    savefig : bool
        If True, saves the figure.
    fig_size : tuple, optional
        Size of the final figure.
    window_size : int
        Number of frames to sample for sequence evolution.
    min_frequency : float
        Minimum frequency for syllables to be included in histogram.
    min_histogram_length : int
        Minimum length for frequency histogram x-axis.

    Returns
    -------
    tuple
        (fig, axs) for the generated figure and its axes.
    """
    z = np.array(model["states"]["z"])
    mask = np.array(data["mask"])
    durations = get_durations(z, mask)
    frequencies = get_frequencies(z, mask)

    with h5py.File(checkpoint_path, "r") as f:
        saved_iterations = np.sort([int(i) for i in f["model_snapshots"]])

    fig, axs = plt.subplots(3,1)
    fig_size = (2,5)
        
    frequencies = np.sort(frequencies[frequencies > min_frequency])[::-1]
    xmax = max(len(frequencies), min_histogram_length)
    axs[0].bar(range(len(frequencies)), frequencies, width=1)
    axs[0].set_ylabel("frequency\ndistribution")
    # axs[0].set_xlabel("syllable")
    axs[0].set_xlim([-1, xmax + 1])
    axs[0].set_yticks([])

    lim = int(np.percentile(durations, 95))
    lim = 80 # so x axis ticks are same for all kappas
    binsize = max(int(np.floor(lim / 30)), 1)
    axs[1].hist(durations, range=(1, lim), bins=(int(lim / binsize)), density=True)
    axs[1].set_xlim([1, lim])
    # axs[1].set_xlabel("syllable duration (frames)")
    axs[1].set_ylabel("duration\ndistribution")
    axs[1].set_yticks([])

    if len(saved_iterations) > 1:
        window_size = int(min(window_size, mask.max(0).sum() - 1))
        nz = np.stack(np.array(mask[:, window_size:]).nonzero(), axis=1)
        batch_ix, start = nz[np.random.randint(nz.shape[0])]

        sample_state_history = []
        median_durations = []

        for i in saved_iterations:
            with h5py.File(checkpoint_path, "r") as f:
                z = np.array(f[f"model_snapshots/{i}/states/z"])
                sample_state_history.append(z[batch_ix, start : start + window_size])
                median_durations.append(np.median(get_durations(z, mask)))

        axs[2].scatter(saved_iterations, median_durations)
        print("median duration: ", median_durations[-1])
        axs[2].set_ylim([-1, 50])
        # axs[2].set_xlabel("iteration")
        axs[2].set_ylabel("median\nduration")
        axs[2].set_yticks([])

    fig.set_size_inches(fig_size)
    plt.tight_layout()

    if not os.path.isdir(project_dir+"/"+model_name+"/figures/"):
        os.mkdir(project_dir+"/"+model_name+"/figures/")
    path = project_dir+"/"+model_name+"/figures/fitting_dist_{}.pdf".format(model_name)
    plt.savefig(path)
    return fig, axs

def get_zvals(save_dir, 
                   model_name, 
                   body_config,
                   body_ind,
                   sub_id,
                   name):
    """
    Retrieves the z-values for a specific body part and subject from the processed data.

    Parameters
    ----------
    save_dir : str
        Directory where results are saved.
    model_name : str
        Name of the model whose results are to be accessed.
    body_config : function
        A function to configure body data.
    body_ind : int
        Index of the body part for which z-values are required.
    sub_id : int
        Identifier for the subject.
    name : str
        The specific name or identifier of the data to retrieve.

    Returns
    -------
    np.ndarray
        The z-values for the specified body part, subject, and data name.
    """

    config = body_config()
    config.update(ids = [sub_id])
    body_config = lambda: config
    data_dict = quickly_get_data(**body_config())
        
    return data_dict["raw"][name][:,body_ind,-1]


def plot_syllable_fingers(save_dir, 
                   model_name, 
                   body_config,
                   plot_ids,
                   body):
    """
    Plots finger-related data (index and thumb movements) with syllable states.

    Parameters
    ----------
    save_dir : str
        Directory where model data is saved.
    model_name : str
        Name of the model whose results are being analyzed.
    body_config : function
        A function to configure body data.
    plot_ids : list of int
        List of subject IDs to plot.
    body : str
        Indicates whether the plot is for the left ('l') or right ('r') hand.

    Returns
    -------
    None
        Generates and saves plots showing finger movement (index and thumb).
    """

    if body not in "lr":
        raise ValueError("body is not fingertapping, should not use plot_syllable_fingers")
    
    pointer = 8
    thumb = 4
    segment = 0
    
    model_dir = os.path.join(save_dir, model_name) 
    if not os.path.isdir(model_dir+"/figures/finger_plots"):
        os.mkdir(model_dir+"/figures/finger_plots")
    
    if body == "l":
        all_colors = ['seagreen', 'deepskyblue', 'rosybrown', 'salmon', 'darkturquoise', 'orangered', 'indianred','slateblue','lightseagreen']
    if body == "r":
        all_colors = ['rosybrown','indianred','darkturquoise','deepskyblue','salmon','orangered','slateblue','lightseagreen']
        
    for i in plot_ids:
        # get z value of right and left heel
        name = "sub{}.{}".format(i,segment)
        point_z = get_zvals(save_dir, model_name, body_config, pointer, i, name)
        point_z = point_z-point_z[0]
        thumb_z = get_zvals(save_dir, model_name, body_config, thumb, i, name)
        thumb_z = thumb_z-thumb_z[0]
        
        # obtain model and states
        res_df = pd.read_csv(model_dir+"/results/"+name+".csv")
        z = np.array(res_df["syllable"].tolist())
        
        fig, axs = plt.subplots(2,1, figsize=(6,3),constrained_layout=True,gridspec_kw={'height_ratios': [2,.25]})
        axs[0].plot(point_z, c="gray", label="index")
        axs[0].plot(thumb_z, c="black", label="thumb")
        axs[0].set_ylabel('relative height (mm)')
        axs[0].set_xlim([0,len(point_z)])
        axs[0].tick_params(bottom=False, labelbottom=False)
        axs[0].legend(loc='upper left',frameon=False)

        colors = all_colors[:max(z)+1]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
        cax = axs[1].imshow(z[np.newaxis,:], cmap = cmap, aspect="auto", interpolation="nearest")
        axs[1].set_xlabel("frame")
        axs[1].set_ylabel("syllable")
        xticks = [int(x) for x in axs[1].get_xticks() if x <= len(z) and x >= 0]
        xticklabels = np.array(xticks)
        xticklabels = xticklabels.astype(int)
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xticklabels)
        axs[1].tick_params(left=False, labelleft=False)
        cbar = fig.colorbar(cax, fraction=0.6, location="bottom",orientation='horizontal', ticks=np.arange(7))
        # plt.subplots_adjust(hspace=0, left=0.1, right=0.9, top=0.9, bottom=0.3)
        # plt.subplots_adjust(hspace=0)
        plt.savefig(model_dir+"/figures/finger_plots/"+name+".pdf")

def plot_syllable_heel(save_dir, 
                   model_name, 
                   body_config,
                   plot_ids,
                   body):
    """
    Plots heel-related data (left and right heel movements) with syllable states.

    Parameters
    ----------
    save_dir : str
        Directory where model data is saved.
    model_name : str
        Name of the model whose results are being analyzed.
    body_config : function
        A function to configure body data.
    plot_ids : list of int
        List of subject IDs to plot.
    body : str
        Indicates the body type ('g' for gait data).

    Returns
    -------
    None
        Generates and saves plots showing heel movement (left and right).
    """

    if body != "g":
        raise ValueError("body is not gait, should not use plot_syllable_heel")
    
    right_heel = 25
    left_heel = 24
    segment = 0
    
    model_dir = os.path.join(save_dir, model_name) 
    if not os.path.isdir(model_dir+"/figures/heel_plots"):
        os.mkdir(model_dir+"/figures/heel_plots")
        
    all_colors = ['salmon','lightseagreen','orangered','darkturquoise','deepskyblue','rosybrown','orange']

    for i in plot_ids:
        # get z value of right and left heel
        name = "sub{}.{}".format(i,segment)
        left_z = get_zvals(save_dir, model_name, body_config, left_heel, i, name)
        left_z = left_z-left_z[0]
        right_z = get_zvals(save_dir, model_name, body_config, right_heel, i, name)
        right_z = right_z-right_z[0]
        
        # obtain model and states
        res_df = pd.read_csv(model_dir+"/results/"+name+".csv")
        z = np.array(res_df["syllable"].tolist())
        
        fig, axs = plt.subplots(2,1, figsize=(6,3),constrained_layout=True,gridspec_kw={'height_ratios': [2,.25]})
        axs[0].plot(left_z, c="b", label="left heel")
        axs[0].plot(right_z, c="r", label="right heel")
        axs[0].set_ylabel('relative height (mm)')
        axs[0].set_xlim([0,len(left_z)])
        axs[0].tick_params(bottom=False, labelbottom=False)
        axs[0].legend(loc='upper left',frameon=False)

        colors = all_colors[:max(z)+1]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
        cax = axs[1].imshow(z[np.newaxis,:], cmap = cmap, aspect="auto", interpolation="nearest")
        axs[1].set_xlabel("frame")
        axs[1].set_ylabel("syllable")
        xticks = [int(x) for x in axs[1].get_xticks() if x <= len(z) and x >= 0]
        xticklabels = np.array(xticks)
        xticklabels = xticklabels.astype(int)
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xticklabels)
        axs[1].tick_params(left=False, labelleft=False)
        cbar = fig.colorbar(cax, fraction=0.6, location="bottom",orientation='horizontal', ticks=np.arange(7))
        # plt.subplots_adjust(hspace=0, left=0.1, right=0.9, top=0.9, bottom=0.3)
        # plt.subplots_adjust(hspace=0)
        plt.savefig(model_dir+"/figures/heel_plots/"+name+".pdf")
    
def get_movies(save_dir,
                    model_name,
                    body_config,
                    kpms_config,
                    get_grid,
                    get_full,
                    get_trajectories,
                    names,
                    body, # config unpacked from here
                    combine,
                    augment,
                    ids,
                    splitting=None,
                    sampling_options={"n_neighbors": 1},
                    **kwargs):
    """
    Generates and saves movies for different body movements (e.g., grid, full movie, trajectory).

    Parameters
    ----------
    save_dir : str
        Directory where results are saved.
    model_name : str
        Name of the model whose results are being analyzed.
    body_config : function
        A function to configure body data.
    kpms_config : dict
        Configuration for the KPMS tool.
    get_grid : bool
        Whether to generate grid movie.
    get_full : bool
        Whether to generate full movie.
    get_trajectories : bool
        Whether to generate trajectory plots.
    names : list
        List of subject names.
    body : str
        Indicates body type ('g' for gait).
    combine : bool
        Whether to combine results from multiple subjects.
    augment : bool
        Whether to augment the data.
    ids : list
        List of subject IDs to process.
    splitting : dict
        How to group subjects
    **kwargs : additional keyword arguments
        Additional arguments passed to the movie generation functions.

    Returns
    -------
    None
        Generates and saves movie files.
    """

    # get data and results
    data_dict = quickly_get_data(**body_config())
    results = kpms.load_results(save_dir, model_name)

    if get_grid:
        if body == "g":
            instances = generate_grid_movies(results, save_dir, model_name, coordinates=data_dict["stand"], rows=3, cols=4, 
                                        min_frequency=0.0, min_duration=0, use_dims=[1,2], pre=0, post=80, 
                                        keypoints_only=True, rotate=True, fps=40, skip=True, **kpms_config())
        else:
            instances = generate_grid_movies(results, save_dir, model_name, coordinates=data_dict["stand"], rows=3, cols=4, 
                                        min_frequency=0.0, min_duration=0, use_dims=[1,2], pre=0, post=80, 
                                        keypoints_only=True, rotate=True, fps=40, skip=True, **kpms_config())

    if get_full:
        if body == "g":
            generate_labeled_movie(results, save_dir, model_name, coordinates=data_dict["stand"], post = 220, keypoints_only=True, keypoints_scale=0.3, window_size=800, use_dims=[1, 2], rotate=True, fps=40, subjects=names, **kpms_config())
        else:
            generate_labeled_movie(results, save_dir, model_name, coordinates=data_dict["stand"], post = 190, keypoints_only=True, fps=40, use_dims=[1,2], rotate=True, subjects=names, **kpms_config())

    if get_trajectories:
        kpms.generate_trajectory_plots(data_dict["rot"], results, save_dir, model_name, fps=40, min_frequency=0.0, projection_planes=["yz"], splitting=splitting, sampling_options=sampling_options,**kpms_config())

def check_dataframes(save_dir,model_name,group):
    """
    Checks and computes missing dataframes required for the analysis (e.g., 'moseq_df' and 'stats_df').

    Parameters
    ----------
    save_dir : str
        Directory where model data is saved.
    model_name : str
        Name of the model whose data is being processed.

    Returns
    -------
    None
        Ensures the necessary dataframes are present in the model directory.
    """

    model_dir = os.path.join(save_dir, model_name) 
    stats_dir = os.path.join(model_dir,'stats_df.csv')
    
    if not os.path.isdir(model_dir+"/figures/"):
        os.mkdir(model_dir+"/figures/")
    
    if not os.path.isfile(model_dir+"/moseq_df.csv"):
        print("computing moseq_df")
        moseq_df = kpms.compute_moseq_df(save_dir, model_name, group, smooth_heading=True) 
    else:
        moseq_df = pd.read_csv(model_dir+"/moseq_df.csv")
        moseq_df['group'] = moseq_df['group'].astype(str)
        
    if not os.path.isfile(stats_dir):
        print("computing stats_df")
        stats_df = kpms.compute_stats_df(save_dir, model_name, moseq_df, group, min_frequency=0.000, groupby=['group', 'name'], fps=80)
      
def get_transition_matrix(save_dir,
                          model_name,
                          group,
                          min_frequency):
    """
    Generates and visualizes transition matrices for syllables with a specified frequency threshold.

    Parameters
    ----------
    save_dir : str
        Directory where model data is saved.
    model_name : str
        Name of the model whose results are being analyzed.
    group : str
        Column in index.csv to group subjects.
    min_frequency : float
        Minimum frequency threshold for syllables to be included in the transition matrix.

    Returns
    -------
    None
        Generates and saves visualizations of transition matrices.
    """

    normalize='bigram' # normalization method ("bigram", "rows" or "columns")
    
    check_dataframes(save_dir, model_name,group)
    
    print("create transition matrices...")
    trans_mats, usages, groups, syll_include=kpms.generate_transition_matrices(
        save_dir, model_name, group, normalize=normalize,
        min_frequency=min_frequency
    )    

    kpms.visualize_transition_bigram(
        save_dir, model_name, groups, trans_mats, syll_include, group, normalize=normalize, 
        show_syllable_names=False 
    )
    
def get_syllable_quantitative_main(save_dir,
                              model_name,
                              group,
                              stats_df
                              ):
    """
    Computes and plots summary statistics (e.g., mean, std) of various features for each syllable.

    Parameters
    ----------
    save_dir : str
        Directory where model data is saved.
    model_name : str
        Name of the model whose results are being analyzed.
    group : str
        Column of index.csv to split subjects into groups

    Returns
    -------
    None
        Generates and saves plots of quantitative syllable features.
    """

    model_dir = os.path.join(save_dir, model_name) 
    
    check_dataframes(save_dir, model_name,group)
    df = stats_df.copy().drop(columns=['name'])

    new_group = []
    if df['group'].tolist()[0] in ["HT","PD"]:
        for group in df[group]:
            if group == 'HT':
                new_group.append(0)
            else:
                new_group.append(1)
        df['group'] = new_group
    
    syllable_df = df.groupby('syllable').mean().reset_index()

    plt.figure(figsize=(6,6))
    features = ['heading_mean','heading_std','angular_velocity_mean','angular_velocity_std']
    titles = ['mean heading', 'std heading', 'mean angular velocity','std angular velocity']
    
    if group == "group":
        color = ['limegreen','limegreen','indianred','indianred','limegreen','indianred','limegreen','limegreen']
    elif group == "stage_2":
        color = ['indianred','limegreen','indianred','indianred','limegreen','indianred','limegreen','limegreen']

    for i, col in enumerate(features):
        print(i,col)
        plt.subplot(2, 2, i+1)  # Create subplot for each column
        if "gait" in save_dir:
            plt.bar(syllable_df.index, syllable_df[col], color='black')
        else:
            plt.bar(syllable_df.index, syllable_df[col], color=color)
        plt.title(titles[i])
        plt.xlabel('syllable')

    plt.tight_layout()
    plt.savefig(model_dir+"/figures/syllable_char_main_{}.pdf".format(group))
    
def get_syllable_quantitative_all(save_dir,
                              model_name,
                              ):
    """
    Computes and plots summary statistics (e.g., mean, std) of all features for each syllable.

    Parameters
    ----------
    save_dir : str
        Directory where model data is saved.
    model_name : str
        Name of the model whose results are being analyzed.

    Returns
    -------
    None
        Generates and saves plots of all quantitative syllable features.
    """

    model_dir = os.path.join(save_dir, model_name) 
    
    check_dataframes(save_dir, model_name,"group")
    stats_dir = os.path.join(model_dir,'stats_df.csv')
    stats_df = pd.read_csv(stats_dir)
    df = stats_df.copy().drop(columns=['name'])

    new_group = []
    for group in df['group']:
        if group == 'HT':
            new_group.append(0)
        else:
            new_group.append(1)
    df['group'] = new_group
    
    syllable_df = df.groupby('syllable').mean().reset_index()
    
    color = ['limegreen','limegreen','indianred','indianred','limegreen','indianred','limegreen','limegreen']

    plt.figure(figsize=(18,10))
    for i, col in enumerate(syllable_df.columns):
        if i == 0:
            continue
        plt.subplot(3, 5, i)  # Create subplot for each column
        plt.bar(syllable_df.index, syllable_df[col], color=color)
        plt.title(col)
        plt.xlabel('Syllable')
        plt.ylabel('Value')

    plt.tight_layout()
    plt.savefig(model_dir+"/figures/syllable_char.pdf")
    

######### MOVIE FUNCTIONS BELOW ########

def get_syllable_instances(
    stateseqs,
    min_duration=3,
    pre=0,
    post=60,
    min_frequency=0,
    min_instances=0,
):
    """
    Extracts instances of syllables from state sequences based on specified duration, frequency, and count thresholds.

    Parameters
    ----------
    stateseqs : dict
        Dictionary containing syllable state sequences for each subject.
    min_duration : int
        Minimum duration of syllable instances to include.
    pre : int
        Number of frames before syllable start.
    post : int
        Number of frames after syllable end.
    min_frequency : float
        Minimum frequency of syllables for inclusion.
    min_instances : int
        Minimum number of instances required for syllable inclusion.

    Returns
    -------
    dict
        A dictionary containing syllable instances that meet the specified thresholds.
    """

    num_syllables = int(max(map(max, stateseqs.values())) + 1)
    syllable_instances = [[] for syllable in range(num_syllables)]
    movie_titles = [[] for syllable in range(num_syllables)]

    for key, stateseq in stateseqs.items():
        transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0] + 1
        starts = np.insert(transitions, 0, 0)
        ends = np.append(transitions, len(stateseq))
        for s, e, syllable in zip(starts, ends, stateseq[starts]):
            if e - s >= min_duration and s >= pre and s < len(stateseq) - post:
                syllable_instances[syllable].append((key, s, e))

    frequencies_filter = get_frequencies(stateseqs) >= min_frequency
    counts_filter = np.array(list(map(len, syllable_instances))) >= min_instances
    use_syllables = np.all([frequencies_filter, counts_filter], axis=0).nonzero()[0]
    return {syllable: syllable_instances[syllable] for syllable in use_syllables}

def write_video_clip(frames, path, fps=30, quality=7):
    """Write a video clip to a file.

    Parameters
    ----------
    frames : np.ndarray
        Video frames as a 4D array of shape `(num_frames, height, width, 3)`
        or a 3D array of shape `(num_frames, height, width)`.

    path : str
        Path to save the video clip.

    fps : int, default=30
        Framerate of video encoding.

    quality : int, default=7
        Quality of video encoding.
    """
    with imageio.get_writer(
        path, pixelformat="yuv420p", fps=fps, quality=quality
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
            
def get_grid_movie_window_size(
    sampled_instances,
    centroids,
    headings,
    coordinates,
    pre,
    post,
    pctl=90,
    fudge_factor=1.1,
    blocksize=16,
):
    """Automatically determine the window size for a grid movie.

    The window size is set such that across all sampled instances,
    the animal is fully visible in at least `pctl` percent of frames.

    Parameters
    ----------
    sampled_instances: dict
        Dictionary mapping syllables to lists of instances, where each
        instance is specified as a tuple with the video name, start frame
        and end frame.

    centroids: dict
        Dictionary mapping video names to arrays of shape `(n_frames, 2)`
        with the x,y coordinates of animal centroid on each frame

    headings: dict
        Dictionary mapping video names to arrays of shape `(n_frames,)`
        with the heading of the animal on each frame (in radians)

    coordinates: dict
        Dictionary mapping recording names to keypoint coordinates as
        ndarrays of shape (n_frames, n_bodyparts, 2).

    pre, post: int
        Number of frames before/after syllable onset that are included
        in the grid movies.

    pctl: int, default=95
        Percentile of frames in which the animal should be fully visible.

    fudge_factor: float, default=1.1
        Factor by which to multiply the window size.

    blocksize: int, default=16
        Window size is rounded up to the nearest multiple of `blocksize`.
    """
    all_trajectories = get_instance_trajectories(
        sum(sampled_instances.values(), []),
        coordinates,
        pre=pre,
        post=post,
        centroids=centroids,
        headings=headings,
    )

    all_trajectories = np.concatenate(all_trajectories, axis=0)
    all_trajectories = all_trajectories[~np.isnan(all_trajectories).all((1, 2))]
    max_distances = np.nanmax(np.abs(all_trajectories), axis=1)
    window_size = np.percentile(max_distances, pctl) * fudge_factor * 2
    window_size = int(np.ceil(window_size / blocksize) * blocksize)
    return window_size
    
def overlay_keypoints_on_image(
    image,
    coordinates,
    edges=[],
    keypoint_colormap="autumn",
    keypoint_colors=None,
    node_size=5,
    line_width=2,
    copy=False,
    opacity=1.0,
    title=None,  # New parameter for title
    rotate=False,
):
    """Overlay keypoints on an image.

    Parameters
    ----------
    image: ndarray of shape (height, width, 3)
        Image to overlay keypoints on.

    coordinates: ndarray of shape (num_keypoints, 2)
        Array of keypoint coordinates.

    edges: list of tuples, default=[]
        List of edges that define the skeleton, where each edge is a
        pair of indexes.

    keypoint_colormap: str, default='autumn'
        Name of a matplotlib colormap to use for coloring the keypoints.

    keypoint_colors : array-like, shape=(num_keypoints,3), default=None
        Color for each keypoint. If None, the keypoint colormap is used.
        If the dtype is int, the values are assumed to be in the range 0-255,
        otherwise they are assumed to be in the range 0-1.

    node_size: int, default=5
        Size of the keypoints.

    line_width: int, default=2
        Width of the skeleton lines.

    copy: bool, default=False
        Whether to copy the image before overlaying keypoints.

    opacity: float, default=1.0
        Opacity of the overlay graphics (0.0-1.0).

    Returns
    -------
    image: ndarray of shape (height, width, 3)
        Image with keypoints overlayed.
    """
    if copy or opacity < 1.0:
        canvas = image.copy()
    else:
        canvas = image

    if keypoint_colors is None:
        cmap = plt.colormaps[keypoint_colormap]
        colors = np.array(cmap(np.linspace(0, 1, coordinates.shape[0])))[:, :3]
    else:
        colors = np.array(keypoint_colors)

    if isinstance(colors[0, 0], float):
        colors = [tuple([int(c) for c in cs * 255]) for cs in colors]
        
    # overlay skeleton
    for i, j in edges:
        if np.isnan(coordinates[i, 0]) or np.isnan(coordinates[j, 0]):
            continue
        pos1 = (int(coordinates[i, 0]), int(coordinates[i, 1]))
        pos2 = (int(coordinates[j, 0]), int(coordinates[j, 1]))
        canvas = cv2.line(canvas, pos1, pos2, colors[i], line_width, cv2.LINE_AA)

    # overlay keypoints
    for i, (x, y) in enumerate(coordinates):
        if np.isnan(x) or np.isnan(y):
            continue
        pos = (int(x), int(y))
        canvas = cv2.circle(canvas, pos, node_size, colors[i], -1, lineType=cv2.LINE_AA)

    if opacity < 1.0:
        image = cv2.addWeighted(image, 1 - opacity, canvas, opacity, 0)
    
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)
        
    if title is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        if not isinstance(title, str):
            title = str(title)
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        # Calculate position for top right corner
        text_x = image.shape[1] - int(image.shape[1] / 5) - text_size[0]
        text_y = text_size[1] + int(image.shape[1] / 20)  # Slightly offset from the top
        
        org = (text_x, text_y)
        font_color = (255, 255, 255)  # White color
        image = cv2.putText(image, title, org, font, font_scale, font_color, thickness, cv2.LINE_AA)
        
    return image


def _grid_movie_tile(
    key,
    start,
    end,
    videos,
    centroids,
    headings,
    dot_color,
    window_size,
    scaled_window_size,
    pre,
    post,
    dot_radius,
    overlay_keypoints,
    edges,
    coordinates,
    plot_options,
    downsample_rate,
    title=None,
    rotate=False,
):
    scale_factor = scaled_window_size / window_size
    cs = centroids[key][start - pre : start + post]
    h, c = headings[key][start], cs[pre]
    # r = np.float32([[np.cos(h), np.sin(h)], [-np.sin(h), np.cos(h)]])
    r = np.float32([[1, 0], [0, 1]])
    c = coordinates[key][pre].mean(axis=0)
    print(c)

    tile = []
    assert overlay_keypoints, fill(
        "If no videos are provided, then `overlay_keypoints` must "
        "be True. Otherwise there is nothing to show"
    )
    scale_factor = scaled_window_size / window_size
    coords = coordinates[key][start - pre : start + post]
    coords = (coords - c) @ r.T * scale_factor + scaled_window_size // 2
    cs = (cs - c) @ r.T * scale_factor + scaled_window_size // 2
    # coords = (coords) @ r.T * scale_factor + scaled_window_size // 2
    # cs = (cs) @ r.T * scale_factor + scaled_window_size // 2
    background = np.zeros((scaled_window_size, scaled_window_size, 3))
    for ii, (uvs, c) in enumerate(zip(coords, cs)):
        if isinstance(title, list) or isinstance(title, np.ndarray):
            title_frame = title[ii]
        else:
            title_frame = title
        frame = overlay_keypoints_on_image(
            background.copy(), uvs, edges=edges, title=title_frame, rotate=rotate, **plot_options
        )
        # if 0 <= ii - pre <= end - start and dot_radius > 0:
        #     pos = (int(c[0]), int(c[1]))
        #     cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
        tile.append(frame)

    return np.stack(tile)

def grid_movie(
    instances,
    rows,
    cols,
    videos,
    centroids,
    headings,
    window_size,
    dot_color=(255, 255, 255),
    dot_radius=4,
    pre=0,
    post=60,
    scaled_window_size=None,
    edges=[],
    overlay_keypoints=False,
    coordinates=None,
    plot_options={},
    downsample_rate=1,
    titles=None,
    rotate=False,
):

    if scaled_window_size is None:
        scaled_window_size = window_size

    tiles = []
    for key, start, end in instances:
        tiles.append(
            _grid_movie_tile(
                key,
                start,
                end,
                videos,
                centroids,
                headings,
                dot_color,
                window_size,
                scaled_window_size,
                pre,
                post,
                dot_radius,
                overlay_keypoints,
                edges,
                coordinates,
                plot_options,
                downsample_rate,
                titles[key],
                rotate
            )
        )

    tiles = np.stack(tiles).reshape(
        rows, cols, post + pre, scaled_window_size, scaled_window_size, 3
    )
    frames = np.concatenate(np.concatenate(tiles, axis=2), axis=2)
    return frames

def generate_grid_movies(
    results,
    project_dir=None,
    model_name=None,
    output_dir=None,
    video_dir=None,
    video_paths=None,
    rows=4,
    cols=6,
    filter_size=9,
    pre=0,
    post=60,
    min_frequency=0.005,
    min_duration=0,
    dot_radius=4,
    dot_color=(255, 255, 255),
    quality=9,
    coordinates=None,
    bodyparts=None,
    use_bodyparts=None,
    sampling_options={},
    video_extension=None,
    max_video_size=1920,
    skeleton=[],
    keypoints_only=True,
    keypoints_scale=1,
    fps=80,
    plot_options={},
    use_dims=[0, 1],
    keypoint_colormap="autumn",
    downsample_rate=1,
    skip=False,
    rotate=False,
    **kwargs,
):
    print("entered function")
    overlay_keypoints = True

    # prepare output directory
    output_dir = _get_path(
        project_dir, model_name, output_dir, "grid_movies", "output_dir"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Writing grid movies to {output_dir}")

    # reindex coordinates if necessary
    if not (bodyparts is None or use_bodyparts is None or coordinates is None):
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    # get edges for plotting skeleton
    edges = []
    if len(skeleton) > 0 and overlay_keypoints:
        edges = get_edges(use_bodyparts, skeleton)

    # load results
    if results is None:
        results = load_results(project_dir, model_name)
    index_df = pd.read_csv(os.path.join(project_dir,'index.csv'))
    print("obtained index df")
    
    # extract syllables and labels from results
    syllables = {k: v["syllable"] for k, v in results.items()}
    labels = {k: index_df.group[index_df.name == k].tolist()[0] for k, v in results.items()}

    # extract and smooth centroids and headings
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    centroids, headings = filter_centroids_headings(
        centroids, headings, filter_size=filter_size
    )

    # scale keypoints if necessary
    if keypoints_only:
        for k, v in coordinates.items():
            coordinates[k] = v * keypoints_scale
        for k, v in centroids.items():
            centroids[k] = v * keypoints_scale

    videos = None

    # sample instances for each syllable
    syllable_instances = get_syllable_instances(
        syllables,
        pre=pre,
        post=post,
        min_duration=min_duration,
        min_frequency=min_frequency,
        min_instances=rows * cols,
    )
    
    print("got {} instances".format(len(syllable_instances)))

    if len(syllable_instances) == 0:
        warnings.warn(
            fill(
                "No syllables with sufficient instances to make a grid movie. "
                "This usually occurs when all frames have the same syllable label "
                "(use `plot_syllable_frequencies` to check if this is the case)"
            )
        )
        return

    sampled_instances = sample_instances(
        syllable_instances,
        rows * cols,
        coordinates=coordinates,
        centroids=centroids,
        headings=headings,
        **sampling_options,
    )
    print(sampled_instances)

    # if the data is 3D, pick 2 dimensions to use for plotting
    keypoint_dimension = next(iter(centroids.values())).shape[-1]
    if keypoint_dimension == 3:
        ds = np.array(use_dims)
        centroids = {k: v[:, ds] for k, v in centroids.items()}
        if coordinates is not None:
            coordinates = {k: v[:, :, ds] for k, v in coordinates.items()}

    # determine window size for grid movies
    window_size = get_grid_movie_window_size(
        sampled_instances, centroids, headings, coordinates, pre, post
    )
    print(f"Using window size of {window_size} pixels")
    if keypoints_only:
        if window_size < 64:
            warnings.warn(
                fill(
                    "The scale of the keypoints is very small. This may result in "
                    "poor quality grid movies. Try increasing `keypoints_scale`."
                )
            )

    # possibly reduce window size to keep grid movies under max_video_size
    scaled_window_size = max_video_size / max(rows, cols)
    scaled_window_size = int(np.floor(scaled_window_size / 16) * 16)
    scaled_window_size = min(scaled_window_size, window_size)
    scale_factor = scaled_window_size / window_size

    # add colormap to plot options
    plot_options.update({"keypoint_colormap": keypoint_colormap})

    # generate grid movies
    for syllable, instances in tqdm.tqdm(
        sampled_instances.items(), desc="Generating grid movies", ncols=72
    ):
        path = os.path.join(output_dir, f"syllable{syllable}.mp4")
        if os.path.exists(path):
            continue
        frames = grid_movie(
            instances,
            rows,
            cols,
            videos,
            centroids,
            headings,
            edges=edges,
            window_size=window_size,
            scaled_window_size=scaled_window_size,
            dot_color=dot_color,
            pre=pre,
            post=post,
            dot_radius=dot_radius,
            overlay_keypoints=overlay_keypoints,
            coordinates=coordinates,
            plot_options=plot_options,
            downsample_rate=downsample_rate,
            titles=labels,
            rotate=rotate,
        )
        write_video_clip(frames, path, fps=fps, quality=quality)

    return sampled_instances


def generate_labeled_movie(
    results,
    project_dir=None,
    model_name=None,
    output_dir=None,
    video_dir=None,
    video_paths=None,
    rows=1,
    cols=1,
    filter_size=9,
    pre=0,
    post=309,
    min_frequency=0.005,
    min_duration=3,
    dot_radius=4,
    dot_color=(255, 255, 255),
    quality=9,
    coordinates=None,
    bodyparts=None,
    use_bodyparts=None,
    sampling_options={},
    video_extension=None,
    max_video_size=1920,
    skeleton=[],
    keypoints_only=True,
    keypoints_scale=1,
    fps=80,
    plot_options={},
    use_dims=[0, 1],
    keypoint_colormap="autumn",
    downsample_rate=1,
    subjects=['sub1'],
    window_size=200,
    rotate=False,
    **kwargs,
):
    overlay_keypoints = True

    # prepare output directory
    output_dir = _get_path(
        project_dir, model_name, output_dir, "labeled_movies", "output_dir"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Writing labeled movies to {output_dir}")

    # reindex coordinates if necessary
    if not (bodyparts is None or use_bodyparts is None or coordinates is None):
        coordinates = reindex_by_bodyparts(coordinates, bodyparts, use_bodyparts)

    # get edges for plotting skeleton
    edges = []
    if len(skeleton) > 0 and overlay_keypoints:
        edges = get_edges(use_bodyparts, skeleton)

    # load results
    if results is None:
        results = load_results(project_dir, model_name)
    index_df = pd.read_csv(os.path.join(project_dir,'index.csv'))
    print("obtained index df")
    
    # extract syllables from results
    syllables = {k: v["syllable"] for k, v in results.items()}
    labels = {k: index_df.group[index_df.name == k].tolist()[0] for k, v in results.items()}
    
    # extract and smooth centroids and headings
    centroids = {k: v["centroid"] for k, v in results.items()}
    headings = {k: v["heading"] for k, v in results.items()}

    centroids, headings = filter_centroids_headings(
        centroids, headings, filter_size=filter_size
    )

    # scale keypoints if necessary
    if keypoints_only:
        for k, v in coordinates.items():
            coordinates[k] = v * keypoints_scale
        for k, v in centroids.items():
            centroids[k] = v * keypoints_scale

    videos = None

    # if the data is 3D, pick 2 dimensions to use for plotting
    keypoint_dimension = next(iter(centroids.values())).shape[-1]
    if keypoint_dimension == 3:
        ds = np.array(use_dims)
        centroids = {k: v[:, ds] for k, v in centroids.items()}
        if coordinates is not None:
            coordinates = {k: v[:, :, ds] for k, v in coordinates.items()}

    # possibly reduce window size to keep grid movies under max_video_size
    scaled_window_size = max_video_size / max(rows, cols)
    scaled_window_size = int(np.floor(scaled_window_size / 16) * 16)
    scaled_window_size = min(scaled_window_size, window_size)
    scale_factor = scaled_window_size / window_size

    # add colormap to plot options
    plot_options.update({"keypoint_colormap": keypoint_colormap})
    start, end = 0, post
    for key in tqdm.tqdm(subjects):
        tiles = []
        tiles.append(
            _grid_movie_tile(
                key,
                start,
                end,
                videos,
                centroids,
                headings,
                dot_color,
                window_size,
                scaled_window_size,
                pre,
                post,
                dot_radius,
                overlay_keypoints,
                edges,
                coordinates,
                plot_options,
                downsample_rate,
                syllables[key],
                rotate
            )
        )

        tiles = np.stack(tiles).reshape(
            rows, cols, post + pre, scaled_window_size, scaled_window_size, 3
        )
        frames = np.concatenate(np.concatenate(tiles, axis=2), axis=2)
    
        path = os.path.join(output_dir, f"{key}_full.mp4")
        write_video_clip(frames, path, fps=fps, quality=quality)

    return subjects