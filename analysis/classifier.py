import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
import keypointmoseq as kpms
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from preprocessing.augment import *
from analysis.features import *
from preprocessing.data import *
from analysis.results import *

def plot_grouped_frequency(model_dir, total_df, thresh, num_syllables=8):
    """
    Plots the average frequency and duration of each syllable across subjects, grouped by UPDRS score threshold.

    Parameters:
    - model_dir (str): Directory where plots will be saved.
    - total_df (pd.DataFrame): DataFrame containing syllable frequency and duration statistics for each subject.
    - thresh (int): UPDRS score threshold to divide subjects into two groups.
    - group (str) : Column used to group subjects.
    - num_syllables (int): Number of unique syllables considered.

    Outputs:
    - Saves two bar plots (frequency and duration) in the 'figures' subdirectory of model_dir.
    """
    # aggregate the data by group
    df = total_df.copy().drop(columns=['name'])
    grouped = df.groupby('group').mean()
    syllable_range = range(num_syllables)

    # extract counts
    healthy_counts = grouped.loc[0, '0':str(num_syllables-1)]
    pd_counts = grouped.loc[1, '0':str(num_syllables-1)]

    # plot the data
    fig, ax = plt.subplots(figsize=(3,3))

    width = 0.35  # Width of the bars
    ax.bar([p - width/2 for p in syllable_range], healthy_counts, width, color='limegreen', label='UPDRS<{}'.format(thresh))
    ax.bar([p + width/2 for p in syllable_range], pd_counts, width, color='indianred', label='UPDRS>={}'.format(thresh))
    ax.set_xlabel('syllables')
    ax.set_ylabel('frequency')
    # ax.set_title('Frequency of Syllables (0-{}) in Patients Grouped by UPDRS'.format(num_syllables-1))
    ax.set_xticks(syllable_range)
    ax.set_xticklabels(syllable_range)
    ax.legend(bbox_to_anchor=(0.5, 1.4),loc='upper center',frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.savefig(model_dir + "/figures/frequency_grouped.pdf")
    plt.show()
    
    healthy_counts = grouped.loc[0, 'dur_0':'dur_'+str(num_syllables-1)]
    pd_counts = grouped.loc[1, 'dur_0':'dur_'+str(num_syllables-1)]

    # plot the data
    fig, ax = plt.subplots(figsize=(3, 3))

    width = 0.35  # Width of the bars
    ax.bar([p - width/2 for p in syllable_range], healthy_counts, width, color='limegreen', label='UPDRS<{}'.format(thresh))
    ax.bar([p + width/2 for p in syllable_range], pd_counts, width, color='indianred', label='UPDRS>={}'.format(thresh))
    ax.set_xlabel('syllables')
    ax.set_ylabel('duration (s)')
    # ax.set_title('Duration of Syllables (0-{}) in Patients Grouped by UPDRS'.format(num_syllables-1))
    ax.set_xticks(syllable_range)
    ax.set_xticklabels(syllable_range)
    ax.legend(bbox_to_anchor=(0.5, 1.4),loc='upper center',frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.savefig(model_dir + "/figures/duration_grouped.pdf")
    plt.show()
    
def plot_grouped_freq_standalone(save_dir, 
                                 model_name, 
                                 titles, 
                                 group, 
                                 ids, 
                                 body, 
                                 num_windows,
                                 num_syllables=8,
                                 **kwargs):
    index_df = pd.read_csv(os.path.join(save_dir,'index.csv'))
    model_dir = save_dir+"/"+model_name
    
    total_df = pd.DataFrame({'name':index_df.name,'group':index_df[group]})
    if not os.path.isfile(model_dir+"/moseq_df_{}.csv".format(group)):
        moseq_df = kpms.compute_moseq_df(save_dir, model_name, group, smooth_heading=True) 
    else:
        moseq_df = pd.read_csv(os.path.join(model_dir,'moseq_df_{}.csv'.format(group)))
    moseq_df['group'] = moseq_df['group'].astype(str)
    if not os.path.isfile(model_dir+"/stats_df_{}.csv".format(group)):
        stats_df = kpms.compute_stats_df(save_dir, model_name, moseq_df, group, min_frequency=0.000, groupby=['group', 'name'], fps=80)
    stats_df = pd.read_csv(model_dir+"/stats_df_{}.csv".format(group))
    
    syllable_counts, syllable_durations = get_counts_durations(model_dir,
                         True,
                         ids,
                         body,
                         stats_df,
                         num_windows,
                         num_syllables
                         )
        
    for syllable in syllable_counts.keys():
        total_df[str(syllable)] = syllable_counts[syllable]
    for syllable in syllable_counts.keys():
        total_df["dur_"+str(syllable)] = syllable_durations[syllable]
    
    df = total_df.copy().drop(columns=['name'])
    grouped = df.groupby('group').mean()
    syllable_range = range(num_syllables)

    # extract counts
    total_counts = []
    num_labels = len(list(index_df[group].unique()))
    if num_labels != len(titles):
        raise ValueError(
            "Number of plot titles and labels mismatch"
        )
    for i in range(num_labels):
        counts = grouped.loc[i, '0':str(num_syllables-1)]
        total_counts.append(counts)
        
    # plot the data
    fig, ax = plt.subplots(figsize=(3,3))

    width = 0.35  # Width of the bars
    if num_labels == 3:
        ax.bar([p - width/3 for p in syllable_range], total_counts[0], width, color='limegreen', label=titles[0])
        ax.bar([p for p in syllable_range], total_counts[1], width, color='orange', label=titles[1])
        ax.bar([p + width/3 for p in syllable_range], total_counts[2], width, color='indianred', label=titles[2])
    elif num_labels == 2:
        ax.bar([p - width/2 for p in syllable_range], total_counts[0], width, color='limegreen', label=titles[0])
        ax.bar([p + width/2 for p in syllable_range], total_counts[1], width, color='indianred', label=titles[1])
    else:
        raise ValueError(
            "Too few labels"
        )
    ax.set_xlabel('syllables')
    ax.set_ylabel('frequency')
    # ax.set_title('Frequency of Syllables (0-{}) in Patients Grouped by UPDRS'.format(num_syllables-1))
    ax.set_xticks(syllable_range)
    ax.set_xticklabels(syllable_range)
    ax.legend(bbox_to_anchor=(0.5, 1.4),loc='upper center',frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.savefig(model_dir + "/figures/frequency_grouped_{}.pdf".format(group))
    plt.show()
    
    total_counts = []
    for i in range(num_labels):
        counts = grouped.loc[i, 'dur_0':'dur_'+str(num_syllables-1)]
        total_counts.append(counts)

    # plot the data
    fig, ax = plt.subplots(figsize=(3, 3))

    width = 0.35  # Width of the bars
    if num_labels == 3:
        ax.bar([p - width/3 for p in syllable_range], total_counts[0], width, color='limegreen', label=titles[0])
        ax.bar([p for p in syllable_range], total_counts[1], width, color='orange', label=titles[1])
        ax.bar([p + width/3 for p in syllable_range], total_counts[2], width, color='indianred', label=titles[2])
    elif num_labels == 2:
        ax.bar([p - width/2 for p in syllable_range], total_counts[0], width, color='limegreen', label=titles[0])
        ax.bar([p + width/2 for p in syllable_range], total_counts[1], width, color='indianred', label=titles[1])
    else:
        raise ValueError(
            "Too few labels"
        )
    ax.set_xlabel('syllables')
    ax.set_ylabel('duration (s)')
    # ax.set_title('Duration of Syllables (0-{}) in Patients Grouped by UPDRS'.format(num_syllables-1))
    ax.set_xticks(syllable_range)
    ax.set_xticklabels(syllable_range)
    ax.legend(bbox_to_anchor=(0.5, 1.4),loc='upper center',frameon=False)
    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.savefig(model_dir + "/figures/duration_grouped_{}.pdf".format(group))
    plt.show()

def plotbar(vals,vals_name,labels,thresh,path):
    """
    Plots a bar chart comparing the mean value of a given metric between UPDRS-based subject groups.

    Parameters:
    - vals (list of float): Metric values for all subjects.
    - vals_name (str): Name of the metric to be used in the y-axis label and filename.
    - labels (list of int): Binary UPDRS-based group labels (0 or 1).
    - thresh (int): UPDRS score threshold used to generate labels.
    - path (str): Directory where the bar plot will be saved.

    Outputs:
    - Saves a bar chart with error bars comparing the mean metric value between groups.
    """
    values_0 = [vals[i] for i in range(len(vals)) if (labels[i] == 0 or labels[i] == 0)]
    values_1 = [vals[i] for i in range(len(vals)) if (labels[i] == 1 or labels[i] == 1)]
    std_err_0 = np.std(values_0) / np.sqrt(len(values_0))
    std_err_1 = np.std(values_1) / np.sqrt(len(values_1))
    plt.figure(figsize=(8,6))
    plt.bar(['UPDRS<{}'.format(thresh), 'UPDRS>={}'.format(thresh)], [np.mean(values_0), np.mean(values_1)],yerr=[std_err_0, std_err_1], capsize=5)
    plt.ylabel(vals_name)
    plt.xticks([0, 1], ['UPDRS<{}'.format(thresh), 'UPDRS>={}'.format(thresh)])  
    plt.savefig(path+"/bar_{}".format(vals_name))
    plt.show(block=True)
    
def get_labels(thresh=2,multiclass=False):
    """
    Retrieves UPDRS-based labels for each subject and task.

    Parameters:
    - thresh (int): UPDRS score threshold for binarizing labels.
    - multiclass (bool): If True, uses raw UPDRS scores as labels; otherwise uses binary thresholding.

    Returns:
    - left_labels (dict): Subject-wise labels for left-hand finger tapping.
    - right_labels (dict): Subject-wise labels for right-hand finger tapping.
    - gait_labels (dict): Subject-wise labels for gait task.
    - left_right_labels (dict): Dictionary of left and right labels for each subject, grouped by 'l' and 'r'.
    """
    df_syll = pd.read_csv("training/data/IndexFingertapping_3DPoses_Labels/Indexfinger_gait_label_final.csv")
    behavorial_tests = ["Finger tapping - Right hand", "Finger tapping - Left hand", "Gait"]
    subjects = np.arange(1,16)
    left_labels, right_labels, gait_labels = {}, {}, {}
    for subject in subjects:
        for test in behavorial_tests:
            if multiclass: 
                left_labels[subject] = df_syll[(df_syll.UPDRS_name == 'Finger tapping - Left hand') & (df_syll.Subject == subject)]['gold_standard'].tolist()[0]
                right_labels[subject] = df_syll[(df_syll.UPDRS_name == 'Finger tapping - Right hand') & (df_syll.Subject == subject)]['gold_standard'].tolist()[0]
                gait_labels[subject] = df_syll[(df_syll.UPDRS_name == 'Gait') & (df_syll.Subject == subject)]['gold_standard'].tolist()[0]
            else:
                left_labels[subject] = df_syll[(df_syll.UPDRS_name == 'Finger tapping - Left hand') & (df_syll.Subject == subject)]['thresh_{}'.format(thresh)].tolist()[0]
                right_labels[subject] = df_syll[(df_syll.UPDRS_name == 'Finger tapping - Right hand') & (df_syll.Subject == subject)]['thresh_{}'.format(thresh)].tolist()[0]
                gait_labels[subject] = df_syll[(df_syll.UPDRS_name == 'Gait') & (df_syll.Subject == subject)]['thresh_{}'.format(thresh)].tolist()[0]
    return left_labels, right_labels, gait_labels, {"l":left_labels, "r":right_labels}

def get_counts_durations(model_dir,
                         augmentation,
                         ids,
                         body,
                         stats_df,
                         num_windows,
                         num_syllables
                         ):
    syllable_range = range(num_syllables)
    num_segments = {}
    syllable_counts = {syllable: [] for syllable in syllable_range}
    syllable_durations = {syllable: [] for syllable in syllable_range}
    
    if augmentation and body != "lr":
        new_labels = {}
        for id in ids:
            sub, skeleton_config = get_raw_data(body, id, False)
            subs = augment_data(sub, id, num_windows, body)
            num_segments[id] = len(subs)
    elif augmentation and body == "lr":
        new_labels = {}
        for id in ids:
            print("updating configuration for patient {}...".format(id))
            for bod in body:
                sub, skeleton_config = get_raw_data(bod, id, True)
                subs = augment_data(sub, id, num_windows, bod)
                
                num_segments[str(id)+bod] = len(subs)
    else:
        print("no augmentation case hasn't been implemented yet")
                    
    if augmentation:
        for id in ids:
            for bod in body:
                numseg = num_segments[str(id)+bod] if body == "lr" else num_segments[id]
                for l in range(numseg):
                    if body == "lr":
                        name = '{}sub{}.{}'.format(bod,id,l)
                    else:
                        name = 'sub{}.{}'.format(id,l)
                    
                    df = pd.read_csv(os.path.join(model_dir,'results/{}.csv'.format(name)))
                    syllables = df.syllable.tolist()
                    freq = get_sub_freq(find_consecutive_sequences(syllables)) # more like number of consecutive sequences of a syllable
                    counts = {i:freq[i] for i in syllable_range}
                    
                    for syllable in syllable_range:
                        syllable_counts[syllable].append(counts[syllable])
                        df_dur = stats_df[(stats_df.name == name)&(stats_df.syllable == syllable)]['duration']
                        if (len(df_dur) > 0):
                            syllable_durations[syllable].append(df_dur.tolist()[0])
                        else:
                            syllable_durations[syllable].append(0.0)
    else:
        for id in ids:
            name = 'sub{}'.format(id)
            df = pd.read_csv(os.path.join(model_dir,'results/sub{}.csv'.format(id)))
            syllables = df.syllable.tolist()
            freq = get_sub_freq(find_consecutive_sequences(syllables))
            counts = {i:freq[i] for i in syllable_range}
            for syllable in syllable_range:
                syllable_counts[syllable].append(counts[syllable])
                df_dur = stats_df[(stats_df.name == name)&(stats_df.syllable == syllable)]['duration']
                if (len(df_dur) > 0):
                    syllable_durations[syllable].append(df_dur.tolist()[0])
                else:
                    syllable_durations[syllable].append(0.0)
    return syllable_counts, syllable_durations


def classification(save_dirs, 
                   model_names, 
                   num_syllables, 
                   thresh, 
                   body,
                   body_name,
                   num_windows,
                   augmentation, 
                   multiclass=False, 
                   class_models=['xgboost','logistic'], 
                   plot_grouped_freq=False, 
                   classification=True,
                   plot_bars=False):
    """
    Performs behavioral classification using syllable-based features extracted from motion capture data.

    This function loads data for specified subjects, applies augmentation if requested, extracts 
    statistical and transition-based features from MoSeq syllable usage, and trains classifiers 
    (XGBoost or Logistic Regression) to distinguish behavior groups.

    Parameters:
        save_dirs (list of str): Directories where model checkpoints are saved.
        model_names (list of str): Names of trained MoSeq models to analyze.
        num_syllables (int): Number of syllables in the MoSeq model.
        thresh (int): UPDRS threshold to define binary classification groups.
        body (str): One of "g", "l", "r", or "lr" indicating the body part or combination used.
        body_name (str): String identifier for the body part (used in naming or display).
        num_windows (int): Number of windows for augmentation.
        augmentation (bool): If True, applies data augmentation (sliding window segmentation).
        multiclass (bool, optional): If True, uses multiclass classification. Defaults to False.
        class_models (list of str, optional): Classifier types to use. Options include 'xgboost' and 'logistic'.
        plot_grouped_freq (bool, optional): If True, generates plots for grouped syllable frequency. Defaults to False.
        classification (bool, optional): If True, run classification. Defaults to True.
        plot_bars (bool, optional): If True and binary classification, plots bar plots of key features. Defaults to False.
    """
    left_labels, right_labels, gait_labels, left_right_labels = get_labels(thresh, multiclass)
    if body == "g":
        labels = gait_labels
    elif body == "r":
        labels = right_labels
    elif body == "l":
        labels = left_labels
    elif body == "lr":
        labels = left_right_labels
    else:
        raise ValueError("wrong body key")
    
    ids = [1,10,11,12,13,14,15,2,3,4,5,6,7,8,9]
    
    frame_nums = {}
    num_segments = {}
    if augmentation and body != "lr":
        new_labels = {}
        for id in ids:
            sub, skeleton_config = get_raw_data(body, id, False)
            subs = augment_data(sub, id, num_windows, body)
            num_segments[id] = len(subs)
            
            for l in range(num_segments[id]):
                num_frames = subs[l].shape[0]
                name = 'sub{}.{}'.format(id,l)
                new_labels[name] = labels[id]
                frame_nums[name] = num_frames
        labels = new_labels
    
    if augmentation and body == "lr":
        new_labels = {}
        for id in ids:
            print("updating configuration for patient {}...".format(id))
            for bod in body:
                sub, skeleton_config = get_raw_data(bod, id, True)
                subs = augment_data(sub, id, num_windows, bod)
                
                num_segments[str(id)+bod] = len(subs)
                for l in range(num_segments[str(id)+bod]):
                    num_frames = subs[l].shape[0]
                    name = "{}sub{}.{}".format(bod,id,l)
                    frame_nums[name] = num_frames
                    new_labels[name] = labels[bod][id]
        labels = new_labels
    
    numlabels = len(set(labels.values()))
    
    # if multiclass than logistic doesn't work
    if multiclass:
        class_models = ['xgboost']
    
    for save_dir, model_name in tqdm.tqdm(zip(save_dirs, model_names), total=len(save_dirs)):
        print("Classifying results in {}...".format(model_name))
        model_dir = os.path.join(save_dir, model_name) # directory to save the moseq_df dataframe
        
        total_df = pd.DataFrame({'name':list(labels.keys()),'group':list(labels.values())})
        print(set(total_df.group))
        syllable_range = range(num_syllables)
        
        if not os.path.isdir(model_dir+"/results"):
            kpms.reindex_syllables_in_checkpoint(save_dir, model_name)
            model, data, metadata, current_iter = kpms.load_checkpoint(save_dir, model_name)
            results = kpms.extract_results(model, metadata, save_dir, model_name)
            kpms.save_results_as_csv(results, save_dir, model_name)
            
        if not os.path.isdir(model_dir+"/figures/"):
            os.mkdir(model_dir+"/figures/")
            
        if not os.path.isfile(model_dir+"/moseq_df.csv"):
            moseq_df = kpms.compute_moseq_df(save_dir, model_name, "group", smooth_heading=True) 
            
        if not os.path.isfile(model_dir+"/stats_df.csv"):
            stats_df = kpms.compute_stats_df(save_dir, model_name, moseq_df, min_frequency=0.000, groupby=['group', 'name'], fps=80)
        stats_df = pd.read_csv(model_dir+"/stats_df.csv")
        results = kpms.load_results(save_dir, model_name)
        syllables = {k: res["syllable"] for k, res in results.items()}
        
        # get number of consecutive sequences and duration of a syllable in each subject
        syllable_counts, syllable_durations = get_counts_durations(model_dir,
                         augmentation,
                         ids,
                         body,
                         stats_df,
                         num_windows,
                         num_syllables
                         )
        
        for syllable in syllable_counts.keys():
            total_df[str(syllable)] = syllable_counts[syllable]
        for syllable in syllable_counts.keys():
            total_df["dur_"+str(syllable)] = syllable_durations[syllable]
        
        if plot_grouped_freq:
            plot_grouped_frequency(model_dir, total_df, thresh, num_syllables)
        
        features_add = {'unique_syllables':[],'consistency':[],'transitions':[]}
        unique_syllables, consistency, transitions = [], [], []
        
        if augmentation:
            for id in ids:
                for bod in body:
                    numseg = num_segments[str(id)+bod] if body == "lr" else num_segments[id]
                    for l in range(numseg):
                        if body == "lr":
                            name = '{}sub{}.{}'.format(bod,id,l)
                        else:
                            name = 'sub{}.{}'.format(id,l)
                        df = pd.read_csv(os.path.join(model_dir,'results/{}.csv'.format(name)))
                        syllables = df.syllable.tolist()
                        features_add['transitions'].append(len(find_consecutive_sequences(syllables)))
                        most_freq = most_freq_syll(find_consecutive_sequences(syllables))
                        features_add['consistency'].append(find_total_duration(find_consecutive_sequences(syllables), most_freq))
                        features_add['unique_syllables'].append(len(set(syllables)))
        else:
            for id in ids:
                df = pd.read_csv(os.path.join(model_dir,'results/sub{}.csv'.format(id)))
                syllables = df.syllable.tolist()
                features_add['transitions'].append(len(find_consecutive_sequences(syllables)))
                most_freq = most_freq_syll(find_consecutive_sequences(syllables))
                features_add['consistency'].append(find_total_duration(find_consecutive_sequences(syllables), most_freq))
                features_add['unique_syllables'].append(len(set(syllables))) 
        
        features_df = total_df.copy()
        for feature in features_add:
            features_df[feature] = features_add[feature]
        
        print("starting classification...")
        if classification:
            y = np.array(list(labels.values()))
            if type(y[0]) == str:
                y = [0 if x == 'HT' else 1 for x in y]
            X = features_df.copy()
            
            
            if plot_bars and not multiclass:
                plotbar(features_df.unique_syllables.tolist(),"unique_syllables",y,thresh,os.path.join(model_dir,"figures"))
                plotbar(features_df.transitions.tolist(),"transitions",y,thresh,os.path.join(model_dir,"figures"))
            
            # split data into training and testing sets
            # if augmented, all windows from a test subject are in the test set
            np.random.seed(42)
            if augmentation:
                leave_out_sub = [13,14,15]
                leave_out_list = []
                for item in features_df.name.tolist():
                    if any(str(sub) in item for sub in leave_out_sub):
                        leave_out_list.append(item)
            
                print("leave_out ", leave_out_list)
                mask = features_df['name'].isin(leave_out_list)
                X_train, X_test, y_train, y_test = X[~mask], X[mask], y[(~mask).values], y[mask.values]
                X_train, X_test = X_train.drop(columns=['name','group']), X_test.drop(columns=['name','group'])
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            feature_names = X_test.columns
            print(feature_names)
            
            # scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=feature_names)
                            
            for model_type in class_models:
                if model_type == 'xgboost':
                    base_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=numlabels)
                    param_grid = {
                        'max_depth': [1,2,3,5,7],
                        'learning_rate': [0.2,0.1,0.01,0.001],
                        'n_estimators': [20,50,70,100,150,200]
                    } 
                elif model_type == 'logistic':
                    base_classifier = LogisticRegression(max_iter=1000)
                    param_grid = {
                        'C': [0.001, 0.01, 0.1, 1, 10, 100]
                    }
                
                grid_search = GridSearchCV(estimator=base_classifier, param_grid=param_grid, cv=3, scoring='accuracy',error_score='raise')
                grid_search.fit(X_train, y_train)

                best_params = grid_search.best_params_
                print("Best Hyperparameters:", best_params)

                if model_type =='xgboost':
                    best_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=numlabels, **best_params)
                elif model_type == 'logistic':
                    best_classifier = LogisticRegression(max_iter=1000, **best_params)

                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                accuracy_scores = cross_val_score(best_classifier, X_train, y_train, cv=cv, scoring='accuracy')
                f1_scores = cross_val_score(best_classifier, X_train, y_train, cv=cv, scoring='f1_macro')

                print("Mean Accuracy:", accuracy_scores.mean())
                print("Mean F1 Score:", f1_scores.mean())

                best_classifier.fit(X_train, y_train)
                y_pred = best_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("Accuracy:", accuracy)

                if model_type == 'xgboost':
                    explainer = shap.Explainer(best_classifier)
                    shap_values = explainer.shap_values(X_test)
                    
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_values, X_test)
                    plt.savefig(save_dir+"/"+model_name+"/figures/shap.pdf")
                    
                    shap_values = np.moveaxis(shap_values, 2, 0)
                    
                    print(shap_values.shape)
                    print(X_test.shape)
                    
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_values[0], X_test)
                    plt.savefig(save_dir+"/"+model_name+"/figures/shap_ht.pdf")
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_values[1], X_test)
                    plt.savefig(save_dir+"/"+model_name+"/figures/shap_pd.pdf")
                    
                else:
                    print(best_classifier.coef_)

        
                conf_matrix = confusion_matrix(y_test, y_pred)
                if not multiclass:
                    class_labels = ["UPDRS<{}".format(thresh), "UPDRS>={}".format(thresh)]
                    suffix = ""
                else:
                    class_labels = set(y_train)
                    suffix = "_multi"
                
                print(class_labels)
                
                plt.figure(figsize=(3,3))
                sns.heatmap(conf_matrix, annot=True, annot_kws={"fontsize":15}, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels, cbar=False)
                plt.xlabel('predicted')
                plt.ylabel('actual')
                # plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig(model_dir +"/figures/confusion_{}{}.pdf".format(model_type,suffix))
                plt.show(block=True)  

def run_classifier(save_dir, 
                   prefix, 
                   body,
                   body_name,
                   num_windows,
                   num_states,
                   thresh,
                   augment,
                   multiclass,
                   plot_grouped_freq,
                   classification,
                   plot_bars,
                   class_models,
                   **kwargs):
    """
    Finds trained MoSeq models in a directory and applies classification to all of them.

    This wrapper function searches for all model subdirectories matching a given prefix and then calls
    the `classification()` function on each to evaluate performance and visualize results.

    Parameters:
        save_dir (str): Path to the directory containing MoSeq model subfolders.
        prefix (str): Model name prefix to filter relevant model folders.
        body (str): One of "g", "l", "r", or "lr" indicating the body part or combination used.
        body_name (str): Display name for the body part.
        num_windows (int): Number of sliding windows for data augmentation.
        num_states (int): Number of behavioral syllables (MoSeq states).
        thresh (int): UPDRS threshold for grouping subjects.
        augment (bool): If True, uses augmented data in classification.
        multiclass (bool): If True, performs multiclass classification instead of binary.
        plot_grouped_freq (bool): Whether to plot grouped syllable frequency bar plots.
        plot_bars (bool): Whether to plot additional bar plots for features.
        class_models (list of str): List of classifier types to use (e.g., ['xgboost', 'logistic']).
        **kwargs: Additional keyword arguments (currently unused but passed through).
    """
    # get all models in save directory
    model_names = []
    for subdir, dirs, files in os.walk(save_dir):
        if len(subdir.split("/")[-1])>0 and prefix in subdir.split("/")[-1]:
            print(subdir.split("/")[-1])
            model_names.append(subdir.split("/")[-1])
    save_dirs = [save_dir]*len(model_names)
    print(model_names)
    
    classification(save_dirs, model_names, num_states, thresh, body, body_name, num_windows, augment, multiclass, class_models, plot_grouped_freq, plot_bars)
            
                    
                    
                    
                    
                    
