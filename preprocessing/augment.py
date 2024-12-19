import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def smooth_signal(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def detect_flat_parts(signal, threshold):
    derivative = np.diff(signal)
    flat_indices = np.where(np.abs(derivative) < threshold)[0]
    return flat_indices

def exclude_high_slope_segments(flat_indices):
    filtered_indices = []
    segment_start = flat_indices[0]
    for i in range(1, len(flat_indices)):
        if flat_indices[i] - flat_indices[i-1] > 1:
            segment_end = flat_indices[i-1]
            if segment_end - segment_start > 100:
                filtered_indices.append((segment_start, segment_end))
            segment_start = flat_indices[i]
    filtered_indices.append((segment_start, flat_indices[-1]))
    return filtered_indices

def cut_signal(signal, segments):
    return [signal[start:end+1] for start, end in segments]

def flatten(lst):
    flat = []
    for sublist in lst:
        flat.extend(sublist)
    return flat


def clean_gaits(sub, plot):
    pt1 = 0
    pt2 = 18
    pt1_points_x = sub[:,pt1,0]
    pt2_points_x = sub[:,pt2,0]
    signal = pt1_points_x-pt2_points_x
    t = np.arange(len(signal))
    smoothed_signal = smooth_signal(signal,150)
    flat_indices = detect_flat_parts(smoothed_signal, threshold=1)
    filtered_indices = exclude_high_slope_segments(flat_indices)
    isolated_flat_parts = cut_signal(signal, filtered_indices)
    isolated_t = cut_signal(t, filtered_indices)

    if plot:
        plt.figure(figsize=(10,6))
        plt.subplot(3, 1, 1)
        plt.plot(t,signal, label='Original Signal')
        plt.title('Head minus back x-coordinate')
        plt.xlim(0, len(t))
        plt.subplot(3, 1, 2)
        plt.plot(t,smoothed_signal, label='Smoothed Signal')
        plt.title('Smoothed head and back movement')
        plt.xlim(0, len(t))
        plt.subplot(3, 1, 3)
        for p in range(len(isolated_flat_parts)):
            plt.plot(isolated_t[p], isolated_flat_parts[p], label=f'Flat Part {p+1}')
            plt.xlim(0, len(t))
        plt.title('Isolated head and back movement')
    
    if plot:
        print("plotting gait windows")
        plt.figure(figsize=(7,2))
        plt.plot(t,signal,color='k',linewidth=3)
        plt.xlim(0,len(t))
        for p in range(len(isolated_flat_parts)):
            plt.plot(isolated_t[p], isolated_flat_parts[p], color='red', linewidth=3)
            plt.xlim(0, len(t))
        plt.tick_params(bottom=False, labelbottom=False)
        plt.ylabel("x-coord of nose")
        plt.xlabel("frame")
        plt.savefig("preprocessing/gait_windows.pdf")
    
    return cut_signal(sub, filtered_indices)

def augment_fingers(sub, window_size, plot):
    distance = np.linalg.norm(sub[:,4,:]-sub[:,8,:], axis=1)

    peaks, _ = find_peaks(distance,distance=10)
    
    subs = []
    window_counter = 0
    while (window_counter < len(peaks)-window_size):
        window_peaks = peaks[window_counter:window_counter+window_size]
        subs.append(sub[min(window_peaks):max(window_peaks)])
        window_counter+=2
    
        if plot and window_counter == 4:
            print("plotting fingertap_window")
            plt.figure(figsize=(7,2))
            plt.plot(distance[min(window_peaks):max(window_peaks)], color='k',linewidth=2)
            plt.plot(window_peaks-min(window_peaks), distance[window_peaks], "x", color='red',label="peaks",markersize=5)
            plt.tick_params(bottom=False, labelbottom=False)
            plt.legend(frameon=False)
            plt.ylabel("index-thumb distance")
            plt.xlabel("frame")
            plt.savefig("preprocessing/fingertap_subwindows.pdf")
            
            plot = False
    
    return subs