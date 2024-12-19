import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import statistics
from scipy.fft import fft
import pickle
from keypointmoseq.jaxmoseq.utils import get_durations, get_frequencies

def find_consecutive_sequences(labels):
    consecutive_sequences = []
    current_label = None
    start_index = 0
    for i, label in enumerate(labels):
        if label != current_label:
            if current_label is not None:
                consecutive_sequences.append((current_label, start_index, i - 1))
            current_label = label
            start_index = i
    consecutive_sequences.append((current_label, start_index, len(labels) - 1))
    return consecutive_sequences

def get_sub_freq(consecutive_sequences):
    frequences = np.zeros(1200)
    for item in consecutive_sequences:
        frequences[item[0]] += 1
    return frequences

def find_repeated_labels(labels):
    current_label = None
    consecutive_count = 0
    unique_repeated_labels = set()
    for label in labels:
        if label == current_label:
            consecutive_count += 1
        else:
            current_label = label
            consecutive_count = 1
        if consecutive_count > 1:
            unique_repeated_labels.add(label)
    return unique_repeated_labels

def most_freq_syll(consecutive_sequences):
    frequency = {}
    for label, _, _ in consecutive_sequences:
        frequency[label] = frequency.get(label, 0) + 1
    most_frequent_label = max(frequency, key=frequency.get)
    return most_frequent_label


def find_total_duration(consecutive_sequences, label):
    total_duration = 0
    for lbl, start_index, end_index in consecutive_sequences:
        if lbl == label:
            total_duration += (end_index - start_index + 1)
    return total_duration

def get_freq_finger(signal,signal_name):
    signal -= np.mean(signal)
    dft_result = fft(signal)
    magnitudes = np.abs(dft_result)
    max_magnitude_index = np.argmax(magnitudes)

    sampling_freq = 1
    num_samples = len(signal)
    frequency = max_magnitude_index / num_samples * sampling_freq
    magnitude = magnitudes[max_magnitude_index]
    
    print("Frequency with highest magnitude:", frequency)
    print("Magnitude at highest frequency:", magnitude)
    
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title(signal_name)

    # Plot the magnitude spectrum
    plt.subplot(2, 1, 2)
    plt.plot(np.fft.fftfreq(num_samples, d=1/sampling_freq), magnitudes)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')

    # Highlight the frequency with the highest magnitude
    plt.plot(frequency, magnitude, 'ro')  # Red dot for the highest magnitude frequency

    plt.tight_layout()
    plt.show()
    return frequency, magnitude

def freq_finger(sub,bod):
    distance = np.linalg.norm(sub[:,4,:]-sub[:,8,:], axis=1)
    frequency, magnitude = get_freq_finger(distance, "")
    return frequency, magnitude
    

def get_features(path_syll, results, sub, bod):
    df = pd.read_csv(path_syll)
    SYLL = df.syllable.tolist()
    unique_syllables = list(set(SYLL))
    unique_freq_syll = find_repeated_labels(SYLL)
    
    syllables = {k: res["syllable"] for k, res in results.items()}
    durations = get_durations(syllables)
    frequencies = get_frequencies(syllables)
    frequencies = frequencies[frequencies > 0.005]
    mean_duration = statistics.mean(durations)
    median_duration = statistics.median(durations)
    if len(durations) == 1:
        std_dev_duration = 0
    else:
        std_dev_duration = statistics.stdev(durations)
    mean_frequency = statistics.mean(frequencies)
    median_frequency = statistics.median(frequencies)
    if len(frequencies) == 1:
        std_dev_frequency = 0
    else:
        std_dev_frequency = statistics.stdev(frequencies)
    
    consecutive_sequences = find_consecutive_sequences(SYLL)
    most_freq_syllable = most_freq_syll(consecutive_sequences)
    total_duration = find_total_duration(consecutive_sequences, most_freq_syllable)
    
    # freq_tap = freq_finger(sub,bod)
    return [len(unique_syllables),len(unique_freq_syll),
            mean_duration,median_duration,std_dev_duration,
            mean_frequency, median_frequency, std_dev_frequency, total_duration/len(sub)]
    # return [len(unique_syllables),len(unique_freq_syll),
    #         mean_duration,median_duration,std_dev_duration,
    #         mean_frequency, median_frequency, std_dev_frequency, total_duration/len(sub), freq_tap]

    






































