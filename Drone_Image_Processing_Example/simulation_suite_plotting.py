"""
Simulation Plotting Suite for Drone Image Processing Experiments

This module provides plotting utilities for visualizing the results of experiments
conducted with the Drone Image Processing system, evaluating the Reflex pattern's effectiveness.

Dependencies:
    - Python libraries: os, json, numpy, matplotlib, scipy

Functions:
    plot_experiment_1(results_file, output_dir): Generates plots for Experiment 1 comparing CPU usage, FPS, and Queue Fill level.
    plot_experiment_2(results_file, output_dir): Generates plots for Experiment 2 showing the impact of switching delays.

Usage:
    python simulation_suite_plotting.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth low-pass filter to smooth data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(N=order, Wn=normal_cutoff, output='ba')
    return filtfilt(b, a, data, padlen=len(data) - 1)


def plot_experiment_1(results_file, output_dir):
    """Generates CPU usage, FPS, and Queue Fill level plots for Experiment 1 results."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Print summary stats for each mode
    for mode in results:
        print(f"mAP {mode}: {results[mode]['mAP']}")
        print(f"Total frames processed {mode}: {results[mode]['total_frames']}")
        print(f"Dropped frames {mode}: {results[mode]['dropped_frames']}")

    # Extract experiment Data
    complex_data = results['complex']
    reflex_data = results['reflex']
    reflex_only_data = results['simple']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 4.2), sharex=True)

    # Queue Fill Level Plot
    ax1.plot(complex_data['queue_time_stamps'],
             complex_data['queue_sizes'],
             label='Conventional System',
             color='#E7004C', linewidth=0.7)
    ax1.plot(reflex_only_data['queue_time_stamps'],
             reflex_only_data['queue_sizes'],
             label='Reflex-Only System',
             color='#E7004C', linestyle='dashed',
             linewidth=0.7)
    ax1.plot(reflex_data['queue_time_stamps'],
             reflex_data['queue_sizes'],
             label='Reflex-Enabled System',
             color='#3A8DDE',
             linewidth=0.7)

    # Shade Reflex intervals
    secondary_intervals = []
    start = None
    for i in range(1, len(reflex_data['inference_active_system'])):
        prev_state = reflex_data['inference_active_system'][i - 1]
        current_state = reflex_data['inference_active_system'][i]
        if current_state == 'reflex' and prev_state == 'complex':
            start = reflex_data['queue_time_stamps'][i]
        elif current_state == 'complex' and prev_state == 'reflex':
            end = reflex_data['queue_time_stamps'][i]
            if start is not None:
                secondary_intervals.append((start, end))
                start = None

    if reflex_data['inference_active_system'][-1] == 'reflex' and start is not None:
        secondary_intervals.append((start, reflex_data['queue_time_stamps'][-1]))

    for interval in secondary_intervals:
        ax1.axvspan(interval[0], interval[1], color='#FF7F41', alpha=0.2)

    handles = (ax1.get_legend_handles_labels()[0]
               + [mpatches.Patch(color='#FF7F41', alpha=0.2, label='Reflex Algorithm Active')])

    ax1.legend(handles=handles, fontsize=8,
               loc='lower right')
    ax1.set_title('(a) Queue Fill (%) Level Over Time', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, 100)

    # FPS Plot
    ax2.plot(complex_data['fps_time_stamps'],
             complex_data['fps_over_time'],
             label='Conventional System',
             color='#E7004C',
             linewidth=0.7)
    ax2.plot(reflex_only_data['fps_time_stamps'],
             reflex_only_data['fps_over_time'],
             label='Reflex-Only System',
             color='#E7004C',
             linestyle='dashed',
             linewidth=0.7)
    ax2.plot(reflex_data['fps_time_stamps'],
             reflex_data['fps_over_time'],
             label='Reflex-Enabled System',
             color='#3A8DDE',
             linewidth=0.7)
    ax2.set_title(f'(b) Frames per second (FPS) Over Time', fontsize=10)
    ax2.set_ylim(0, 18)  # FPS over time.
    ax2.set_xlim(0, 100)

    # CPU Usage Plot
    smoothed_cpu_usage_complex = butter_lowpass_filter(complex_data['cpu_usage_over_time'], 1.0, 20)
    smoothed_cpu_usage_reflex_only = butter_lowpass_filter(reflex_only_data['cpu_usage_over_time'], 1.0, 20)
    smoothed_cpu_usage_reflex = butter_lowpass_filter(reflex_data['cpu_usage_over_time'], 1.0, 20)

    ax3.plot(complex_data['cpu_time_stamps'],
             smoothed_cpu_usage_complex,
             label='Complex-Only System',
             color='#E7004C',
             linewidth=0.7)
    ax3.plot(reflex_only_data['cpu_time_stamps'],
             smoothed_cpu_usage_reflex_only,
             label='Reflex-Only System',
             color='#E7004C',
             linestyle='dashed',
             linewidth=0.7)
    ax3.plot(reflex_data['cpu_time_stamps'],
             smoothed_cpu_usage_reflex,
             label='Reflex-Enabled System',
             color='#3A8DDE',
             linewidth=0.7)
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_title(f'(c) Processor utilization (%) Over Time', fontsize=10)
    ax3.set_ylim(30, 110)
    ax3.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'experiment_1_results.pdf'), dpi=300)
    plt.close()


def plot_experiment_2(results_file, output_dir):
    """Generates CPU usage and error rate plots for Experiment 2 results."""
    with open(results_file, 'r') as f:
        results = json.load(f)

    def plot_queue_fill_level_comparison():
        # Plot CPU Usage over time Comparison between different switching delays
        fig, axs = plt.subplots(3, 1, figsize=(8, 4.2), sharex=True)
        delays = ['2', '5', '10']
        number_mapping = {0: 'a', 1: 'b', 2: 'c'}
        for idx, delay in enumerate(delays):
            delay_data = results[delay]
            axs[idx].plot(delay_data['queue_time_stamps'],
                          delay_data['queue_sizes'],
                          color='#3A8DDE',
                          linewidth=0.7)

            secondary_intervals = []
            start = None
            for i in range(1, len(delay_data['inference_active_system'])):
                prev_state = delay_data['inference_active_system'][i - 1]
                current_state = delay_data['inference_active_system'][i]
                if current_state == 'reflex' and prev_state == 'complex':
                    start = delay_data['queue_time_stamps'][i]
                elif current_state == 'complex' and prev_state == 'reflex':
                    end = delay_data['queue_time_stamps'][i]
                    if start is not None:
                        secondary_intervals.append((start, end))
                        start = None

            if delay_data['inference_active_system'][-1] == 'reflex' and start is not None:
                secondary_intervals.append((start, delay_data['queue_time_stamps'][-1]))

            for interval in secondary_intervals:
                axs[idx].axvspan(interval[0], interval[1], color='#FF7F41', alpha=0.2)
            # Create custom patch for the legend to represent shaded areas
            secondary_patch = mpatches.Patch(color='#FF7F41', alpha=0.2, label='Reflex Algorithm Active')
            # Combine existing handles and labels with the custom patch
            handles, labels = axs[idx].get_legend_handles_labels()
            handles.append(secondary_patch)
            labels.append('Reflex Algorithm Active')
            axs[idx].set_title(f'({number_mapping[idx]}) Queue Fill Level Over Time (Delay = {delay}s)', fontsize=12)
            axs[idx].set_ylim(0, 105)  # Queue Fill Level
            axs[idx].set_xlim(0, 505)

        axs[2].legend(handles=handles, fontsize=10, loc='lower right')
        axs[2].set_xlabel('Time (s)', fontsize=12)
        axs[1].set_ylabel('Queue Fill Level (%)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment_2_cpu_usage_over_time_delays.pdf'), dpi=300)
        plt.close()

    def plot_error_rate_over_time():
        # Plot CPU Usage over time Comparison between different switching delays
        plt.figure(figsize=(8, 4))
        for delay in ['0', '2', '5', '10', '15', '20', '25']:
            delay_data = results[delay]
            cumulated_dropped_frames_over_time = np.cumsum(delay_data['dropped_frames_over_time']).tolist()
            plt.plot(delay_data['dropped_frames_time_stamps'], cumulated_dropped_frames_over_time, label=f'Delay = {delay}s')

        plt.title('Error Rate Over Time for Different Switching Delays')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative Dropped Frames (%)')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'experiment_2_results_error_rate_over_time.pdf'), dpi=300)
        plt.close()

    plot_queue_fill_level_comparison()
    plot_error_rate_over_time()


def main():
    """Main function to execute plotting for both experiments."""
    output_dir = 'simulation_plots'
    os.makedirs(output_dir, exist_ok=True)
    results_file_experiment_1 = 'simulation_results/experiment_1_results.json'
    plot_experiment_1(results_file_experiment_1, output_dir)
    results_file_experiment_2 = 'simulation_results/experiment_2_results.json'
    plot_experiment_2(results_file_experiment_2, output_dir)

if __name__ == '__main__':
    main()
