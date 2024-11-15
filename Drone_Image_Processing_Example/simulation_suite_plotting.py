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

    smoothed_dropped_frames_complex = butter_lowpass_filter(complex_data['dropped_frames_over_time'], 10, 30)
    smoothed_dropped_frames_reflex_only = butter_lowpass_filter(reflex_only_data['dropped_frames_over_time'], 10, 30)
    smoothed_dropped_frames_reflex = butter_lowpass_filter(reflex_data['dropped_frames_over_time'], 10, 30)

    ax1.plot(complex_data['dropped_frames_time_stamps'],
             smoothed_dropped_frames_complex,
             label='Complex-Only System',
             color='#E7004C', linewidth=0.7)
    ax1.plot(reflex_only_data['dropped_frames_time_stamps'],
             smoothed_dropped_frames_reflex_only,
             label='Reflex-Only System',
             color='#E7004C', linestyle='dashed',
             linewidth=0.7)
    ax1.plot(reflex_data['dropped_frames_time_stamps'],
             smoothed_dropped_frames_reflex,
             label='Reflex-Enabled System',
             color='#3A8DDE',
             linewidth=0.7)

    ax1.legend(fontsize=8,
               loc='lower right')
    ax1.set_title('(a) Dropped Frames per Second Over Time', fontsize=10)
    # Add a grid
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.set_yticks([0, 1, 2])
    ax1.set_ylim(-0.2, 3)
    ax1.set_xlim(0, 100)

    time_stamps_complex = sorted(results['complex']['mAP_per_second'].keys())
    time_stamps_simple = sorted(results['simple']['mAP_per_second'].keys())
    time_stamps_reflex = sorted(results['reflex']['mAP_per_second'].keys())

    mAP_values_complex_only = [results['complex']['mAP_per_second'][t] for t in time_stamps_complex]
    mAP_values_reflex_only = [results['simple']['mAP_per_second'][t] for t in time_stamps_simple]
    mAP_values_reflex = [results['reflex']['mAP_per_second'][t] for t in time_stamps_reflex]

    smoothed_map_per_second_complex = butter_lowpass_filter(mAP_values_complex_only, 3, 30)
    smoothed_map_per_second_reflex_only = butter_lowpass_filter(mAP_values_reflex_only, 5, 30)
    smoothed_map_per_second_reflex = butter_lowpass_filter(mAP_values_reflex, 3, 30)
    ax2.plot(complex_data['fps_time_stamps'],
             smoothed_map_per_second_complex,
             label='Complex-Only System',
             color='#E7004C', linewidth=0.7)
    ax2.plot(reflex_only_data['fps_time_stamps'],
             smoothed_map_per_second_reflex_only,
             label='Reflex-Only System',
             color='#E7004C', linestyle='dashed',
             linewidth=0.7)
    ax2.plot(reflex_data['fps_time_stamps'],
             smoothed_map_per_second_reflex,
             label='Reflex-Enabled System',
             color='#3A8DDE',
             linewidth=0.7)
    ax2.set_title(f'(b) mAP per second Over Time', fontsize=10)
    # Add a grid
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
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
    ax3.set_title(f'(c) Processor Utilization (%) Over Time', fontsize=10)
    ax3.set_ylim(0, 110)
    ax3.set_yticks([0, 20, 40, 60, 80, 100])
    ax3.set_xlim(0, 100)

    # Add a grid
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'experiment_1_results.pdf'), dpi=300)


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

        plt.title('Accumulated Dropped Frames for Different Switching Delays')
        plt.xlabel('Time (s)')
        plt.ylabel('Accumulated Dropped Frames (%)')
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
