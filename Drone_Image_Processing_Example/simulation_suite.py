"""
Simulation Suite for Drone Image Processing with Reflex Pattern

This module simulates a drone image processing system, evaluating the Reflex pattern's effectiveness
in adapting to varying resource conditions. It includes functions to run experiments,
process frames using adaptive algorithms, and measure performance metrics.

Requirements:
    - Python libraries: multiprocessing, os, cv2, time, json, torch, psutil, threading, queue, pycocotools

Usage:
    python simulation_suite.py
"""

import os
import cv2
import time
import json
import torch
import psutil
import threading
from queue import Queue, Full, Empty
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import warnings

from DroneImageProcessingService.main import process, service_instance

# Suppress deprecated function warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Limit PyTorch to a single thread for controlled CPU usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Initialize process to last core to measure CPU load accurately
num_cores = psutil.cpu_count(logical=True)
target_core = num_cores - 1 if num_cores > 1 else 0
p = psutil.Process(os.getpid())
p.cpu_affinity([target_core])

# Set up paths and directories
image_dir = 'COCO/val2017'
annotation_file = 'COCO/annotations/instances_val2017.json'
output_dir = 'simulation_results'
os.makedirs(output_dir, exist_ok=True)

# Load COCO dataset
coco_gt = COCO(annotation_file)
image_ids = coco_gt.getImgIds()
images = coco_gt.loadImgs(image_ids)
image_files = [img['file_name'] for img in images]

# Predefined mapping from model class indices to COCO category IDs (for performance measurement)
class_index_to_category_id = {
    0: 1,    # person
    1: 2,    # bicycle
    2: 3,    # car
    3: 4,    # motorcycle
    4: 5,    # airplane
    5: 6,    # bus
    6: 7,    # train
    7: 8,    # truck
    8: 9,    # boat
    9: 10,   # traffic light
    10: 11,  # fire hydrant
    11: 13,  # stop sign
    12: 14,  # parking meter
    13: 15,  # bench
    14: 16,  # bird
    15: 17,  # cat
    16: 18,  # dog
    17: 19,  # horse
    18: 20,  # sheep
    19: 21,  # cow
    20: 22,  # elephant
    21: 23,  # bear
    22: 24,  # zebra
    23: 25,  # giraffe
    24: 27,  # backpack
    25: 28,  # umbrella
    26: 31,  # handbag
    27: 32,  # tie
    28: 33,  # suitcase
    29: 34,  # frisbee
    30: 35,  # skis
    31: 36,  # snowboard
    32: 37,  # sports ball
    33: 38,  # kite
    34: 39,  # baseball bat
    35: 40,  # baseball glove
    36: 41,  # skateboard
    37: 42,  # surfboard
    38: 43,  # tennis racket
    39: 44,  # bottle
    40: 46,  # wine glass
    41: 47,  # cup
    42: 48,  # fork
    43: 49,  # knife
    44: 50,  # spoon
    45: 51,  # bowl
    46: 52,  # banana
    47: 53,  # apple
    48: 54,  # sandwich
    49: 55,  # orange
    50: 56,  # broccoli
    51: 57,  # carrot
    52: 58,  # hot dog
    53: 59,  # pizza
    54: 60,  # donut
    55: 61,  # cake
    56: 62,  # chair
    57: 63,  # couch
    58: 64,  # potted plant
    59: 65,  # bed
    60: 67,  # dining table
    61: 70,  # toilet
    62: 72,  # tv
    63: 73,  # laptop
    64: 74,  # mouse
    65: 75,  # remote
    66: 76,  # keyboard
    67: 77,  # cell phone
    68: 78,  # microwave
    69: 79,  # oven
    70: 80,  # toaster
    71: 81,  # sink
    72: 82,  # refrigerator
    73: 84,  # book
    74: 85,  # clock
    75: 86,  # vase
    76: 87,  # scissors
    77: 88,  # teddy bear
    78: 89,  # hair drier
    79: 90,  # toothbrush
}


def run_experiment(mode='complex', delay='0', test_duration=150, arrival_rate=10, max_queue_size=100):
    """
       Run an experiment with specified mode, delay, and duration.

       Parameters:
           mode (str): 'complex', 'simple', or 'reflex'
           delay (str): Switching delay in seconds
           test_duration (int): Duration of the experiment in seconds
           arrival_rate (int): Image arrival rate (frames per second)
           max_queue_size (int): Maximum size of the image processing queue

       Returns:
           dict: Collected data and metrics from the experiment
       """
    # Initialize metrics
    images_per_time_stamp = {}
    detections = []
    inference_times = []
    total_frames, dropped_frames, dropped_frames_in_interval = 0, 0, 0
    complex_frames, simple_frames = 0, 0
    cpu_usage_over_time, fps_over_time, queue_sizes = [], [], []
    idle_durations, processing_durations = [], []
    dropped_frames_over_time = []
    dropped_frames_time_stamps = []
    queue_time_stamps = []
    fps_time_stamps = []
    cpu_time_stamps = []
    frames_in_interval = 0
    image_id_list = []
    inference_active_system = []
    image_queue = Queue(maxsize=max_queue_size)
    service_instance.__init__()

    # Configure Mode specific settings
    if mode == 'complex':
        service_instance.switch_to_complex()
    elif mode == 'simple':
        service_instance.switch_to_reflex()

    start_time, end_time = time.time(), time.time() + test_duration

    def monitor_cpu_usage():
        """Monitor and record CPU usage over time."""
        while time.time() < end_time:
            current_time = time.time() - start_time
            cpu_time_stamps.append(current_time)
            cpu_usage_over_time.append(psutil.cpu_percent(interval=None, percpu=True)[7])
            time.sleep(0.05)

    # Function for image arrival
    def image_arrival():
        """Simulate image arrival at a specified rate."""
        arrival_interval = 1.0 / arrival_rate
        image_index = 0
        total_images = len(images)
        nonlocal dropped_frames, dropped_frames_in_interval
        interval_start_time = start_time
        while True:
            arrival_time = time.time()
            if arrival_time >= end_time:
                break
            img_info = images[image_index % total_images]
            try:
                image_queue.put_nowait(img_info)
            except Full:
                # If the queue is full, drop the image
                dropped_frames += 1
                dropped_frames_in_interval += 1
            image_index += 1
            time.sleep(arrival_interval)
            # Check if a second has passed for recording the dropped frames in each interval
            if time.time() - interval_start_time >= 1.0:
                # Record dropped frames for this interval
                dropped_frames_over_time.append(dropped_frames_in_interval)
                dropped_frames_time_stamps.append(time.time() - start_time)
                dropped_frames_in_interval = 0
                interval_start_time = time.time()

    # Function for image processing
    def image_processing():
        """Process images from the queue and measure processing performance."""
        nonlocal total_frames, complex_frames, simple_frames, frames_in_interval
        image_ids_in_interval = []
        interval_start_time = start_time
        while True:
            current_time = time.time()
            if current_time >= end_time:
                break
            try:
                wait_start = time.time()
                img_info = image_queue.get(timeout=1)
                idle_time = time.time() - wait_start
                idle_durations.append(idle_time)
            except Empty:
                continue

            process_start = time.time()
            image_id = img_info['id']
            image_file = img_info['file_name']
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image {image_path}")
                image_queue.task_done()
                continue
            # Collect image IDs and convert to RGB
            image_id_list.append(image_id)
            image_ids_in_interval.append(image_id)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Simulate CPU load fluctuation
            if mode == 'reflex':
                service_instance.resource_manager.set_queue_length(image_queue.qsize())
            elif mode == 'complex':
                service_instance.resource_manager.set_queue_length(0)  # Low simulated load to trigger complex algorithm
            else:  # If mode Simple
                service_instance.resource_manager.set_queue_length(int(0.9*max_queue_size))  # High simulated load to trigger reflex algorithm

            # Run detection
            inference_start = time.time()

            # Count frames processed by each algorithm
            if service_instance.use_reflex:
                simple_frames += 1
                inference_active_system.append('reflex')
            else:
                complex_frames += 1
                inference_active_system.append('complex')

            # Include delay if a switch occurred to simulate a switching delay
            if len(inference_active_system) > 2:
                if inference_active_system[-1] is not inference_active_system[-2]:
                    time.sleep(float(delay))

            results = process(image_rgb)
            inference_time = time.time() - inference_start

            queue_sizes.append(image_queue.qsize())
            queue_time_stamps.append(time.time() - start_time)

            inference_times.append(inference_time)
            total_frames += 1

            frames_in_interval += 1
            if current_time - interval_start_time >= 1.0:
                fps = frames_in_interval / (current_time - interval_start_time)
                fps_over_time.append(fps)
                time_stamp = current_time - start_time
                fps_time_stamps.append(time_stamp)
                images_per_time_stamp[int(time_stamp)] = image_ids_in_interval.copy()
                interval_start_time = current_time
                frames_in_interval = 0
                image_ids_in_interval = []

            # Extract detections
            preds = results.xywh[0]  # [x_center, y_center, w, h, conf, class]
            for pred in preds:
                x_center, y_center, w, h, conf, cls = pred.tolist()
                class_index = int(cls)
                if class_index not in class_index_to_category_id:
                    continue
                category_id = class_index_to_category_id[class_index]
                x_min = x_center - (w / 2)
                y_min = y_center - (h / 2)
                detection = {
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x_min, y_min, w, h],
                    'score': conf
                }
                detections.append(detection)

            processing_duration = time.time() - process_start
            processing_durations.append(processing_duration)
            image_queue.task_done()

    # Start the arrival, processing and cpu monitoring threads
    arrival_thread = threading.Thread(target=image_arrival)
    processing_thread = threading.Thread(target=image_processing)
    cpu_monitor_thread = threading.Thread(target=monitor_cpu_usage)

    cpu_monitor_thread.daemon = True

    arrival_thread.start()
    processing_thread.start()
    cpu_monitor_thread.start()

    # Wait for the threads to finish
    arrival_thread.join()
    processing_thread.join()
    cpu_monitor_thread.join()

    # Save detections to JSON
    detections_file = os.path.join(output_dir, f'detections_{mode}.json')
    with open(detections_file, 'w') as f:
        json.dump(detections, f)

    # Return collected data dictionary
    return {
        'detections_file': detections_file,
        'image_id_list': image_id_list,
        'total_frames': total_frames,
        'dropped_frames': dropped_frames,
        'queue_time_stamps': queue_time_stamps,
        'queue_sizes': queue_sizes,
        'fps_time_stamps': fps_time_stamps,
        'fps_over_time': fps_over_time,
        'cpu_usage_over_time': cpu_usage_over_time,
        'cpu_time_stamps': cpu_time_stamps,
        'inference_active_system': inference_active_system,
        'dropped_frames_over_time': dropped_frames_over_time,
        'dropped_frames_time_stamps': dropped_frames_time_stamps,
        'images_per_time_stamp': images_per_time_stamp
    }


def evaluate_detections(detections_file, image_id_list):
    """Evaluate detection results using COCO API and calculate mAP."""
    coco_dt = coco_gt.loadRes(detections_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    processed_image_ids = image_id_list
    coco_eval.params.imgIds = processed_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # map -> AP@[IoU=0.50:0.95 | area=all | maxDets=100]


def evaluate_detections_per_time_stamp(detections_file, images_per_time_stamp):
    """Evaluate detection results per time stamp using COCO API and calculate mAP per second."""
    with open(detections_file, 'r') as f:
        all_detections = json.load(f)

    detections_by_image = {}
    for det in all_detections:
        image_id = det['image_id']
        if image_id not in detections_by_image:
            detections_by_image[image_id] = []
        detections_by_image[image_id].append(det)

    mAP_per_second = {}
    for time_stamp in sorted(images_per_time_stamp.keys()):
        image_ids = images_per_time_stamp[time_stamp]
        detections = []
        for image_id in image_ids:
            if image_id in detections_by_image:
                detections.extend(detections_by_image[image_id])
        if not detections:
            mAP_per_second[time_stamp] = 0.0
            continue
        detections_temp_file = os.path.join(output_dir, f'detections_{time_stamp}.json')
        with open(detections_temp_file, 'w') as f:
            json.dump(detections, f)
        coco_dt = coco_gt.loadRes(detections_temp_file)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP_per_second[time_stamp] = coco_eval.stats[0]
        os.remove(detections_temp_file)
    return mAP_per_second



def experiment_1():
    """Run Experiment 1: Evaluate Reflex vs. Non-Reflex Systems."""
    results = {}
    for mode in ['complex', 'simple', 'reflex']:
        print(f"\nRunning experiment: {mode.upper()}")
        experiment_data = run_experiment(mode=mode, delay='0', test_duration=150, arrival_rate=4, max_queue_size=50)
        mAP = evaluate_detections(experiment_data['detections_file'], experiment_data['image_id_list'])
        mAP_per_second = evaluate_detections_per_time_stamp(experiment_data['detections_file'],
                                                            experiment_data['images_per_time_stamp'])
        results[mode] = {
            'mAP': mAP,
            'mAP_per_second': mAP_per_second,
            'total_frames': experiment_data['total_frames'],
            'dropped_frames': experiment_data['dropped_frames'],
            'inference_active_system': experiment_data['inference_active_system'],
            'queue_time_stamps': experiment_data['queue_time_stamps'],
            'queue_sizes': experiment_data['queue_sizes'],
            'fps_time_stamps': experiment_data['fps_time_stamps'],
            'fps_over_time': experiment_data['fps_over_time'],
            'cpu_usage_over_time': experiment_data['cpu_usage_over_time'],
            'cpu_time_stamps': experiment_data['cpu_time_stamps'],
            'dropped_frames_over_time': experiment_data['dropped_frames_over_time'],
            'dropped_frames_time_stamps': experiment_data['dropped_frames_time_stamps']
        }
        results_file = os.path.join(output_dir, 'experiment_1_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Experiment 1 results saved to {results_file}")


def experiment_2():
    """Run Experiment 2: Analyze Impact of Switching Delays."""
    results = {}
    arrival_rate = 5
    for delay in ['0', '2', '5', '10', '15', '20', '25', '30']:
        print(f"\nRunning second experiment. Delay: {delay} s")
        experiment_data = run_experiment(mode='reflex', delay=delay,test_duration=500,
                                         arrival_rate=arrival_rate, max_queue_size=100)
        results[delay] = {
            'queue_sizes': experiment_data['queue_sizes'],
            'queue_time_stamps': experiment_data['queue_time_stamps'],
            'inference_active_system': experiment_data['inference_active_system'],
            'dropped_frames_over_time': experiment_data['dropped_frames_over_time'],
            'dropped_frames_time_stamps': experiment_data['dropped_frames_time_stamps']
        }

        results_file = os.path.join(output_dir, f'experiment_2_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Experiment 2 results saved to {results_file}")


def main():
    """Run both experiments."""
    experiment_1()
    experiment_2()


if __name__ == '__main__':
    main()
