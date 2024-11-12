# Drone Image Processing Example 

This repository contains the Drone Image Processing Example from the associated research paper. 
It contains the example Drone Image Processing Specification from which the code in `DroneImageProcessingService/`is generated. 
Furthermore, it contains a simulation script for simulating this example and a plotting script. 


--- 
## Repository Structure
The repository is organized as follows:

```
└── Drone_image_processing_example/
    ├──simulation_suite.py
    ├──simulation_suite_plotting.py
    ├──simulation_results/
        ├── experiment_2_results_rate_5_fps.json
        ├── experiment_1_results.json
    ├── DroneImageProcessingService
        ├── __init.py__
        ├── abstract_algorithm.py
        ├── algorithms.py
        ├── complex_detection.py
        ├── main.py
        ├── monitoring.py
        ├── resources.py
        ├── simple_detection.py
        ├── state_management.py
```
- `drone_image_processing_code_generation.py` Code generation Module for the Drone Image Processing Example including the Specification and Call to the ReflexLang 
- `simulation_suite.py` Simulation script for the simulations from the associated research paper
- `simulation_suite_plotting.py` Plotting script 
- `simulation_results/` Folder containing the simulation results from experiment 1 and 2
- `DroneImageProcessingService/` Folder containing the generated code from the `drone_image_processing_code_generation.py`  module and the external files `complex_detection.py` and `simple_detection`

--- 
## Prerequisites
### Python Libraries 
Ensure you have the following Python libraries installed for the Simulation:

```bash 
pip install psutil torch pycocotools scipy matplotlib
``` 
--- 
### COCO Dataset
To run the simulation, download the COCO dataset:

1. **Download** the COCO val2017 images and annotations from [COCO Dataset Download](https://cocodataset.org/#download) 
2. **Place the dataset** in the following structure:
```
└── Drone_Image_Processing_Example/
    ├──COCO/
        ├──annotations/
        ├──val2017/
```
---
## Running the Code 

### Code Generation
The `drone_image_processing_code_generation.py` file generates code in the `DroneImageProcessingService/` folder based on the ReflexLang specification. You can modify the specification to customize the generated code for different configurations.

### Simulation Suite

**Running the Simulation**: Use the following command to simulate the example:

```bash
python simulation_suite.py
```
This will generate results that analyze the Reflex pattern’s effectiveness under different load conditions.

**Plotting Results**: After running the simulation, visualize the results with:

```bash
python simulation_suite_plotting.py
```
This script produces plots showing error rate, FPS, CPU utilization, and queue fill levels.