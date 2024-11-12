import torch
from .abstract_algorithm import Algorithm


class SimpleDetect(Algorithm):
    def __init__(self, state):
        # Load the YOLOv5 Large model and trust the repository
        self.model = None
        self.state = state

    def initialize(self):
        # Load the YOLOv5 Nano model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, trust_repo=True)

    def deinitialize(self):
        self.model = None

    def process(self, input_data):
        # Perform object detection
        results = self.model(input_data)
        return results
