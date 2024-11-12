# Main Module
import threading
import time
from .algorithms import *
from .resources import ResourceManager
from .monitoring import *
from .state_management import *


class Service:
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.state = SharedDataStore()
        self.use_reflex = False
        self.lock = threading.Lock()
        self.active_algorithm = None
        self.result_queue = []  # To store results when shadowing
        self.init_algorithms()
        self.init_monitoring()

    def init_algorithms(self):
        self.complexdetection = ComplexDetection(self.state)
        self.simpledetection = SimpleDetection(self.state)
        self.complexdetection.initialize()
        self.active_algorithm = self.complexdetection
        self.simpledetection.initialize()

    def init_monitoring(self):
        self.queuelengthmonitor = QueueLengthMonitor(self.resource_manager)
        threading.Thread(target=self.monitor_queuelengthmonitor, daemon=True).start()

    def monitor_queuelengthmonitor(self):
        while True:
            self.resource_manager.update_resources()
            if self.queuelengthmonitor.check():
                self.switch_to_reflex()
            else:
                self.switch_to_complex()
            time.sleep(1)

    def process(self, input_data):
        with self.lock:
            return self.active_algorithm.process(input_data)

    def switch_to_reflex(self):
        with self.lock:
            if not self.use_reflex:
                print('Switching to Reflex Algorithm')
                # Perform state exchange
                self.active_algorithm = self.simpledetection
                self.use_reflex = True

    def switch_to_complex(self):
        with self.lock:
            if self.use_reflex:
                print('Switching to Complex Algorithm')
                # Perform state exchange
                self.active_algorithm = self.complexdetection
                self.use_reflex = False

# Create an instance of the service
service_instance = Service()


# Interface function
def process(input_data, resource_values=None):
    if resource_values:
        # Set resource values
        for key, value in resource_values.items():
            setattr(service_instance.resource_manager, key, value)
    return service_instance.process(input_data)
