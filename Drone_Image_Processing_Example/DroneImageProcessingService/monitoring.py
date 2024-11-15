# Monitoring Module

class QueueLengthMonitor:
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.activate_threshold = 40.0
        self.deactivate_threshold = 30.0
        self.active = False

    def check(self):
        # Replace 'resource_value' with actual resource value
        resource_value = self.resource_manager.queue_length
        if not self.active and resource_value > self.activate_threshold:
            self.active = True
            return True
        elif self.active and resource_value < self.deactivate_threshold:
            self.active = False
            return False
        return self.active
        