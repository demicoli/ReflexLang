# AST node classes

class ServiceDefinition:
    """
    Represents a service definition in ReflexLang, encapsulating
    algorithms, resources, monitoring, and state management.
    """

    def __init__(self, name, complex_algorithm, reflex_algorithm, resources, monitoring_blocks, state_management):
        """
        Initializes a ServiceDefinition.

        Parameters:
            name (str): Name of the service.
            complex_algorithm (Algorithm): The complex algorithm associated with the service.
            reflex_algorithm (Algorithm): The reflex (fallback) algorithm associated with the service.
            resources (list of Resource): List of resources required by the service.
            monitoring_blocks (list of MonitoringBlock): List of monitoring blocks for resource tracking.
            state_management (StateManagement): The strategy for managing service state.
        """
        self.name = name
        self.complex_algorithm = complex_algorithm
        self.reflex_algorithm = reflex_algorithm
        self.resources = resources
        self.monitoring_blocks = monitoring_blocks
        self.state_management = state_management


class Algorithm:
    """Represents an algorithm in the service, either complex or reflex."""

    def __init__(self, name, body, shadowing=False):
        """
        Initializes an Algorithm.

        Parameters:
            name (str): Name of the algorithm.
            body (AlgorithmBody): The body containing code or an external reference for the algorithm.
            shadowing (bool): Indicates if this algorithm shadows another (only applicable for reflex algorithms).
        """
        self.name = name
        self.body = body
        self.shadowing = shadowing


class AlgorithmBody:
    """Represents the body of an algorithm, containing either code or an external reference."""

    def __init__(self, code=None, include=None, language=None, module_name=None):
        """
        Initializes an AlgorithmBody.

        Parameters:
            code (str, optional): Inline code for the algorithm.
            include (str, optional): External file to include as the algorithm's code.
            language (str, optional): Programming language of the code (if specified).
            module_name (str, optional): Optional alias for the included module.
        """
        self.code = code
        self.include = include
        self.language = language
        self.module_name = module_name


class Resource:
    """Represents a resource required by the service, with additional parameters."""

    def __init__(self, name, resource_type, data_type, parameters):
        """
        Initializes a Resource.

        Parameters:
            name (str): Name of the resource.
            resource_type (str): Type of the resource (e.g., Variable, Buffer).
            data_type (str): Data type associated with the resource.
            parameters (dict): Dictionary of additional parameters for the resource.
        """
        self.name = name
        self.type = resource_type
        self.data_type = data_type
        self.parameters = parameters


class MonitoringBlock:
    """Represents a monitoring block, which tracks a resource based on a monitoring function."""

    def __init__(self, name, inputs, function):
        """
        Initializes a MonitoringBlock.

        Parameters:
            name (str): Name of the monitoring block.
            inputs (list of str): List of resource names to monitor.
            function (ThresholdMonitoring or ExternalMonitoring): Monitoring function used in the block.
        """
        self.name = name
        self.inputs = inputs
        self.function = function


class ThresholdMonitoring:
    """Defines a threshold-based monitoring function with activation and deactivation thresholds."""

    def __init__(self, activate_threshold, deactivate_threshold):
        """
        Initializes a ThresholdMonitoring function.

        Parameters:
            activate_threshold (float): The threshold to activate monitoring.
            deactivate_threshold (float): The threshold to deactivate monitoring.
        """
        self.activate_threshold = activate_threshold
        self.deactivate_threshold = deactivate_threshold


class ExternalMonitoring:
    """Defines an external monitoring function that uses an algorithm and additional parameters."""

    def __init__(self, name, algorithm_body, parameters):
        """
        Initializes an ExternalMonitoring function.

        Parameters:
            name (str): Name of the monitoring algorithm.
            algorithm_body (AlgorithmBody): The body of the monitoring algorithm.
            parameters (dict): Dictionary of parameters for the monitoring algorithm.
        """
        self.name = name
        self.algorithm_body = algorithm_body
        self.parameters = parameters


class StateManagement:
    """Represents the state management strategy of the service, including data items to manage."""

    def __init__(self, strategy, data_items):
        """
        Initializes a StateManagement strategy.

        Parameters:
            strategy (str): The state management strategy (e.g., Stateless, Shared_Data_Store).
            data_items (list of DataItem): List of data items managed by the strategy.
        """
        self.strategy = strategy
        self.data_items = data_items


class DataItem:
    """Represents a data item managed within a state management strategy."""

    def __init__(self, name, data_type, structure=None):
        """
        Initializes a DataItem.

        Parameters:
            name (str): Name of the data item.
            data_type (str): Data type of the item (e.g., Integer, List).
            structure (str, optional): Optional structure type (e.g., List, Dictionary).
        """
        self.name = name
        self.data_type = data_type
        self.structure = structure
