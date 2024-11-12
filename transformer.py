from lark import Transformer, Tree
from ast_classes import *


class ReflexLangTransformer(Transformer):
    """Transforms parsed ReflexLang syntax into an AST representation."""

    @staticmethod
    def start(items):
        return items[0]

    def service_definition(self, items):
        """Transforms a service definition."""
        service_name, complex_algorithm, reflex_algorithm, *rest = items
        resources, monitoring_blocks, state_management = [], [], None

        for item in rest:
            if isinstance(item, Resource):
                resources.append(item)
            elif isinstance(item, MonitoringBlock):
                monitoring_blocks.append(item)
            elif isinstance(item, StateManagement):
                state_management = item

        return ServiceDefinition(
            service_name, complex_algorithm, reflex_algorithm, resources, monitoring_blocks, state_management
        )

    def complex_algorithm(self, items):
        """Transforms the complex algorithm component."""
        return Algorithm(name=items[0], body=items[1])

    def reflex_algorithm(self, items):
        """Transforms the reflex algorithm component, considering shadowing."""
        name, body, shadowing = items[0], items[1], items[2] == 'true'
        return Algorithm(name=name, body=body, shadowing=shadowing)

    def algorithm_body(self, items):
        """Extracts the algorithm body, either a code block or reference."""
        return items[0]

    def code_block(self, items):
        """Transforms a code block with optional language specification."""
        language, code_content = (items if len(items) == 2 else (None, items[0]))
        return AlgorithmBody(code=self.flatten_items(code_content), language=language)

    def flatten_items(self, items):
        """Recursively flattens nested items into a single string, handling Trees properly."""
        result = ''
        if isinstance(items, Tree):
            items = items.children  # Access children if it's a Tree
        for item in items:
            if isinstance(item, Tree):
                # Recursively flatten if item is a Tree
                result += self.flatten_items(item.children)
            elif isinstance(item, list):
                # Recursively flatten if item is a list
                result += self.flatten_items(item)
            else:
                result += str(item)
        return result

    def code_reference(self, items):
        """Transforms a code reference with an optional module name."""
        include, module_name = items[0], items[1] if len(items) > 1 else None
        return AlgorithmBody(include=include, module_name=module_name)

    def resource_definition(self, items):
        """Transforms a resource definition with parameters."""
        name, resource_type, data_type, *parameters = items
        param_dict = {param[0]: param[1] for param in parameters}
        return Resource(name, resource_type, data_type, param_dict)

    def monitoring_block(self, items):
        """Transforms a monitoring block, linking inputs to a monitoring function."""
        name = items[0]
        inputs = items[1]
        if isinstance(inputs, Tree):  # Convert Tree to list if necessary
            inputs = [str(child) for child in inputs.children]
        function = items[2]
        return MonitoringBlock(name=name, inputs=inputs, function=function)


    def threshold_monitoring(self, items):
        """Transforms threshold monitoring with activation and deactivation thresholds."""
        activate_threshold = self.extract_value(items[0])
        deactivate_threshold = self.extract_value(items[1])
        return ThresholdMonitoring(activate_threshold=activate_threshold, deactivate_threshold=deactivate_threshold)

    def external_monitoring(self, items):
        """Transforms external monitoring with an algorithm and parameters."""
        name, algorithm_body = items[0], items[1]
        parameters = items[2] if len(items) > 2 else {}
        return ExternalMonitoring(name, algorithm_body, parameters)

    def state_management(self, items):
        """Transforms state management with strategy and optional data definitions."""
        strategy = items[0]
        data_items = items[1].children if len(items) > 1 and isinstance(items[1], Tree) else items[1] if len(items) > 1 else []
        return StateManagement(strategy, data_items)

    def data_item(self, items):
        """Transforms a data item in state management."""
        name, data_type, structure = items[0], items[1], items[2] if len(items) > 2 else None
        return DataItem(name, data_type, structure)

    def parameters(self, items):
        """Transforms parameters into a dictionary format."""
        params = items[0].children if isinstance(items[0], Tree) else items
        return {param.children[0]: self.extract_value(param.children[1]) for param in params}

    def extract_value(self, item):
        """Helper function to extract value from a Tree or return the item directly if it's not a Tree."""
        return item.children[0] if isinstance(item, Tree) and len(item.children) == 1 else item

    def value(self, items):
        """Parses values, ensuring they are properly formatted as numbers or strings."""
        return self.extract_value(items[0])


    # Terminal token transformations
    def SERVICE_NAME(self, token): return token.value
    def ALGORITHM_NAME(self, token): return token.value
    def RESOURCE_NAME(self, token): return token.value
    def BLOCK_NAME(self, token): return token.value
    def MODULE_NAME(self, token): return token.value
    def PARAMETER_NAME(self, token): return token.value
    def VARIABLE_NAME(self, token): return token.value
    def LANGUAGE_NAME(self, token): return token.value
    def DATA_TYPE(self, token): return token.value
    def RESOURCE_TYPE(self, token): return token.value
    def STATE_STRATEGY(self, token): return token.value
    def DATA_STRUCTURE(self, token): return token.value
    def BOOLEAN(self, token): return token.value
    def STRING_LITERAL(self, token): return token.value.strip('"')
    def NUMBER(self, token): return float(token.value)

    # Define monitoring function values explicitly to avoid redundancy
    def monitoring_function(self, items): return items[0]
    def activate_threshold(self, items): return items[0]
    def deactivate_threshold(self, items): return items[0]
