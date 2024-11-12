from lark import Lark
from transformer import *
from reflexlang_grammar import *
from typing import cast

# Instantiate the parser with the transformer
parser = Lark(reflexlang_grammar, parser='lalr', transformer=ReflexLangTransformer())

# Example ReflexLang specification
reflexlang_specification = '''
Service DroneImageProcessing {
    Complex_Algorithm ComplexDetection {
        Include "complex_detection.py" as ComplexDetect
    }
    Reflex_Algorithm SimpleDetection {
        Include "simple_detection.py" as SimpleDetect
        Shadowing: true
    }
    Resource CPU {
        Type: Variable
        DataType: Float
    }
    Monitoring_Block CPUMonitor {
        Inputs: CPU
        Threshold_Monitoring {
            Activate_Threshold: 80%
            Deactivate_Threshold: 60%
        }
    }
    State_Management {
        Strategy: Shared_Data_Store
        Data {
            lastProcessedImageID: Integer
            detectionResults: List
            processingQueue: Queue
        }
    }
}
'''

# Parse and transform
ast = parser.parse(reflexlang_specification)
ast = cast(ServiceDefinition, ast)


def print_ast(service):
    """
    Prints the abstract syntax tree (AST) of a parsed ReflexLang service definition.

    Parameters:
        service (ServiceDefinition): Parsed service definition containing algorithms, resources, and state management.
    """
    print(f"Service Name: {service.name}")
    print(f"Complex Algorithm: {service.complex_algorithm.name}")
    print(f"Reflex Algorithm: {service.reflex_algorithm.name}, Shadowing: {service.reflex_algorithm.shadowing}")

    print("\nResources:")
    for res in service.resources:
        print(f"  - {res.name}: Type={res.type}, DataType={res.data_type}, Parameters={res.parameters}")

    print("\nMonitoring Blocks:")
    for mb in service.monitoring_blocks:
        print(f"  - {mb.name}, Inputs={mb.inputs}")
        if isinstance(mb.function, ThresholdMonitoring):
            print(
                f"    Threshold Monitoring: Activate={mb.function.activate_threshold}, Deactivate={mb.function.deactivate_threshold}")
        elif isinstance(mb.function, ExternalMonitoring):
            print(f"    External Monitoring: {mb.function.name}, Parameters={mb.function.parameters}")

    print(f"\nState Management Strategy: {service.state_management.strategy}")
    print("Shared Data Items:")
    for data_item in service.state_management.data_items:
        print(f"  - {data_item.name}: {data_item.data_type}, Structure={data_item.structure}")


# Call the function to print the AST, verifying successful parsing and transformation
print_ast(ast)

