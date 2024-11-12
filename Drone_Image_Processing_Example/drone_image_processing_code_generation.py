"""
Drone Image Processing Code Generation Script

This script defines a ReflexLang specification for a Drone Image Processing service, parses it to generate an abstract syntax tree (AST), 
and uses the CodeGenerator to produce Python code based on this specification. The generated code is output to the 
'DroneImageProcessingService' directory.

"""

from lark import Lark
from typing import cast
from ast_classes import ServiceDefinition
from reflexlang_grammar import reflexlang_grammar
from transformer import ReflexLangTransformer
from python_code_generation_module import CodeGenerator


# Define the ReflexLang specification for the Drone Image Processing service
reflexlang_specification = '''
Service DroneImageProcessing {
    Complex_Algorithm ComplexDetection {
        Include "complex_detection.py" as ComplexDetect
    }
    Reflex_Algorithm SimpleDetection {
        Include "simple_detection.py" as SimpleDetect
        Shadowing: true
    }
    Resource Queue_length {
        Type: Variable
        DataType: Integer
    }
    Monitoring_Block QueueLengthMonitor {
        Inputs: Queue_length
        Threshold_Monitoring {
            Activate_Threshold: 80
            Deactivate_Threshold: 60
        }
    }
    State_Management {
        Strategy: Shared_Data_Store
        Data {
            detectionResults: List
        }
    }
}
'''

# Parse the specification into an AST
parser = Lark(reflexlang_grammar, parser='lalr', transformer=ReflexLangTransformer())
ast = parser.parse(reflexlang_specification)
ast = cast(ServiceDefinition, ast)
# Instantiate the CodeGenerator with the parsed AST and generate code
code_generator = CodeGenerator(ast, output_dir='DroneImageProcessingService')
code_generator.generate()

print("Code generation complete. Files saved to 'DroneImageProcessingService'.")