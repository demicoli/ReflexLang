# ReflexLang: A Design Specification Language for Service Code Generation
This repository contains **ReflexLang**, a design specification language framework for defining and generating service-oriented code for the Reflex Pattern. ReflexLang allows users to define complex and reflex algorithms, resources, monitoring strategies, and state management configurations in a simple DSL. This repository also includes an example application with experiments used in our associated paper, showcasing ReflexLang's capabilities in a **drone image processing example**.

--- 

## Repository Structure
The repository is organized as follows:

```
├── reflexlang_grammar.py    
├── transformer.py    
├── ast_classes.py                           
├── python_code_generation_module.py       
├── test_reflexlang_parser.py             
├── parsing_example.py                     
├── drone_image_processing_example/
├── README.md
└── .gitignore
```
- `reflexlang_grammar.py` Grammar definition for ReflexLang, defining services, algorithms, resources, monitoring blocks, and state management.
- `transformer.py` Contains the `ReflexLangTransformer` class to convert parsed ReflexLang syntax into an abstract syntax tree (AST).
- `ast_classes.py` Defines the data structures for each ReflexLang component, such as `ServiceDefinition`, `Algorithm`, and `Resource`.
- `python_code_generation_module.py` Code generation engine that produces Python modules from the parsed AST.
- `test_reflexlang_parser.py` Unit tests for verifying parsing and transformation of ReflexLang definitions.
- `parsing_example.py` An example ReflexLang service configuration to demonstrate parsing and AST generation.
- `drone_image_processing_example/` Contains the example and experiments from the paper, including generated code for a drone image processing service.
  
--- 
## Getting Started
### Prerequisites
Ensure you have the following Python libraries installed:

```bash 
pip install lark-parser
```

### Installation and Setup

1. **Clone the repository**:
```
git clone <repository-url>
cd <repository-folder>
```

2. **Run the Example**: You can test the parsing and AST generation using the example provided in parsing_example.py.
```
python parsing_example.py
```
This will output the abstract syntax tree (AST) for the DroneImageProcessing service configuration.

3. **Run Tests**: 
Use the test_reflexlang_parser.py file to verify that the parser and transformer work correctly.

```
python -m unittest test_reflexlang_parser.py
```
--- 
## Using ReflexLang

ReflexLang is a domain-specific language (DSL) designed to facilitate the specification and implementation of services employing the **Reflex Pattern**. 
It provides constructs for defining services, algorithms, resources, monitoring blocks, and state management strategies. 
By abstracting the complexities involved in implementing the reflex pattern, ReflexLang enables developers to focus on the core functionality of their applications.

### Language Components
ReflexLang specifications consist of the following key components:

- Service Definition
- Algorithm Definitions (Complex and Reflex Algorithms)
- Resource Definitions
- Monitoring Blocks
- State Management

Below, we provide detailed explanations and examples for each component.

--- 
1. **Service Definition**:

A service in ReflexLang is defined using the `Service` keyword, followed by the service name and a block containing the service components.

**Syntax**:
```ebnf
Service ServiceName {
    Complex_Algorithm ...
    Reflex_Algorithm ...
    Resource ...
    Monitoring_Block ...
    State_Management ...
}
```
**Example**:
```ebnf
Service MyService {
    Complex_Algorithm ComplexAlgo {
        Include "complex_algo.py" as Complex
    }
    Reflex_Algorithm ReflexAlgo {
        Include "reflex_algo.py" as Reflex
        Shadowing: true
    }
    // Other components...
}
```
--- 
2. **Algorithm Definitions**:

Algorithms are defined as either `Complex_Algorithm` or `Reflex_Algorithm`. Each algorithm can include code either inline or by reference to external files.

**Syntax**:
```ebnf
Complex_Algorithm AlgorithmName {
    Algorithm_Body
}

Reflex_Algorithm AlgorithmName {
    Algorithm_Body
    Shadowing: true | false
}

Algorithm_Body ::= Code_Block | Code_Reference

Code_Block ::= Code {
    Language: LanguageName
    // Code content here
}

Code_Reference ::= Include "path/to/code.py" as ModuleName
```
**Examples**:
- **Complex Algorithm with inline Code**:
```
Complex_Algorithm ImageProcessing {
    Code {
        Language: Python
        def process_image(image):
            # Image processing logic here
            return processed_image
    }
}
```
- **Reflex Algorithm with External Code and Shadowing**:
```
Reflex_Algorithm BasicProcessing {
    Include "basic_processing.py" as BasicProc
    Shadowing: true
}
```

--- 

3. **Resource Definitions**:

Resources represent system resources or data elements used by the service. They are defined using the `Resource` keyword.
**Syntax**:
```ebnf
Resource ResourceName {
    Type: Variable | Buffer | List | Queue | Custom
    DataType: TypeName
    [ Additional Parameters ]
}
```
Resource types: `Variable`, `Buffer`, `List`, `Queue`, `Custom`

**Examples**:
- **Variable**:  
```ebnf
Resource CPU {
    Type: Variable
    DataType: Float
}
```
- **Buffer**:  
```
Resource SensorData {
    Type: Buffer
    DataType: SensorReading
    Size: 1024
}
```

---

4. **Monitoring Blocks**:

Monitoring blocks define how the service monitors resources to decide when to switch between algorithms.

**Syntax**:
```ebnf
Monitoring_Block BlockName {
    Inputs: Resource1, Resource2, ...
    Monitoring_Function
}
```
**Monitoring Functions**:
- **Threshold Monitoring**:
```ebnf
Threshold_Monitoring {
    Activate_Threshold: Value
    Deactivate_Threshold: Value
}
```
- **External Monitoring Algorithm**:
```ebnf
Monitoring_Algorithm AlgorithmName {
    Include "path/to/algorithm.py"
    Parameters {
        ParameterName: Value
        // Additional parameters
    }
}
```
**Examples**:
- **Threshold Monitoring with Hysteresis**:  
```ebnf
Monitoring_Block CPUMonitor {
    Inputs: CPU
    Threshold_Monitoring {
        Activate_Threshold: 80%
        Deactivate_Threshold: 60%
    }
}
```
- **External Monitoring Algorithm**:  
```
Monitoring_Block AnomalyDetector {
    Inputs: SensorData
    Monitoring_Algorithm AnomalyDetectionAlgorithm {
        Include "anomaly_detection.py"
        Parameters {
            Sensitivity: 0.9
        }
    }
}
```
---
5. **State Management**:

State management defines how the service manages its internal state, supporting strategies like stateless, checkpointing, or shared data store.
**Syntax**:
```ebnf
State_Management {
    Strategy: StrategyType
    Data {
        VariableName: DataType [Structure: DataStructure]
        // Additional data items
    }
}
```
**Strategies**: `Stateless`, `Checkpointing`, `Shared_Data_Store`

**Examples**:
```ebnf
State_Management {
    Strategy: Shared_Data_Store
    Data {
        sessionData: Dictionary
        userData: List
    }
}
```
---
## Complete Service Example
Combining all the components, here is a complete example of a service defined using ReflexLang.
```ebnf
Service DroneControl {
    Complex_Algorithm AdvancedNavigation {
        Include "advanced_navigation.py" as AdvNav
    }
    Reflex_Algorithm SimpleAvoidance {
        Include "simple_avoidance.py" as ObAvoid
        Shadowing: false
    }
    Resource CPU {
        Type: Variable
        DataType: Float
    }
    Resource SensorData {
        Type: Buffer
        DataType: SensorReading
    }
    Monitoring_Block CPUMonitor {
        Inputs: CPU
        Threshold_Monitoring {
            Activate_Threshold: 80%
            Deactivate_Threshold: 60%
        }
    }
    Monitoring_Block AnomalyDetector {
        Inputs: SensorData
        Monitoring_Algorithm AnomalyDetectionAlgorithm {
            Include "anomaly_detection.py"
            Parameters {
                Sensitivity: 0.9
            }
        }
    }
    State_Management {
        Strategy: Shared_Data_Store
        Data {
            droneState: DroneState Structure: Custom
            navigationGoals: List Structure: List
        }
    }
}
```

--- 
## Using ReflexLang
To define a new service with ReflexLang:

1. **Create a Service Definition**: Start by defining the service name, algorithms, resources, monitoring blocks, and state management strategy.

```
Example ReflexLang Specification:

Service MyService {
    Complex_Algorithm ComplexAlgo {
        Include "complex_algo.py" as Complex
    }
    Reflex_Algorithm ReflexAlgo {
        Include "reflex_algo.py" as Reflex
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
            sessionData: Dictionary
            userData: List
        }
    }
}
```
2. **Parse the Specification**: Use the parsing_example.py script to parse this specification into an AST, which can then be used for code generation.

3. **Generate Code**: Run python_code_generation_module.py with the AST generated from your service specification to output organized Python modules.

---
## Documentation
Each module in this repository contains detailed docstrings explaining the purpose of classes and functions. Key modules include:

- **reflexlang_grammar.py**: Defines the grammar rules for ReflexLang.
- **transformer.py**: Handles the transformation of parsed data into AST components.
- **ast_classes.py**: Contains class definitions for each ReflexLang entity, such as ServiceDefinition, Algorithm, and Resource.
- **python_code_generation_module.py**: Generates organized Python modules from AST based on ReflexLang specifications.

--- 
## Python Code Generation Module

The `CodeGenerator` class dynamically generates Python modules based on a ReflexLang service definition. These modules include:

- **Algorithms**: Contains both the complex and reflex algorithms.
- **Resources**: Manages system resources with data initialization and update functions.
- **Monitoring**: Defines monitoring functions, either threshold-based or external.
- **State Management**: Implements the state management strategy, supporting stateless, checkpointing, and shared data store strategies.
- **Main Module**: Integrates all components, providing a service interface for data processing.

To use `CodeGenerator`, initialize it with a parsed AST and call `generate()`:

```python
generator = CodeGenerator(parsed_ast)
generator.generate()
```
Generated files will be stored in the GeneratedService directory (configurable) and will include an __init__.py file, marking it as a package.