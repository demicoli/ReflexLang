import unittest
from lark import Lark
from ast_classes import *
from transformer import ReflexLangTransformer
from reflexlang_grammar import *
from typing import cast


class TestReflexLangParser(unittest.TestCase):
    """Unit tests for the ReflexLang parser and its AST transformation."""

    @classmethod
    def setUpClass(cls):
        """Initializes the parser with the ReflexLang grammar and transformer."""
        cls.parser = Lark(reflexlang_grammar, parser='lalr', transformer=ReflexLangTransformer())

    def test_minimal_service(self):
        """Tests parsing a minimal service definition with basic components."""
        reflexlang_spec = '''
        Service MinimalService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py" as Complex
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py" as Reflex
                Shadowing: false
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
                Strategy: Stateless
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        self.assertEqual(ast.name, "MinimalService")
        self.assertEqual(ast.complex_algorithm.name, "ComplexAlgo")
        self.assertEqual(ast.reflex_algorithm.name, "ReflexAlgo")
        self.assertFalse(ast.reflex_algorithm.shadowing)
        self.assertEqual(len(ast.resources), 1)
        self.assertEqual(ast.resources[0].name, "CPU")
        self.assertEqual(ast.state_management.strategy, "Stateless")

    def test_shared_data_store(self):
        """Tests parsing a service with shared data store and multiple data items."""
        reflexlang_spec = '''
        Service SharedDataService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py"
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py"
                Shadowing: true
            }
            Resource Network {
                Type: Variable
                DataType: Float
            }
            Monitoring_Block NetworkMonitor {
                Inputs: Network
                Threshold_Monitoring {
                    Activate_Threshold: 90%
                    Deactivate_Threshold: 70%
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
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        self.assertEqual(ast.name, "SharedDataService")
        self.assertTrue(ast.reflex_algorithm.shadowing)
        self.assertEqual(ast.state_management.strategy, "Shared_Data_Store")
        self.assertEqual(len(ast.state_management.data_items), 2)
        self.assertEqual(ast.state_management.data_items[0].name, "sessionData")
        self.assertEqual(ast.state_management.data_items[0].data_type, "Dictionary")

    def test_checkpointing_state_management(self):
        """Tests parsing a service with checkpointing state management."""
        reflexlang_spec = '''
        Service CheckpointService {
            Complex_Algorithm ComplexAlgo {
                Code {
                    def complex_task():
                        pass
                }
            }
            Reflex_Algorithm ReflexAlgo {
                Code {
                    def reflex_task():
                        pass
                }
                Shadowing: false
            }
            Resource Memory {
                Type: Variable
                DataType: Float
            }
            Monitoring_Block MemoryMonitor {
                Inputs: Memory
                Threshold_Monitoring {
                    Activate_Threshold: 85%
                    Deactivate_Threshold: 65%
                }
            }
            State_Management {
                Strategy: Checkpointing
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        self.assertEqual(ast.state_management.strategy, "Checkpointing")

    def test_external_monitoring_algorithm(self):
        """Tests parsing a service with an external monitoring algorithm."""
        reflexlang_spec = '''
        Service ExternalMonitoringService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py"
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py"
                Shadowing: true
            }
            Resource SensorData {
                Type: Buffer
                DataType: SensorReading
            }
            Monitoring_Block AnomalyDetector {
                Inputs: SensorData
                Monitoring_Algorithm AnomalyDetectionAlgorithm {
                    Include "anomaly_detection.py"
                    Parameters {
                        Sensitivity: 0.95
                    }
                }
            }
            State_Management {
                Strategy: Stateless
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        self.assertEqual(len(ast.monitoring_blocks), 1)
        mb = ast.monitoring_blocks[0]
        self.assertEqual(mb.name, "AnomalyDetector")
        self.assertEqual(mb.inputs, ["SensorData"])
        self.assertIsInstance(mb.function, ExternalMonitoring)
        self.assertEqual(mb.function.name, "AnomalyDetectionAlgorithm")
        self.assertEqual(mb.function.parameters["Sensitivity"], 0.95)

    def test_nested_code_blocks(self):
        """Tests parsing a service with nested code blocks in algorithms."""
        reflexlang_spec = '''
        Service NestedCodeService {
            Complex_Algorithm ComplexAlgo {
                Code {
                    def complex_task():
                        if True:
                            print("Nested code")
                }
            }
            Reflex_Algorithm ReflexAlgo {
                Code {
                    def reflex_task():
                        while False:
                            pass
                }
                Shadowing: false
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
                Strategy: Stateless
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        complex_code = ast.complex_algorithm.body.code
        self.assertIn('def complex_task():', complex_code)
        self.assertIn('print("Nested code")', complex_code)

    def test_multiple_resources_and_monitoring_blocks(self):
        """Tests parsing a service with multiple resources and monitoring blocks."""
        reflexlang_spec = '''
        Service MultiResourceService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py"
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py"
                Shadowing: true
            }
            Resource CPU {
                Type: Variable
                DataType: Float
            }
            Resource Memory {
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
            Monitoring_Block MemoryMonitor {
                Inputs: Memory
                Threshold_Monitoring {
                    Activate_Threshold: 85%
                    Deactivate_Threshold: 65%
                }
            }
            State_Management {
                Strategy: Stateless
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        self.assertEqual(len(ast.resources), 2)
        self.assertEqual(len(ast.monitoring_blocks), 2)
        resource_names = [res.name for res in ast.resources]
        self.assertIn("CPU", resource_names)
        self.assertIn("Memory", resource_names)
        monitoring_block_names = [mb.name for mb in ast.monitoring_blocks]
        self.assertIn("CPUMonitor", monitoring_block_names)
        self.assertIn("MemoryMonitor", monitoring_block_names)

    def test_invalid_syntax(self):
        """Tests parsing an invalid service definition to check error handling."""
        reflexlang_spec = '''
        Service InvalidService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py"
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py"
                Shadowing true  # Missing colon
            }
            State_Management {
                Strategy: Stateless
            }
        }
        '''
        with self.assertRaises(Exception):
            self.parser.parse(reflexlang_spec)

    def test_threshold_monitoring_values(self):
        """Tests that activate and deactivate thresholds are parsed as numeric values."""
        reflexlang_spec = '''
        Service ThresholdService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py"
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py"
                Shadowing: false
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
                Strategy: Stateless
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)

        # Verify that the thresholds are parsed as floats
        cpumonitor = ast.monitoring_blocks[0]
        self.assertEqual(cpumonitor.function.activate_threshold, 80.0)
        self.assertEqual(cpumonitor.function.deactivate_threshold, 60.0)
        self.assertIsInstance(cpumonitor.function.activate_threshold, float)
        self.assertIsInstance(cpumonitor.function.deactivate_threshold, float)

    def test_empty_service(self):
        """Tests parsing an empty service definition without resources or monitoring blocks."""
        reflexlang_spec = '''
        Service EmptyService {
            Complex_Algorithm ComplexAlgo {
                Include "complex_algo.py"
            }
            Reflex_Algorithm ReflexAlgo {
                Include "reflex_algo.py"
                Shadowing: false
            }
            State_Management {
                Strategy: Stateless
            }
        }
        '''
        ast = self.parser.parse(reflexlang_spec)
        ast = cast(ServiceDefinition, ast)
        self.assertEqual(len(ast.resources), 0)
        self.assertEqual(len(ast.monitoring_blocks), 0)


if __name__ == '__main__':
    unittest.main()
