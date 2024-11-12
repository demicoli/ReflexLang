reflexlang_grammar = r"""
start: service_definition

service_definition: "Service" SERVICE_NAME "{" complex_algorithm reflex_algorithm resource_definition* monitoring_block* state_management "}"

complex_algorithm: "Complex_Algorithm" ALGORITHM_NAME "{" algorithm_body "}"
reflex_algorithm: "Reflex_Algorithm" ALGORITHM_NAME "{" algorithm_body "Shadowing" ":" BOOLEAN "}"

algorithm_body: code_block | code_reference
code_block: "Code" "{" code_content "}"
code_reference: "Include" STRING_LITERAL ("as" MODULE_NAME)?

code_content: ( /[^{}]+/ | "{" code_content "}" )*

resource_definition: "Resource" RESOURCE_NAME "{" "Type" ":" RESOURCE_TYPE "DataType" ":" DATA_TYPE resource_parameters* "}"
resource_parameters: PARAMETER_NAME ":" value

monitoring_block: "Monitoring_Block" BLOCK_NAME "{" "Inputs" ":" resource_list monitoring_function "}"
monitoring_function: threshold_monitoring | external_monitoring
threshold_monitoring: "Threshold_Monitoring" "{" "Activate_Threshold" ":" value "Deactivate_Threshold" ":" value "}"
external_monitoring: "Monitoring_Algorithm" ALGORITHM_NAME "{" algorithm_body parameters? "}"
parameters: "Parameters" "{" parameter_list "}"
parameter_list: parameter*
parameter: PARAMETER_NAME ":" value

state_management: "State_Management" "{" "Strategy" ":" STATE_STRATEGY data_definition? "}"
data_definition: "Data" "{" data_item* "}"
data_item: VARIABLE_NAME ":" DATA_TYPE ("Structure" ":" DATA_STRUCTURE)?

resource_list: RESOURCE_NAME ("," RESOURCE_NAME)*

SERVICE_NAME: CNAME
ALGORITHM_NAME: CNAME
RESOURCE_NAME: CNAME
BLOCK_NAME: CNAME
MODULE_NAME: CNAME
PARAMETER_NAME: CNAME
VARIABLE_NAME: CNAME
LANGUAGE_NAME: CNAME
DATA_TYPE: CNAME
RESOURCE_TYPE: "Variable" | "Buffer" | "List" | "Queue" | "Custom"
STATE_STRATEGY: "Stateless" | "Checkpointing" | "Shared_Data_Store"
DATA_STRUCTURE: "List" | "Queue" | "Set" | "Dictionary" | "Custom"
BOOLEAN: "true" | "false"

value: NUMBER "%"  -> percentage
     | NUMBER      -> number
     | STRING_LITERAL

STRING_LITERAL: ESCAPED_STRING

%import common.CNAME
%import common.NUMBER
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""
