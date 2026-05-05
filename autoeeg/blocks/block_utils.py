from autoeeg.blocks.joint_block import EEGJointBlock
from autoeeg.blocks.conditioning_block import EEGConditioningBlock
from autoeeg.blocks.alternating_block import EEGAlternatingBlock

def get_node_type(node_list, index):
    # This is the magic router that picks our EEG blocks
    # Supports both the new YAML dictionary format and the old tuple format for backwards compatibility
    node = node_list[index]
    if isinstance(node, dict):
        node_name = node.get('type')
    else:
        node_name = node[0]

    if node_name == 'joint':
        return EEGJointBlock
    elif node_name == 'condition':
        return EEGConditioningBlock
    elif node_name == 'alternate':
        return EEGAlternatingBlock
    else:
        raise ValueError(f"Unknown block type: {node_name}")
