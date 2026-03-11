from autoeeg.blocks.joint_block import EEGJointBlock
from autoeeg.blocks.conditioning_block import EEGConditioningBlock
from autoeeg.blocks.alternating_block import EEGAlternatingBlock

def get_eeg_execution_tree(execution_id):
    # Mimic mindware's tree structures but we can customize for EEG
    trees = {0: [('joint', [])],
             1: [('condition', [1]), ('joint', [])],
             2: [('condition', [1]), ('alternate', [2, 3]), ('joint', []), ('joint', [])],
             3: [('alternate', [1, 2]), ('joint', []), ('joint', [])]}
    return trees.get(execution_id, [('joint', [])])

def get_node_type(node_list, index):
    # This is the magic router that picks our EEG blocks
    node_name = node_list[index][0]
    if node_name == 'joint':
        return EEGJointBlock
    elif node_name == 'condition':
        return EEGConditioningBlock
    elif node_name == 'alternate':
        return EEGAlternatingBlock
    else:
        raise ValueError(f"Unknown block type: {node_name}")
