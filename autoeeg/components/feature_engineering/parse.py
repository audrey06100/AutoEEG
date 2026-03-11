from mindware.components.feature_engineering import task_space
from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode
import numpy as np

def tran_operate(op_id, tran_set, config, node: EEGDataNode):
    _config = {}
    for key in config:
        # Use startswith with a colon to precisely match parameters for this op_id
        if key.startswith(f"{op_id}:"):
            parts = key.split(':')
            if len(parts) > 1:
                config_name = parts[1]
                _config[config_name] = config[key]
            
    # CRITICAL: tran_set needs to handle the requested op_id (usually lowercase)
    tran_map = {k.lower(): v for k, v in tran_set.items()}
    if op_id not in tran_map:
        raise KeyError(f"Operation ID '{op_id}' not found in candidates. Available: {list(tran_map.keys())}")

    # Log the parameters being passed
    print(f"[DEBUG]   -> Parameters: {_config}")

    tran = tran_map[op_id](**_config)
    output_node = tran.operate(node)
    return output_node, tran

def parse_config(data_node: EEGDataNode, config: dict, record=False):
    config_dict = config.copy()
    _node = data_node.copy_()
    tran_dict = dict()

    current_stages = task_space.stage_list
    candidates_dict = task_space.thirdparty_candidates_dict

    print(f"\n[DEBUG] === parse_config starting (Algorithm: {config_dict.get('algorithm')}) ===")

    for stage in current_stages:
        if stage in config_dict:
            op_id = config_dict[stage]
            print(f"[DEBUG] Stage: '{stage}' -> Algorithm: '{op_id}'")
            
            if stage in candidates_dict:
                _node, tran = tran_operate(op_id, candidates_dict[stage], config_dict, _node)
                
                # --- Detailed Dimension Logging ---
                d0, d1 = _node.data
                # Inspect Data[0] (Signals)
                if isinstance(d0, list):
                    n_sub = len(d0)
                    if n_sub > 0:
                        try:
                            # MNE Raw objects have n_times and ch_names
                            detail0 = f"List[{n_sub}], first element: ({len(d0[0].ch_names)} ch, {d0[0].n_times} samples)"
                        except:
                            detail0 = f"List[{n_sub}] of {type(d0[0])}"
                    else:
                        detail0 = "Empty List"
                else:
                    detail0 = f"Array {d0.shape}"

                # Inspect Data[1] (Events/Labels)
                if isinstance(d1, list):
                    n_sub = len(d1)
                    if n_sub > 0:
                        detail1 = f"List[{n_sub}], first element shape: {d1[0].shape}"
                    else:
                        detail1 = "Empty List"
                else:
                    detail1 = f"Array {d1.shape}"

                print(f"[DEBUG]   -> Data[0]: {detail0}")
                print(f"[DEBUG]   -> Data[1]: {detail1}")
                # --------------------------------
                
                tran_dict[stage] = tran
        else:
            pass

    _node.config = config
    if record:
        return _node, tran_dict
    return _node

def construct_node(data_node: EEGDataNode, tran_dict):
    _node = data_node.copy_()
    for stage in task_space.stage_list:
        if stage in tran_dict:
            _node = tran_dict[stage].operate(_node)
    return _node
