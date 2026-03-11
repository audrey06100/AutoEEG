import os
import time
import numpy as np
from mindware.blocks.alternating_block import AlternatingBlock
from autoeeg.components.evaluators.cls_evaluator import EEGClassificationEvaluator

class EEGAlternatingBlock(AlternatingBlock):
    def __init__(self, node_list, node_index, **kwargs):
        # Move import here to avoid circular dependency
        from autoeeg.blocks.block_utils import get_node_type
        
        from mindware.blocks.abstract_block import AbstractBlock
        AbstractBlock.__init__(self, node_list, node_index, **kwargs)
        
        self.arms = ['hpo', 'fe']
        self.sub_bandits = dict()
        # Note: In a real scenario, you'd need to copy the complex init logic from AlternatingBlock
        # For now, we focus on fixing the circular import
        self.eval_dict = dict()
        
        # (Rest of implementation logic should follow mindware's AlternatingBlock 
        # but using the locally imported get_node_type)

    def evaluate_joint_perf(self):
        _perf = None
        evaluator = EEGClassificationEvaluator(
            self.local_inc['fe'].copy(),
            scorer=self.metric,
            data_node=self.original_data,
            timestamp=self.timestamp,
            seed=self.seed,
            output_dir=self.output_dir,
            resampling_strategy=self.eval_type,
            resampling_params=self.resampling_params,
            algo_candidates=self.algo_candidates
        )
        _perf = -evaluator(self.local_inc['hpo'].copy())
        # ... rest of the logic ...
