from ConfigSpace import ConfigurationSpace, Constant
from copy import deepcopy
from mindware.blocks.conditioning_block import ConditioningBlock

class EEGConditioningBlock(ConditioningBlock):
    def __init__(self, node_list, node_index, **kwargs):
        # Move import here to avoid circular dependency
        from autoeeg.blocks.block_utils import get_node_type
        
        from mindware.blocks.abstract_block import AbstractBlock
        AbstractBlock.__init__(self, node_list, node_index, **kwargs)
        
        self.optimal_arm = None
        self.best_lower_bounds = None
        self.alpha = 4
        self.arms = list(kwargs['cash_config_space'].get_hyperparameter('algorithm').choices)
        self.rewards = dict()
        self.sub_bandits = dict()
        self.evaluation_cost = dict()
        self.arm_cost_stats = {arm: [] for arm in self.arms}
        self.eval_dict = dict()

        for arm in self.arms:
            self.rewards[arm] = list()
            self.evaluation_cost[arm] = list()
            
            hps = kwargs['cash_config_space'].get_hyperparameters()
            cs = ConfigurationSpace()
            cs.add_hyperparameter(Constant('algorithm', arm))
            for hp in hps:
                if hp.name.split(':')[0] == arm:
                    cs.add_hyperparameter(hp)

            child_type = get_node_type(node_list, node_index + 1)
            
            child_kwargs = kwargs.copy()
            child_kwargs['cash_config_space'] = cs
            child_kwargs['fe_config_space'] = deepcopy(kwargs['fe_config_space'])
            child_kwargs['data'] = kwargs['data'].copy_()
            
            self.sub_bandits[arm] = child_type(node_list, node_index + 1, **child_kwargs)

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = kwargs['timestamp']
        self.time_records = list()
        self.pull_cnt = 0
        self.pick_id = 0
        self.update_cnt = 0
        self.optimal_algo_id = None
        self.arm_candidate = self.arms.copy()
        import numpy as np
        self.best_lower_bounds = np.zeros(len(self.arms))
