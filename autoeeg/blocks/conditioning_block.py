import time
import numpy as np
from ConfigSpace import ConfigurationSpace, Constant
from copy import deepcopy
from mindware.utils.constant import MAX_INT
from mindware.blocks.conditioning_block import ConditioningBlock

class EEGConditioningBlock(ConditioningBlock):
    def __init__(self, node_list, node_index,
                 task_type, timestamp,
                 fe_config_space: ConfigurationSpace,
                 cash_config_space: ConfigurationSpace,
                 data,
                 fixed_config=None,
                 trial_num=0,
                 time_limit=None,
                 metric='acc',
                 optimizer='smac',
                 ensemble_method='ensemble_selection',
                 ensemble_size=50,
                 per_run_time_limit=300,
                 output_dir="logs",
                 dataset_name='default_dataset',
                 eval_type='holdout',
                 resampling_params=None,
                 algo_candidates=None,
                 n_jobs=1,
                 seed=1):
        
        # --- MASKING (Standard for EEG DataNode) ---
        from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode
        if isinstance(data, EEGDataNode) and isinstance(data.data[1], list):
            flat_labels = np.concatenate([evt[:, 2] for evt in data.data[1]])
            masked_data = data.copy_()
            masked_data.data[1] = flat_labels
        else:
            masked_data = data

        from mindware.blocks.abstract_block import AbstractBlock
        AbstractBlock.__init__(self, node_list, node_index, task_type, timestamp,
                               fe_config_space, cash_config_space, masked_data,
                               fixed_config=fixed_config,
                               trial_num=trial_num,
                               time_limit=time_limit,
                               metric=metric,
                               optimizer=optimizer,
                               ensemble_method=ensemble_method,
                               ensemble_size=ensemble_size,
                               per_run_time_limit=per_run_time_limit,
                               output_dir=output_dir,
                               dataset_name=dataset_name,
                               eval_type=eval_type,
                               resampling_params=resampling_params,
                               n_jobs=n_jobs,
                               seed=seed)

        from autoeeg.blocks.block_utils import get_node_type
        self.original_data = data.copy_()
        self.algo_candidates = algo_candidates
        
        current_node = node_list[node_index]
        if not isinstance(current_node, dict):
            raise ValueError("ConditioningBlock expects node_list elements to be dictionaries.")
            
        stage_to_split = current_node.get('on_stage')
        if not stage_to_split:
            raise ValueError("ConditioningBlock requires 'on_stage' to be specified in the execution tree node.")

        self.is_algorithm_split = (stage_to_split == 'algorithm')
        
        if self.is_algorithm_split:
            self.target_space = self.cash_config_space
            self.splitting_hp_name = 'algorithm'
        else:
            self.target_space = self.fe_config_space
            self.splitting_hp_name = stage_to_split
            
        all_target_hps = self.target_space.get_hyperparameters()
        self.arms = list(self.target_space.get_hyperparameter(self.splitting_hp_name).choices)
        
        self.alpha = 4
        self.sub_bandits = dict()
        self.rewards = {arm: [] for arm in self.arms}
        self.evaluation_cost = {arm: [] for arm in self.arms}
        self.arm_cost_stats = {arm: [] for arm in self.arms}

        for arm in self.arms:
            arm_fixed_config = (fixed_config or {}).copy()
            arm_fixed_config[self.splitting_hp_name] = arm
            
            reduced_space = ConfigurationSpace()
            for hp in all_target_hps:
                if hp.name == self.splitting_hp_name: continue
                hp_prefix = hp.name.split(':')[0]
                if hp_prefix in self.arms:
                    if hp_prefix == arm: reduced_space.add_hyperparameter(hp)
                else:
                    reduced_space.add_hyperparameter(hp)
            
            for cond in self.target_space.get_conditions():
                if cond.parent.name in reduced_space.get_hyperparameter_names():
                    reduced_space.add_condition(cond)
            for forbi in self.target_space.get_forbiddens():
                reduced_space.add_forbidden_clause(forbi)

            if self.is_algorithm_split:
                child_fe_cs = deepcopy(self.fe_config_space)
                child_cash_cs = reduced_space
            else:
                child_fe_cs = reduced_space
                child_cash_cs = deepcopy(self.cash_config_space)

            child_type = get_node_type(node_list, node_index + 1)
            self.sub_bandits[arm] = child_type(
                node_list, node_index + 1, task_type, timestamp,
                child_fe_cs, child_cash_cs, self.original_data.copy_(),
                fixed_config=arm_fixed_config, trial_num=trial_num, time_limit=time_limit,
                metric=metric, optimizer=optimizer, ensemble_method=ensemble_method,
                ensemble_size=ensemble_size, per_run_time_limit=per_run_time_limit,
                output_dir=output_dir, dataset_name=dataset_name, eval_type=eval_type,
                resampling_params=resampling_params, algo_candidates=algo_candidates,
                n_jobs=n_jobs, seed=seed
            )

        self.action_sequence = list()
        self.final_rewards = list()
        self.start_time = time.time()
        self.time_records = list()
        
        # New: Track chronological evaluation history
        self.global_history = []

        self.pull_cnt = 0
        self.pick_id = 0
        self.update_cnt = 0
        arm_num = len(self.arms)
        self.optimal_arm_id = None
        self.arm_candidate = self.arms.copy()
        self.best_lower_bounds = np.zeros(arm_num)
        if self.time_limit is None:
            if arm_num * self.alpha > self.trial_num:
                raise ValueError('Trial number should be larger than %d.' % (arm_num * self.alpha))
        else:
            self.trial_num = MAX_INT

    def iterate(self, trial_num=10):
        # 1. Capture history size before pulling
        arm_to_pull = self.arm_candidate[self.pick_id]
        history_before = len(self.sub_bandits[arm_to_pull].get_history())
        
        # 2. Perform the actual pull (calls sub-bandit.iterate)
        val = super().iterate(trial_num=trial_num)
        
        # 3. Capture new observations added during this pull
        history_after = self.sub_bandits[arm_to_pull].get_history()
        new_obs = history_after[history_before:]
        
        for obs in new_obs:
            self.global_history.append((arm_to_pull, obs))
            
        return val

    def get_history(self):
        """Returns the correctly ordered global history across all branches."""
        return self.global_history
