import time
import numpy as np
from ConfigSpace import ConfigurationSpace
from mindware.blocks.joint_block import JointBlock
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.optimizers import build_hpo_optimizer
from autoeeg.components.evaluators.cls_evaluator import EEGClassificationEvaluator
from mindware.components.feature_engineering.transformation_graph import DataNode

class EEGJointBlock(JointBlock):
    def __init__(self, node_list, node_index,
                 task_type, timestamp,
                 fe_config_space: ConfigurationSpace,
                 cash_config_space: ConfigurationSpace,
                 data,
                 fixed_config=None,
                 time_limit=None,
                 trial_num=0,
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
        
        # --- MASKING START ---
        # Hack: AbstractBlock.__init__ calls is_imbalanced_dataset(data) 
        # which fails on EEGDataNode list labels.
        # We pass a temporary copy with flattened labels to the base class.
        from autoeeg.components.feature_engineering.transformation_graph import EEGDataNode
        if isinstance(data, EEGDataNode) and isinstance(data.data[1], list):
            flat_labels = np.concatenate([evt[:, 2] for evt in data.data[1]])
            masked_data = data.copy_()
            masked_data.data[1] = flat_labels # Base class can handle this
        else:
            masked_data = data
        # --- MASKING END ---

        from mindware.blocks.abstract_block import AbstractBlock
        AbstractBlock.__init__(self, node_list, node_index, task_type, timestamp,
                               fe_config_space, cash_config_space, masked_data,
                               fixed_config=fixed_config,
                               time_limit=time_limit,
                               trial_num=trial_num,
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

        # Restore the original EEG data (with lists) after base init
        self.original_data = data.copy_()
        self.fixed_config = fixed_config
        self.algo_candidates = algo_candidates
        print(f"[DEBUG] EEGJointBlock Initialized. Candidates: {list(algo_candidates.keys()) if algo_candidates else 'None'}")

        # Combine configuration space
        cs = ConfigurationSpace()
        if fe_config_space is not None:
            cs.add_hyperparameters(fe_config_space.get_hyperparameters())
            cs.add_conditions(fe_config_space.get_conditions())
            cs.add_forbidden_clauses(fe_config_space.get_forbiddens())
        if cash_config_space is not None:
            cs.add_hyperparameters(cash_config_space.get_hyperparameters())
            cs.add_conditions(cash_config_space.get_conditions())
            cs.add_forbidden_clauses(cash_config_space.get_forbiddens())
        self.joint_cs = cs

        # Define EEG evaluator
        self.evaluator = EEGClassificationEvaluator(
            fixed_config=fixed_config,
            scorer=self.metric,
            data_node=self.original_data,
            timestamp=self.timestamp,
            output_dir=self.output_dir,
            seed=self.seed,
            resampling_strategy=self.eval_type,
            resampling_params=self.resampling_params,
            algo_candidates=self.algo_candidates)

        self.optimizer = build_hpo_optimizer(self.eval_type, self.evaluator, self.joint_cs,
                                             optimizer=self.optimizer,
                                             output_dir=self.output_dir,
                                             per_run_time_limit=self.per_run_time_limit,
                                             inner_iter_num_per_iter=1,
                                             timestamp=self.timestamp,
                                             seed=self.seed, n_jobs=self.n_jobs)
