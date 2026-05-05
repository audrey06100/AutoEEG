import time
import numpy as np
from mindware.automl import AutoML
from autoeeg.blocks.block_utils import get_node_type
from autoeeg.components.utils.cs_utils import get_fe_cs, get_cash_cs

class EEGAutoML(AutoML):
    def __init__(self, fe_candidates=None, algo_candidates=None, execution_tree=None, **kwargs):
        super().__init__(**kwargs)
        self.fe_candidates = fe_candidates
        self.algo_candidates = algo_candidates
        self.execution_tree = execution_tree
        print(f"[DEBUG] EEGAutoML Initialized.")

    def initialize(self, train_data, **kwargs):
        print("[DEBUG] EEGAutoML.initialize() started")
        from mindware.utils.functions import is_imbalanced_dataset
        from mindware.components.feature_engineering.transformation_graph import DataNode

        y = train_data.data[1]
        if isinstance(y, list):
            flat_labels = np.concatenate([evt[:, 2] for evt in y])
        elif isinstance(y, np.ndarray) and len(y.shape) == 2 and y.shape[1] == 3:
            flat_labels = y[:, 2]
        else:
            flat_labels = y

        tmp_node = DataNode(data=[None, flat_labels])
        self.if_imbal = is_imbalanced_dataset(tmp_node)

        print("[DEBUG] Building FE and CASH ConfigSpaces...")
        self.fe_config_space = get_fe_cs(self.fe_candidates)
        self.cash_config_space = get_cash_cs(self.algo_candidates)

        tree = self.execution_tree
        if tree is None:
            tree = [{'type': 'joint', 'children': []}]
            
        solver_type = get_node_type(tree, 0)

        self.timestamp = time.time()
        print(f"[DEBUG] Creating Solver: {solver_type.__name__}")
        
        # FIX: Pass extra arguments as KWARGS so ConditioningBlock can handle them via AbstractBlock
        self.solver = solver_type(tree, 0, 
                                  task_type=self.task_type, 
                                  timestamp=self.timestamp,
                                  fe_config_space=self.fe_config_space, 
                                  cash_config_space=self.cash_config_space, 
                                  data=train_data,
                                  per_run_time_limit=self.per_run_time_limit,
                                  dataset_name=self.dataset_name,
                                  optimizer=self.optimizer,
                                  ensemble_method=self.ensemble_method,
                                  ensemble_size=self.ensemble_size,
                                  metric=self.metric,
                                  seed=self.seed,
                                  time_limit=self.time_limit,
                                  trial_num=self.amount_of_resource,
                                  eval_type=self.evaluation_type,
                                  resampling_params=self.resampling_params,
                                  output_dir=self.output_dir,
                                  algo_candidates=self.algo_candidates,
                                  n_jobs=self.n_jobs)
        
        print("[DEBUG] EEGAutoML.initialize() completed")
