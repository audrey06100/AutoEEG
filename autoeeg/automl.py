import time
import numpy as np
from mindware.automl import AutoML
from autoeeg.blocks.block_utils import get_eeg_execution_tree, get_node_type
from autoeeg.components.utils.cs_utils import get_fe_cs, get_cash_cs

class EEGAutoML(AutoML):
    def __init__(self, fe_candidates=None, algo_candidates=None, tree_id=0, **kwargs):
        super().__init__(**kwargs)
        self.fe_candidates = fe_candidates
        self.algo_candidates = algo_candidates
        self.tree_id = tree_id
        print(f"[DEBUG] EEGAutoML Initialized. Tree ID: {tree_id}")

    def initialize(self, train_data, **kwargs):
        print("[DEBUG] EEGAutoML.initialize() started")
        from mindware.utils.functions import is_imbalanced_dataset
        from mindware.components.feature_engineering.transformation_graph import DataNode

        # 1. Robust label extraction for imbalanced check
        y = train_data.data[1]
        if isinstance(y, list):
            # Extract 3rd column from List[events]
            flat_labels = np.concatenate([evt[:, 2] for evt in y])
        elif isinstance(y, np.ndarray) and len(y.shape) == 2 and y.shape[1] == 3:
            flat_labels = y[:, 2]
        else:
            flat_labels = y

        # Create a temporary DataNode because mindware's is_imbalanced_dataset
        # expects a DataNode object and accesses .data[1]
        tmp_node = DataNode(data=[None, flat_labels])
        self.if_imbal = is_imbalanced_dataset(tmp_node)

        print("[DEBUG] Building FE and CASH ConfigSpaces...")
        self.fe_config_space = get_fe_cs(self.fe_candidates)
        self.cash_config_space = get_cash_cs(self.algo_candidates)

        tree = get_eeg_execution_tree(self.tree_id)
        solver_type = get_node_type(tree, 0)

        self.timestamp = time.time()
        print(f"[DEBUG] Creating Solver: {solver_type.__name__}")
        self.solver = solver_type(tree, 0, self.task_type, self.timestamp,
                                  self.fe_config_space, self.cash_config_space, train_data,
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

