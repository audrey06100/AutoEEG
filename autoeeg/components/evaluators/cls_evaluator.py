import warnings
import os
import time
import numpy as np
import pickle as pkl
from sklearn.metrics._scorer import balanced_accuracy_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupKFold, KFold

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.evaluators.evaluate_func import validation
from autoeeg.components.feature_engineering.parse import parse_config, construct_node
from autoeeg.components.utils.cs_utils import get_estimator
from mindware.components.utils.constants import CLASSIFICATION
from mindware.components.metrics.metric import is_threshold_scorer

class EEGClassificationEvaluator(_BaseEvaluator):
    def __init__(self, fixed_config=None, scorer=None, data_node=None, task_type=CLASSIFICATION, 
                 resampling_strategy='cv', resampling_params=None, 
                 algo_candidates=None,
                 timestamp=None, output_dir=None, seed=1):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params or {}

        self.fixed_config = fixed_config
        self.scorer = scorer if scorer is not None else balanced_accuracy_scorer
        self.task_type = task_type
        self.data_node = data_node
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        
        self.algo_candidates = algo_candidates
        
        # method: 'kfold' or 'loso'
        self.val_method = self.resampling_params.get('method', 'kfold')
        # split_by: 'subject' or 'trial'
        self.split_by = self.resampling_params.get('split_by', 'subject')
        self.folds = self.resampling_params.get('folds', 5)
        
        self.train_node = data_node.copy_()
        self.val_node = data_node.copy_()
        self.timestamp = timestamp

    def __call__(self, config, **kwargs):
        start_time = time.time()
        
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()
        if self.fixed_config is not None:
            config.update(self.fixed_config)
        
        # --- NEW: Print Iteration Summary ---
        print(f"\n[ITER] Evaluator received new config:")
        print(f"  -> Algorithm: {config.get('algorithm')}")
        # Print FE params (filtered)
        fe_params = {k: v for k, v in config.items() if ':' in k and not k.startswith(config.get('algorithm'))}
        if fe_params: print(f"  -> FE Params: {fe_params}")
        # ------------------------------------

        classifier_id, clf = get_estimator(config, self.algo_candidates)

        raw_list = self.data_node.data[0] 
        event_list = self.data_node.data[1] 
        subject_ids = np.array(self.data_node.subject_ids) 

        # 1. Define Splitting Strategy
        if self.split_by == 'subject':
            X_indices = np.arange(len(raw_list))
            if self.val_method == 'loso':
                splitter = LeaveOneGroupOut()
                iterator = splitter.split(X_indices, groups=X_indices)
                actual_folds = len(X_indices)
            else:
                splitter = KFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
                iterator = splitter.split(X_indices)
                actual_folds = self.folds
        elif self.split_by == 'trial':
            flat_events_meta = []
            for sub_idx, events in enumerate(event_list):
                for trial_idx, evt in enumerate(events):
                    flat_events_meta.append({'sub_idx': sub_idx, 'local_idx': trial_idx, 'label': evt[2]})
            labels = np.array([m['label'] for m in flat_events_meta])
            X_indices = np.arange(len(flat_events_meta))
            splitter = StratifiedKFold(n_splits=self.folds, random_state=self.seed, shuffle=True)
            iterator = splitter.split(X_indices, y=labels)
            actual_folds = self.folds

        # 2. Run CV
        scores = []
        for i, (train_idx, val_idx) in enumerate(iterator):
            if self.split_by == 'subject':
                t_raw = [raw_list[j] for j in train_idx]
                t_evt = [event_list[j] for j in train_idx]
                t_sub = [subject_ids[j] for j in train_idx]
                v_raw = [raw_list[j] for j in val_idx]
                v_evt = [event_list[j] for j in val_idx]
                v_sub = [subject_ids[j] for j in val_idx]
                # Calculate trial counts for logging
                n_train = sum(len(e) for e in t_evt)
                n_val = sum(len(e) for e in v_evt)
            else:
                # Group trials by subject index
                def get_trial_data(indices):
                    sub_to_trials = {}
                    for idx in indices:
                        s_idx = flat_events_meta[idx]['sub_idx']
                        l_idx = flat_events_meta[idx]['local_idx']
                        if s_idx not in sub_to_trials: sub_to_trials[s_idx] = []
                        sub_to_trials[s_idx].append(l_idx)
                    active_subs = sorted(list(sub_to_trials.keys()))
                    return [raw_list[j] for j in active_subs], \
                           [event_list[j][sub_to_trials[j]] for j in active_subs], \
                           [subject_ids[j] for j in active_subs]
                t_raw, t_evt, t_sub = get_trial_data(train_idx)
                v_raw, v_evt, v_sub = get_trial_data(val_idx)
                n_train, n_val = len(train_idx), len(val_idx)

            print(f"  [Fold {i+1}/{actual_folds}] Train: {n_train} trials, Val: {n_val} trials")

            self.train_node.data = [t_raw, t_evt]
            self.train_node.subject_ids = t_sub
            self.val_node.data = [v_raw, v_evt]
            self.val_node.subject_ids = v_sub

            try:
                # Use verbose=True only for the FIRST fold
                data_node, op_list = parse_config(self.train_node, config, record=True, verbose=(i==0))
                _val_node = construct_node(self.val_node.copy_(), op_list)

                _x_train_processed, _y_train_processed = data_node.data
                _x_val_processed, _y_val_processed = _val_node.data

                if not isinstance(_x_train_processed, list) and len(_x_train_processed.shape) > 2:
                    _x_train_processed = _x_train_processed.reshape(_x_train_processed.shape[0], -1)
                    _x_val_processed = _x_val_processed.reshape(_x_val_processed.shape[0], -1)

                if self.onehot_encoder is None:
                    self.onehot_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
                    all_labels = np.concatenate(event_list)[:, 2]
                    self.onehot_encoder.fit(all_labels.reshape(-1, 1))

                _score = validation(clf, self.scorer, _x_train_processed, _y_train_processed, 
                                    _x_val_processed, _y_val_processed,
                                    random_state=self.seed,
                                    onehot=self.onehot_encoder if is_threshold_scorer(self.scorer) else None)
                scores.append(_score)
                print(f" -> Score: {_score:.4f}")
            except Exception as e:
                self.logger.error(f"Error during CV fold evaluation: {e}")
                scores.append(0.0)
                print(" -> FAILED")
        
        final_score = np.mean(scores) if scores else 0.0
        print(f"  [RESULT] Mean Score: {final_score:.4f}")
        return {'objectives': [-final_score]}
