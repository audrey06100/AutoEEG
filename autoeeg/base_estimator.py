from mindware.base_estimator import BaseEstimator
from autoeeg.automl import EEGAutoML

class BaseEEGEstimator(BaseEstimator):
    def __init__(self, fe_candidates=None, algo_candidates=None, **kwargs):
        super().__init__(**kwargs)
        self.fe_candidates = fe_candidates
        self.algo_candidates = algo_candidates
        print(f"[DEBUG] BaseEEGEstimator Initialized. FE Stages: {len(fe_candidates) if fe_candidates else 0}")

    def get_automl(self):
        return EEGAutoML

    def build_engine(self):
        print("[DEBUG] BaseEEGEstimator.build_engine() started")
        automl_class = self.get_automl()
        engine = automl_class(
            fe_candidates=self.fe_candidates,
            algo_candidates=self.algo_candidates,
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            metric=self.metric,
            time_limit=self.time_limit,
            amount_of_resource=self.amount_of_resource,
            include_algorithms=self.include_algorithms,
            include_preprocessors=self.include_preprocessors,
            enable_meta_algorithm_selection=self.enable_meta_algorithm_selection,
            enable_fe=self.enable_fe,
            optimizer=self.optimizer,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            per_run_time_limit=self.per_run_time_limit,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            evaluation=self.evaluation,
            resampling_params=self.resampling_params,
            output_dir=self.output_dir
        )
        print("[DEBUG] BaseEEGEstimator.build_engine() completed")
        return engine

    def feature_origin(self):
        return None

    def feature_corelation(self, data):
        return None
