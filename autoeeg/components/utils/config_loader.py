import yaml
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter, 
    UniformIntegerHyperparameter, 
    CategoricalHyperparameter,
    Constant
)
from autoeeg.components.utils.registry import get_transformer_class, get_model_class

class ConfigLoader:
    @staticmethod
    def load_yaml(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def build_hp(name, spec):
        hp_type = spec[0]
        if hp_type == "uniform_float":
            return UniformFloatHyperparameter(name, spec[1], spec[2], default_value=spec[3])
        elif hp_type == "uniform_int":
            return UniformIntegerHyperparameter(name, spec[1], spec[2], default_value=spec[3])
        elif hp_type == "categorical":
            return CategoricalHyperparameter(name, spec[1], default_value=spec[2])
        elif hp_type == "constant":
            return Constant(name, spec[1])
        else:
            raise ValueError(f"Unknown hyperparameter type: {hp_type}")

    @classmethod
    def build_cs(cls, search_space_spec):
        cs = ConfigurationSpace()
        for hp_name, hp_spec in search_space_spec.items():
            cs.add_hyperparameter(cls.build_hp(hp_name, hp_spec))
        return cs

    @classmethod
    def get_fe_candidates(cls, config):
        fe_cfg = config.get('fe_pipeline', {})
        candidates = {}
        ss_spec = fe_cfg.get('search_space', {})
        
        for stage, cand_names in fe_cfg.get('candidates', {}).items():
            cand_classes = []
            for name in cand_names:
                cls_obj = get_transformer_class(name)
                # Attach custom ConfigSpace to the class dynamically
                if name in ss_spec:
                    cls_obj.custom_cs = cls.build_cs(ss_spec[name])
                else:
                    # Clear any previous experiment's custom_cs
                    if hasattr(cls_obj, 'custom_cs'):
                        del cls_obj.custom_cs
                cand_classes.append(cls_obj)
            candidates[stage] = cand_classes
        return candidates

    @classmethod
    def get_algo_candidates(cls, config):
        algo_cfg = config.get('algorithms', {})
        candidates = {}
        ss_spec = algo_cfg.get('search_space', {})
        
        for algo_id, class_name in algo_cfg.items():
            if algo_id == 'search_space': continue
            cls_obj = get_model_class(class_name)
            if class_name in ss_spec:
                cls_obj.custom_cs = cls.build_cs(ss_spec[class_name])
            else:
                if hasattr(cls_obj, 'custom_cs'):
                    del cls_obj.custom_cs
            candidates[algo_id] = cls_obj
        return candidates

    @classmethod
    def get_execution_tree(cls, config):
        tree = config.get('execution_tree')
        if tree is None:
            # Default to a simple JointBlock tree
            return [{'type': 'joint', 'children': []}]
        return tree

