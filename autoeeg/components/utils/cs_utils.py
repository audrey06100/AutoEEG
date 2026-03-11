from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

def _add_hierachical_configspace(cs, config, parent_name):
    config_cand = list(config.keys())
    
    # 1. Improved Default Selection Logic
    # Priority: 'emptytransformer' > 'empty' > first available
    if 'emptytransformer' in config_cand:
        default_val = 'emptytransformer'
    elif 'empty' in config_cand:
        default_val = 'empty'
    else:
        default_val = config_cand[0]
    
    config_option = CategoricalHyperparameter(parent_name, config_cand, default_value=default_val)
    cs.add_hyperparameter(config_option)
    for config_item in config_cand:
        sub_cs = config[config_item]
        parent_hyperparameter = {'parent': config_option, 'value': config_item}
        cs.add_configuration_space(config_item, sub_cs, parent_hyperparameter=parent_hyperparameter)

def get_cash_cs(algo_candidates):
    if not algo_candidates:
        raise ValueError("algo_candidates cannot be empty!")
    cs = ConfigurationSpace()
    # Normalize algo names to lowercase
    algo_dict = {name.lower(): cls.get_hyperparameter_search_space() for name, cls in algo_candidates.items()}
    _add_hierachical_configspace(cs, algo_dict, 'algorithm')
    return cs

def get_fe_cs(fe_candidates):
    cs = ConfigurationSpace()
    for stage_name, candidates_list in fe_candidates.items():
        stage_dict = {}
        for cand_class in candidates_list:
            # 2. Simplified ID naming: Always use class name lowercase
            cand_name = cand_class.__name__.lower()
            
            # Handle EmptyTransformer specifically only for its search space (empty)
            if cand_name == 'emptytransformer':
                sub_cs = ConfigurationSpace()
            else:
                sub_cs = cand_class.get_hyperparameter_search_space()
            
            stage_dict[cand_name] = sub_cs
        
        _add_hierachical_configspace(cs, stage_dict, stage_name)
    return cs

def get_estimator(config, algo_candidates):
    algo_id = config['algorithm']
    # Match with lowercase algo_candidates keys
    algo_map = {k.lower(): v for k, v in algo_candidates.items()}
    algo_class = algo_map[algo_id]
    model_params = {k.split(':')[1]: v for k, v in config.items() if k.startswith(f"{algo_id}:")}
    return algo_id, algo_class(**model_params)
