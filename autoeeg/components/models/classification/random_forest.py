from sklearn.ensemble import RandomForestClassifier
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        # Handle 'None' string from ConfigSpace/YAML
        actual_max_depth = self.max_depth
        if actual_max_depth == 'None' or actual_max_depth == 'none':
            actual_max_depth = None
            
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                            max_depth=actual_max_depth,
                                            min_samples_split=self.min_samples_split,
                                            random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 200, default_value=100)
        max_depth = CategoricalHyperparameter("max_depth", [None, 5, 10, 20], default_value=None)
        cs.add_hyperparameters([n_estimators, max_depth])
        return cs
