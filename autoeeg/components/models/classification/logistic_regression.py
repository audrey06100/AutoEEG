from sklearn.linear_model import LogisticRegression
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

class LR:
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', random_state=1):
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        # Scikit-learn 1.2+ requires None instead of 'none' string
        actual_penalty = self.penalty
        if actual_penalty == 'none':
            actual_penalty = None
            
        self.model = LogisticRegression(C=self.C, penalty=actual_penalty, solver=self.solver, 
                                        random_state=self.random_state, max_iter=1000)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        C = UniformFloatHyperparameter("C", 0.01, 100.0, default_value=1.0, log=True)
        # Simplified: keeping it compatible with lbfgs solver for now
        penalty = CategoricalHyperparameter("penalty", ['l2', 'none'], default_value='l2')
        cs.add_hyperparameters([C, penalty])
        return cs
