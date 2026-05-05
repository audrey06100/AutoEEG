from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

class QDA:
    def __init__(self, reg_param=0.0, random_state=1):
        self.reg_param = reg_param
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = QuadraticDiscriminantAnalysis(reg_param=self.reg_param)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        cs = ConfigurationSpace()
        reg_param = UniformFloatHyperparameter("reg_param", 0.0, 1.0, default_value=0.0)
        cs.add_hyperparameters([reg_param])
        return cs
