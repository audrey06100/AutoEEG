from sklearn.svm import SVC
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

class SVM:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=1):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, 
                         random_state=self.random_state, probability=True)
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
        kernel = CategoricalHyperparameter("kernel", ['linear', 'rbf', 'poly', 'sigmoid'], default_value='rbf')
        cs.add_hyperparameters([C, kernel])
        return cs
