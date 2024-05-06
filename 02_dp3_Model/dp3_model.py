from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
from itertools import combinations
import pandas as pd
import numpy as np

class Dp3Model:
    def __init__(self, features, target, models, param_grids, n_splits=5, metric=f1_score, preprocess=None, max_features=None):
        """
        Inicializador de la clase Dp3Model, diseñada para evaluar múltiples modelos de machine learning sobre
        todas las combinaciones posibles de características de un conjunto de datos.

        :param features: DataFrame con las características (features) del dataset.
        :param target: Serie con la variable objetivo (target) del dataset.
        :param models: Lista de modelos de sklearn (o compatibles) para evaluar.
        :param n_splits: Número de divisiones para la validación cruzada.
        :param metric: Función métrica para evaluar el rendimiento de los modelos.
        :param preprocess: Función para el preprocesamiento de los datos.
        :param max_features: Número máximo de características a considerar en las combinaciones.
        """
        self.features = features
        self.target = target
        self.models = models
        self.param_grids = param_grids
        self.n_splits = n_splits
        self.metric = metric
        self.preprocess = preprocess
        self.kf = KFold(n_splits=self.n_splits)
        self.max_features = max_features if max_features is not None else len(features.columns)

    def feature_combinations(self):
        """
        Genera todas las posibles combinaciones de características hasta el máximo especificado.
        """
        features = list(self.features.columns)
        for i in range(1, min(len(features), self.max_features) + 1):
            for combo in combinations(features, i):
                yield combo

    def evaluate_models(self, X, y):
        """
        Evalúa todos los modelos utilizando GridSearchCV para optimizar los hiperparámetros sobre un conjunto específico de características.
        """
        best_score = -np.inf
        best_model = None
        best_params = None
        for model, params in zip(self.models, self.param_grids):
            grid_search = GridSearchCV(model, params, cv=self.kf, scoring=self.metric)
            grid_search.fit(X, y)
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
        return best_model, best_score, best_params

    def run(self, X_test):
        """
        Ejecuta el proceso de selección y optimización del modelo, evaluando todas las combinaciones de características.
        Evalúa el modelo optimizado en X_test y devuelve las predicciones.
        """
        best_score = -np.inf
        best_model = None
        best_features = None
        for combo in self.feature_combinations():
            X = self.features[list(combo)]
            y = self.target
            if self.preprocess:
                X = self.preprocess(X)
            model, score, params = self.evaluate_models(X, y)
            if score > best_score:
                best_score = score
                best_model = model
                best_features = combo
        # Use the best model and features to predict on the test data
        if best_features:
            X_test_transformed = X_test[list(best_features)]
            if self.preprocess:
                X_test_transformed = self.preprocess(X_test_transformed)
            return best_model.predict(X_test_transformed)
        else:
            return None  # or raise an exception if no model was found


