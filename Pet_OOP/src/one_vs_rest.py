import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
 
class OneVsRestClassifier:
    """
    Класс для реализации стратегии "Один против всех" (One-vs-Rest) для многоклассовой классификации.

    Этот класс использует заданную модель для обучения и предсказания с использованием
    стратегии "один против всех", что означает, что для каждого класса обучается отдельная
    бинарная модель, которая решает, принадлежит ли наблюдение данному классу или нет.
    Поддерживает возможность подбора гиперпараметров для каждой модели.

    Атрибуты
    ---------
    - model: object
        Объект модели, который будет использоваться для классификации.
    - optuna: bool, default=False,
        Переменная для экземпляра класса CustomOptuna.
    - hpspace:
        Пространство гиперпараметров для настройки модели, если используется Optuna.
    - best_models: dict,
        Словарь для хранения лучших моделей для каждого класса.

    Методы
    -------
    - ovr_fit(X, y): Обучает отдельную модель для каждого класса.
    - ovr_predict_proba(X): Предсказывает вероятность принадлежности к классу для заданного набора данных.
    - ovr_predict(X): Предсказывает классы для заданного набора данных
    - ovr_cv(X, y, stratified_kfold): Оценивает модели на кросс-валидации.
    """
    
    def __init__(self, model, optuna=False, hpspace=None):      
        self.model = model
        self.optuna = optuna
        self.hpspace = hpspace
        self.best_models = {}

    def __repr__(self):
        return "OneVsRestClassifier()"

    def ovr_fit(self, X, y):
        """
        Функция для обучения модели с использованием подхода "один против всех" (One-vs-Rest).
    
        Параметры
        ---------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков для обучения модели.
    
        y : array-like, shape (n_samples,)
            Вектор целевых значений, содержащий классы.
    
        Возвращает
        ----------
        self : объект
            Возвращает сам объект.
        """
        
        self.y = y
        custom_optuna = self.optuna
    
        for current_class in sorted(self.y.unique()):
            y_train_single_class = np.where(self.y == current_class, 1, 0)
                
            model_copy = deepcopy(self.model)
        
            if self.optuna:
                study = custom_optuna.study_optimize(model_copy, X, y_train_single_class, self.hpspace)
                model_copy = custom_optuna.objective_to_study.best_model_
                            
            model_copy.fit(X, y_train_single_class)
            self.best_models[str(current_class)] = model_copy        
    
        return self

    def ovr_predict_proba(self, X, normalized=False):
        """
        Функция для предсказания вероятностей принадлежности к классам с использованием моделей, обученных методом "один против всех".
    
        Параметры
        ---------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков для предсказания вероятностей.
    
        normalized : bool, optional, default=False
            Параметр, указывающий на необходимость нормализации вероятностей. Если True, вероятность каждого класса будет обрабатываться так, чтобы их сумма по строкам равнялась 1.
    
        Возвращает
        ----------
        result : DataFrame, shape (n_samples, n_classes)
            DataFrame с вероятностями принадлежности к каждому классу для всех образцов. Индексы соответствуют индексам входного DataFrame `X`.
        """
        
        self.result = pd.DataFrame(index=X.index)
    
        for current_class in sorted(self.y.unique()):
            model = self.best_models[str(current_class)]
            self.result[str(current_class)] = model.predict_proba(X)[:, 1]
    
        if normalized:
            self.result = self.result.div(self.result.sum(axis=1), axis=0) 
    
        return self.result

    def ovr_predict(self, X):
        """
        Функция для предсказания классов на основе вероятностей, полученных с использованием моделей, обученных методом "один против всех".
    
        Параметры
        ---------
        X : array-like, shape (n_samples, n_features)
            Матрица признаков для предсказания классов.
    
        Возвращает
        ----------
        predictions : Series, shape (n_samples,)
            Series, содержащая предсказанные классы для каждого образца, соответствующие индексу входного DataFrame `X`.
        """
        
        proba = self.ovr_predict_proba(X)
        
        return proba.idxmax(axis=1)

    def ovr_cv(self, X, y, stratified_kfold, scorer='f1_score'):
        """
        Выполняет кросс-валидацию с использованием метода "один против всех" для оценки качества классификаторов.
    
        Параметры
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
            Матрица признаков.
    
        y : pandas Series, shape (n_samples,)
            Целевые метки классов для каждого образца.
    
        stratified_kfold : StratifiedKFold
            Объект StratifiedKFold, используемый для разбиения данных на обучающую и тестовую выборки, сохраняя пропорцию классов.
    
        scorer : callable, по умолчанию 'f1_score'
            Функция для оценки качества модели. Должна принимать два аргумента: истинные метки и предсказанные метки. 
            Если строка передана, она должна соответствовать названию функции в sklearn.metrics.
    
        Возвращает
        ----------
        scores : numpy.ndarray, shape (n_splits,)
            Массив оценок, полученных на каждой итерации кросс-валидации.
        
        Исключения
        ----------
        ValueError: 
            Если `stratified_kfold` не является объектом StratifiedKFold.
            Если `self.best_models` пуст, необходимо сначала выполнить `ovr_fit`.
        """
        
        scores = np.array([])
    
        if not isinstance(stratified_kfold, StratifiedKFold):
            raise ValueError("stratified_kfold должен быть объектом StratifiedKFold")
    
        if len(self.best_models) == 0:
            raise ValueError("Сначала выполните ovr_fit")
    
        for train_index, test_index in stratified_kfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
            predictions = np.array([])
    
            for current_class in sorted(y_train.unique()):
                y_train_single_class = np.where(y_train == current_class, 1, 0)
    
                model_copy = deepcopy(self.best_models[f"{current_class}"])
                model_copy.fit(X_train, y_train_single_class)
    
                predictions = np.append(predictions, model_copy.predict_proba(X_test)[:, 1])
                
            predicted_classes = np.argmax(predictions.reshape(len(y_test.unique()), -1), axis=0)
            
            mapping_dict = dict(zip(list(range(len(y_train))), sorted(y_train.unique())))
            predicted_classes_mapped = [mapping_dict[val] for val in predicted_classes]
    
            scores = np.append(scores, scorer(y_test, predicted_classes_mapped))
    
        return scores