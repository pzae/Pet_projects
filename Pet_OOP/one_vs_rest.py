import pandas as pd
import numpy as np
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold
 
class OneVsRestClassifier():
    def __init__(self, model, optuna=False, hpspace=None):
        
        self.model = model
        self.optuna = optuna
        self.hpspace = hpspace

    def __repr__(self):
        return ""

    def ovr_fit(self, X, y):

        self.y = y
        custom_optuna = self.optuna
        self.best_models = {}
        
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
    
        self.result = pd.DataFrame(index=X.index)
    
        for current_class in sorted(self.y.unique()):
            
            model = self.best_models[str(current_class)]
            self.result[str(current_class)] = model.predict_proba(X)[:,1]
    
        if normalized:
            self.result = self.result.div(self.result.sum(axis=1), axis=0) 
            
        return self.result

    def ovr_predict(self, X):
        
        proba = self.ovr_predict_proba(X)
        
        return proba.idxmax(axis=1)

    def ovr_cv(self, X, y, stratified_kfold, scorer='f1_score'):
    
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