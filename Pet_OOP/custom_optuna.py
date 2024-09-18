import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline, make_pipeline

from IPython.display import Markdown, display
from copy import deepcopy

class CustomOptuna:
    """
    Класс CustomOptuna представляет собой настраиваемый класс для оптимизации гиперпараметров с использованием библиотеки Optuna.

    Параметры
    ---------
    n_trials : int, default=100
        Количество пробных запусков для оптимизации.
        
    timeout : int, default=180
        Временной лимит на выполнение оптимизации.
        
    cv : int, default=7
        Количество фолдов для кросс-валидации для оценки набора гипер-параметров.
        
    interval : int, default=100
        Количество иттераций для остановки расчёта, если не было улучшений метрики.
        
    n_jobs : int, default=1
        Количество процессорных ядер, используемых параллельно для оптимизации.
        
    random_state : int, default=None
        Случайное состояние для воспроизводимости результатов.
        
    is_pipeline : bool, default=True
        Флаг, указывающий, используется ли класс Pipeline.
        
    show_progress_bar : bool, default=True
        Флаг для отображения прогресс-бара в процессе оптимизации.
        
    early_stopping_callback : bool, default=True
        Флаг для использования функции ранней остановки.
        
    scorer : str, default='f1_macro'
        Метрика для оценки качества модели во время оптимизации.

    Ошибки
    ------
    Если n_trials меньше или равен нулю, выбрасывается исключение ValueError.
    Если timeout меньше нуля, выбрасывается исключение ValueError.
    Если interval меньше или равен нулю, выбрасывается исключение ValueError.
    Если n_jobs не равен 1, отображается предупреждение о возможном отличии результатов.
    Если random_state равен None, отображается предупреждение о возможном отличии результатов.
    """
    
    def __init__(self, n_trials=100, timeout=180, cv=7, interval=100, n_jobs=1, random_state=None, 
                 is_pipeline=True, show_progress_bar=True, early_stopping_callback=True, scorer='f1_macro'):
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.cv = cv
        self.show_progress_bar = show_progress_bar
        self.early_stopping_callback = early_stopping_callback
        self.interval = interval
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.is_pipeline = is_pipeline
        self.scorer = scorer
        
        if self.n_trials <= 0:
            raise ValueError("n_trials должен быть отрицательным числом")

        if self.timeout < 0:
            raise ValueError("timeout не может быть положительным числом")

        if self.interval <= 0:
            raise ValueError("interval должен быть отрицательным числом")

        if self.n_jobs != 1:
            display(Markdown("**Результаты выполнения optuna могут отличаться из-за n_jobs != 1**"))
            
        if self.random_state == None:
            display(Markdown("**Результаты выполнения optuna могут отличаться из-за random_state == None**"))

    class Objective:
        
        """
        Класс Objective используется для оптимизации гиперпараметров модели.
    
        Параметры
        ----------
        model : object
            Модель  для оптимизации.
        X : array-like
            Признаки модели.
        y : array-like
            Целевая переменная модели.
        hpspace : function
            Функция, возвращающая пространство гиперпараметров для оптимизации.
        cv : int
            Количество фолдов для кросс-валидации.
        scorer : str
            Метрика для оценки качества модели во время оптимизации.
        random_state : int
            Случайное состояние для воспроизводимости.
        is_pipeline : bool
            Флаг, указывающий является ли модель пайплайном.
    
        Атрибуты
        --------
        best_model_ : object, по умолчанию=None
            Лучшая модель после оптимизации.
        """
        
        def __init__(self, model, X, y, hpspace, cv, scorer, random_state, is_pipeline):
            """
            Инициализирует экземпляр класса Objective с переданными параметрами.
            """
            self.model = model
            self.X = X
            self.y = y
            self.hpspace = hpspace
            self.cv = cv
            self.scorer = scorer
            self.random_state = random_state
            self.is_pipeline = is_pipeline
            self.best_model_ = None
            
        def __call__(self, trial):
            """
            Основной метод для оптимизации модели.
            """
            if self.is_pipeline:
                self.model.steps[-1][1].set_params(**self.hpspace(trial))
            else:
                self.model.set_params(**self.hpspace(trial))
    
            stratified_kfold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(self.model, self.X, self.y, scoring=self.scorer, cv=stratified_kfold)
    
            return np.mean(scores)
    
        def BestModelCallback(self, study, trial):
            """
            Callback-метод для определения лучшей модели после оптимизации.
            """
            if study.best_trial.number == trial.number:
                self.best_model_ = self.model

    class EarlyStoppingCallback:
        """
        Класс обратного вызова для ранней остановки изучения (study) в оптимизации гиперпараметров.
    
        Параметры
        ---------
        interval : int
            Интервал для ранней остановки. Если разница в номерах между текущим испытанием (trial) 
            и лучшим испытанием (best_trial) больше этого интервала, изучение прекращается.
        """
        def __init__(self, interval):
            """
            Инициализирует экземпляр класса EarlyStoppingCallback с переданными параметрами.
            """
            self.interval = interval
            
        def __call__(self, study, trial):
            """
            Метод, который вызывается для проверки условия ранней остановки и при необходимости прекращает изучение.
    
            Параметры
            ---------
            study : объект Study
                Объект исследования, содержащий информацию о текущем исследовании.
            trial : объект Trial
                Текущее испытание, для которого проверяется условие ранней остановки.
            """
            best_trial = study.best_trial
            if trial.number - best_trial.number > self.interval:
                study.stop()

    def hpspaces_default(self, model):
        """
        Возвращает пространство поиска гиперпараметров по умолчанию для указанной модели.
    
        Параметры:
        - model: объект модели
    
        Возвращает функцию с настройками гиперпараметров для указанной модели.
        """
    
        def hpspace_LR_default(trial):
            """
            Возвращает пространство поиска гиперпараметров для LogisticRegression.
    
            Параметры:
            - trial: объект Trial
    
            Возвращает словарь с параметрами модели LogisticRegression.
            """
            return {
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
                    'random_state': self.random_state
                }
    
        def hpspace_RFC_default(trial):
            """
            Возвращает пространство поиска гиперпараметров для RandomForestClassifier.
    
            Параметры:
            - trial: объект Trial
    
            Возвращает словарь с параметрами модели RandomForestClassifier.
            """
            return {
                    'max_depth' : trial.suggest_int('max_depth', 2, 10),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'min_samples_split' : trial.suggest_int('min_samples_split', 2, 10),
                    'n_estimators' : trial.suggest_int('n_estimators', 25, 200),
                    'criterion': 'entropy',
                    'random_state': self.random_state
                }
    
        def hpspace_LGBM_default(trial):
            """
            Возвращает пространство поиска гиперпараметров для LGBMClassifier.
    
            Параметры:
            - trial: объект Trial
    
            Возвращает словарь с параметрами модели LGBMClassifier.
            """
            return {
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                    'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                    'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
                    'verbose': -1,
                    'random_state': self.random_state
                }
    
        def hpspace_XGB_default(trial):
            """
            Возвращает пространство поиска гиперпараметров для XGBClassifier.
    
            Параметры:
            - trial: объект Trial
    
            Возвращает словарь с параметрами модели XGBClassifier.
            """
            return {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-8, 10.0),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
                    'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                    'verbosity': 0,
                    'seed': self.random_state
                }
    
        def hpspace_CB_default(trial):
            """
            Возвращает пространство поиска гиперпараметров для CatBoostClassifier.
    
            Параметры:
            - trial: объект Trial
    
            Возвращает словарь с параметрами модели CatBoostClassifier.
            """
            return {
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
                    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
                    'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 10.0),
                    'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 1e-8, 10.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_state':self.random_state,
                    'verbose': False
                }
    
        params = {'LogisticRegression': hpspace_LR_default,
                  'RandomForestClassifier': hpspace_RFC_default,
                  'LGBMClassifier': hpspace_LGBM_default,
                  'XGBClassifier': hpspace_XGB_default,
                  'CatBoostClassifier': hpspace_CB_default}
    
        if self.is_pipeline:
            name_model = model.steps[-1][1].__class__.__name__
        else:
            name_model = model.__class__.__name__
            
        return params[name_model]

    def study_optimize(self, model, X, y, hpspace=None):
        """
        Метод для оптимизации модели.
    
        Параметры
        ---------
        model : object
            Обучаемая модель, которую необходимо оптимизировать.
        X : array-like
            Матрица признаков для обучения модели.
        y : array-like
            Целевая переменная для обучения модели.
        hpspace : dict
            Пространство поиска гиперпараметров модели.
    
        Возвращает
        ----------
        study : object
            Объект Study после завершения оптимизации.
        """
        
        optuna_model = deepcopy(model)
        callbacks = []
        study = None
    
        if self.early_stopping_callback:
            callbacks.append(self.EarlyStoppingCallback(self.interval))
    
        if hpspace:
            self.hpspace = hpspace
        else:
            self.hpspace = self.hpspaces_default(optuna_model)
    
        self.objective_to_study = self.Objective(optuna_model, X, y, self.hpspace, self.cv, self.scorer, self.random_state, self.is_pipeline)
        callbacks.append(self.objective_to_study.BestModelCallback)
        
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.random_state), direction='maximize')
        study.optimize(self.objective_to_study, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs, callbacks=callbacks, show_progress_bar=self.show_progress_bar)
    
        return study

    def optuna_results(self, study):
        """
        Функция для отображения результатов оптимизации.
    
        Параметры
        ---------
        study : object
            Объект Study, содержащий результаты оптимизации.
    
        Возвращает
        ----------
        Вывод результатов обучения.
        Пустая строка.
        """
        
        display(Markdown(f'Лучшая метрика: {study.best_trial.value:.4}'))
        display(Markdown(f'Лучший набор гиперпараметров: {study.best_params}'))
        
        return ''