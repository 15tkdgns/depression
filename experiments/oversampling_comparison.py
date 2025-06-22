# depression/experiments/oversampling_comparison.py

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

from sklearn.ensemble import RandomForestClassifier
from evaluations.evaluation import evaluate_model
import pandas as pd

def oversampling_comparison(X_train, y_train, X_test, y_test, random_state):
    """
    여러 오버샘플링 기법(Original, SMOTE, ADASYN, BorderlineSMOTE) 적용 후
    RandomForestClassifier로 성능 평가, DataFrame 반환
    """
    results = []
    samplers = {
        'Original': None,
        'SMOTE': SMOTE(random_state=random_state),
        'ADASYN': ADASYN(random_state=random_state),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=random_state)
    }
    for sampler_name, sampler in samplers.items():
        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results.append({'Sampler': sampler_name, **metrics})
    return pd.DataFrame(results)

def smote_k_neighbors_tuning(X_train, y_train, X_test, y_test, random_state, k_values=[3, 5, 7, 10, 15]):
    """
    SMOTE의 k_neighbors 파라미터 값별 성능 비교, DataFrame 반환
    """
    results = []
    for k in k_values:
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results.append({'k_neighbors': k, **metrics})
    return pd.DataFrame(results)
