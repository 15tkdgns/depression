import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import optuna
from datetime import datetime
from preprocess.data_loader import load_data
from preprocess.preprocessing import preprocess
from preprocess.feature_selection import select_features
from models.dnn import build_dnn
from models.traditional import train_traditional_models
from evaluations.evaluation import evaluate_model
from configs import config
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
import os

#오버샘플링 실험 함수 import
from experiments.oversampling_comparison import (
    oversampling_comparison, smote_k_neighbors_tuning
)


# ----------------- (1) 각 모델별 Optuna objective 함수 정의 -----------------

def objective_dnn(trial, X_train, y_train):
    """
    Optuna로 DNN 하이퍼파라미터(은닉층/드롭아웃) 튜닝
    """
    layers = trial.suggest_categorical(
        'layers',
        [(512, 256), (1024, 512, 256), (1024, 512, 256, 128)]
    )
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    model = build_dnn(X_train.shape[1], layers, dropout=dropout, optimizer='Adam')
    history = model.fit(
        X_train, y_train,
        epochs=30, batch_size=32,
        validation_split=0.2, verbose=0
    )
    return max(history.history['val_accuracy'])

def objective_rf(trial, X_train, y_train):
    """
    Optuna로 RandomForest 하이퍼파라미터 튜닝
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }
    rf = RandomForestClassifier(**params, random_state=config.SEED)
    return cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy').mean()

def objective_xgb(trial, X_train, y_train):
    """
    Optuna로 XGBoost 하이퍼파라미터 튜닝
    """
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0)
    }
    clf = xgb.XGBClassifier(**params, random_state=config.SEED, use_label_encoder=False, eval_metric='mlogloss')
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()

def objective_lgbm(trial, X_train, y_train):
    """
    Optuna로 LightGBM 하이퍼파라미터 튜닝
    """
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0)
    }
    clf = lgb.LGBMClassifier(**params, random_state=config.SEED)
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()

def objective_svm(trial, X_train, y_train):
    """
    Optuna로 SVM 하이퍼파라미터 튜닝
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    params = {
        'C': trial.suggest_float('C', 0.1, 10.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
    }
    clf = SVC(**params, probability=True, random_state=config.SEED)
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()

def objective_lr(trial, X_train, y_train):
    """
    Optuna로 LogisticRegression 하이퍼파라미터 튜닝
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    params = {
        'C': trial.suggest_float('C', 0.01, 10.0, log=True),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    }
    clf = LogisticRegression(**params, max_iter=500, random_state=config.SEED)
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()

def objective_tabnet(trial, X_train, y_train, X_valid, y_valid):
    """
    Optuna로 TabNet 하이퍼파라미터 튜닝
    """
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-1, log=True)
    }
    clf = TabNetClassifier(**params, seed=config.SEED)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        patience=10, max_epochs=100, batch_size=1024
    )
    preds = clf.predict(X_valid)
    return evaluate_model(y_valid, preds)['accuracy']


# ----------------- (2) 메인실행: 전체 파이프라인 + 오버샘플링 실험 추가 -----------------

def main(search_time_minutes=5):
    # === [1] 데이터 로딩/전처리 ===
    df_original = load_data(config.DATA_PATH)
    df, numeric_cols, categorical_cols = preprocess(df_original)
    # 우울증 점수 그룹화: 0(정상), 1(경증), 2(중등도 이상)
    df['mh_PHQ_S_grouped'] = df['mh_PHQ_S'].apply(lambda x: 0 if x <= 4 else 1 if x <= 9 else 2)
    # 타겟/피처 분리
    X = df.drop(['mh_PHQ_S', 'mh_PHQ_S_grouped'], axis=1)
    y = df['mh_PHQ_S_grouped']
    # 결측치 평균대치
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    # 학습/평가 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED)
    # 피처 선택 (미리 지정된 컬럼만)
    X_train_selected, selected_features, selector = select_features(X_train, y_train, config.SELECTED_FEATURES)
    X_test_selected = selector.transform(X_test)
    # 스케일러 (표준화)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # === [2] 오버샘플링 비교 실험 ===
    print("\n[실험] 오버샘플링 방법별 성능 비교 (Original vs SMOTE vs ADASYN vs Borderline-SMOTE)")
    df_os_results = oversampling_comparison(
        X_train_scaled, y_train, X_test_scaled, y_test, config.SEED
    )
    os.makedirs("reports", exist_ok=True)
    df_os_results.to_excel('reports/oversampling_methods_comparison.xlsx', index=False)
    print("[저장] 오버샘플링 성능 비교 결과 → reports/oversampling_methods_comparison.xlsx")

    # === [3] SMOTE k_neighbors 튜닝 실험 ===
    print("\n[실험] SMOTE의 k_neighbors 파라미터 튜닝")
    df_k_results = smote_k_neighbors_tuning(
        X_train_scaled, y_train, X_test_scaled, y_test, config.SEED,
        k_values=[3, 5, 7, 10, 15]
    )
    df_k_results.to_excel('reports/smote_k_neighbors_tuning.xlsx', index=False)
    print("[저장] SMOTE k_neighbors 튜닝 결과 → reports/smote_k_neighbors_tuning.xlsx")


    # === [4] Optuna 하이퍼파라미터 탐색 ===
    timeout = search_time_minutes * 1  # 단위: 초
    model_objectives = [
        ('DNN', objective_dnn),
        ('RandomForest', objective_rf),
        ('XGBoost', objective_xgb),
        ('LightGBM', objective_lgbm),
        ('SVM', objective_svm),
        ('LogisticRegression', objective_lr),
    ]
    studies = {}
    for model_name, objective in model_objectives:
        print(f"\n[Optuna] {model_name} 하이퍼파라미터 튜닝 시작")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), timeout=timeout)
        studies[model_name] = study
        print(f"[Optuna] {model_name} Best score: {study.best_value}, Best params: {study.best_params}")
    # TabNet만 별도
    print("\n[Optuna] TabNet 하이퍼파라미터 튜닝 시작")
    tabnet_study = optuna.create_study(direction='maximize')
    tabnet_study.optimize(
        lambda trial: objective_tabnet(
            trial,
            X_train_scaled.astype(np.float32), y_train.values,
            X_test_scaled.astype(np.float32), y_test.values
        ),
        timeout=timeout
    )
    studies['TabNet'] = tabnet_study
    print(f"[Optuna] TabNet Best score: {tabnet_study.best_value}, Best params: {tabnet_study.best_params}")


    # === [5] Best 파라미터로 모델별 재학습/성능 기록 ===
    model_records = []
    param_records = []
    for model_name, study in studies.items():
        best_params = study.best_params
        print(f"\n[최종 모델 학습] {model_name} | best_params: {best_params}")
        if model_name == 'DNN':
            model = build_dnn(
                X_train_scaled.shape[1],
                best_params['layers'],
                dropout=best_params['dropout']
            )
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            y_pred = model.predict(X_test_scaled).argmax(axis=1)
        elif model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**best_params, random_state=config.SEED)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_name == 'XGBoost':
            import xgboost as xgb
            model = xgb.XGBClassifier(**best_params, random_state=config.SEED, use_label_encoder=False, eval_metric='mlogloss')
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_name == 'LightGBM':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**best_params, random_state=config.SEED)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_name == 'SVM':
            from sklearn.svm import SVC
            model = SVC(**best_params, probability=True, random_state=config.SEED)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**best_params, max_iter=500, random_state=config.SEED)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_name == 'TabNet':
            model = TabNetClassifier(**best_params, seed=config.SEED)
            model.fit(X_train_scaled.astype(np.float32), y_train.values, max_epochs=100, batch_size=1024)
            y_pred = model.predict(X_test_scaled.astype(np.float32))
        else:
            continue
        metrics = evaluate_model(y_test, y_pred)
        model_records.append({'Model': model_name, **metrics})
        param_records.append({'Model': model_name, 'Hyperparameters': best_params})


    # === [6] 결과 리포트 저장 ===
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    excel_filename = f"reports/{now}_detailed_model_report.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        pd.DataFrame(model_records).to_excel(writer, sheet_name='Performance', index=False)
        pd.DataFrame(param_records).to_excel(writer, sheet_name='Hyperparameters', index=False)
    print(f"\n모든 모델 결과가 '{excel_filename}'에 저장되었습니다.")

if __name__ == "__main__":
    main(search_time_minutes=5)
