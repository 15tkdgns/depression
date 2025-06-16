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

def objective_dnn(trial, X_train, y_train):
    layers = trial.suggest_categorical('layers', [(512, 256), (1024, 512, 256), (1024, 512, 256, 128)])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    model = build_dnn(X_train.shape[1], layers, dropout=dropout, optimizer='Adam')
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

    return max(history.history['val_accuracy'])


def objective_rf(trial, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }
    rf = RandomForestClassifier(**params, random_state=config.SEED)
    return cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy').mean()


def objective_tabnet(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-1, log=True)
    }
    clf = TabNetClassifier(**params, seed=config.SEED)
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], patience=10, max_epochs=100, batch_size=1024)
    preds = clf.predict(X_valid)
    return evaluate_model(y_valid, preds)['accuracy']


def main(search_time_minutes=5):
    df_original = load_data(config.DATA_PATH)
    df, numeric_cols, categorical_cols = preprocess(df_original)

    df['mh_PHQ_S_grouped'] = df['mh_PHQ_S'].apply(lambda x: 0 if x <= 4 else 1 if x <= 9 else 2)
    X = df.drop(['mh_PHQ_S', 'mh_PHQ_S_grouped'], axis=1)
    y = df['mh_PHQ_S_grouped']

    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED)
    X_train_selected, selected_features, selector = select_features(X_train, y_train, config.SELECTED_FEATURES)
    X_test_selected = selector.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    timeout = search_time_minutes * 5

    studies = {}
    for model_name, objective in [('DNN', objective_dnn), ('RandomForest', objective_rf)]:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), timeout=timeout)
        studies[model_name] = study

    tabnet_study = optuna.create_study(direction='maximize')
    tabnet_study.optimize(lambda trial: objective_tabnet(trial, X_train_scaled.astype(np.float32), y_train.values, X_test_scaled.astype(np.float32), y_test.values), timeout=timeout)
    studies['TabNet'] = tabnet_study

    model_records = []
    param_records = []

    for model_name, study in studies.items():
        best_params = study.best_params
        if model_name == 'DNN':
            model = build_dnn(X_train_scaled.shape[1], best_params['layers'], dropout=best_params['dropout'])
            model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
            y_pred = model.predict(X_test_scaled).argmax(axis=1)
        elif model_name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**best_params, random_state=config.SEED)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        elif model_name == 'TabNet':
            model = TabNetClassifier(**best_params, seed=config.SEED)
            model.fit(X_train_scaled.astype(np.float32), y_train.values, max_epochs=100, batch_size=1024)
            y_pred = model.predict(X_test_scaled.astype(np.float32))

        metrics = evaluate_model(y_test, y_pred)

        model_records.append({
            'Model': model_name,
            **metrics,
        })

        param_records.append({
            'Model': model_name,
            'Hyperparameters': best_params
        })

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    excel_filename = f"reports/{now}_detailed_model_report.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        pd.DataFrame(model_records).to_excel(writer, sheet_name='Performance', index=False)
        pd.DataFrame(param_records).to_excel(writer, sheet_name='Hyperparameters', index=False)

    print(f"Detailed report saved to '{excel_filename}'")

if __name__ == "__main__":
    main(search_time_minutes=5)
