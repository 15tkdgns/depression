import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
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
from models.cnn1d import build_1dcnn
from pytorch_tabnet.tab_model import TabNetClassifier

def main():
    # 데이터 로딩 및 전처리
    df_original = load_data(config.DATA_PATH)
    df, numeric_cols, categorical_cols = preprocess(df_original)

    df['mh_PHQ_S_grouped'] = df['mh_PHQ_S'].apply(lambda x: 0 if x <=4 else 1 if x <=9 else 2)

    X = df.drop(['mh_PHQ_S', 'mh_PHQ_S_grouped'], axis=1)
    y = df['mh_PHQ_S_grouped']

    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED)

    X_train_selected, selected_features, selector = select_features(X_train, y_train, config.SELECTED_FEATURES)
    X_test_selected = selector.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # DNN 모델 학습
    arch_layers = [1024, 512, 256, 128]
    model = build_dnn(X_train_scaled.shape[1], arch_layers, optimizer='Adam')

    dnn_start = time.time()
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    dnn_end = time.time()

    y_pred_dnn = model.predict(X_test_scaled).argmax(axis=1)
    metrics_dnn = evaluate_model(y_test, y_pred_dnn)

    model_records = [{
        'Model': 'DNN',
        'Accuracy': metrics_dnn['accuracy'],
        'Precision': metrics_dnn['precision'],
        'Recall': metrics_dnn['recall'],
        'F1_Score': metrics_dnn['f1_score'],
        'Fit_Time_sec': dnn_end - dnn_start
    }]

    param_records = [{
        'Model': 'DNN',
        'Hyperparameters': f"layers={len(model.layers)}, params={model.count_params()}, optimizer=Adam, batch_size=32, epochs=50, validation_split=0.2"
    }]

    # 다른 전통 모델 학습
    traditional_models = train_traditional_models(X_train_scaled, y_train, seed=config.SEED)

    for name, clf in traditional_models.items():
        t0 = time.time()
        clf.fit(X_train_scaled, y_train)
        t1 = time.time()
        y_pred = clf.predict(X_test_scaled)
        metrics = evaluate_model(y_test, y_pred)

        model_records.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'Fit_Time_sec': t1 - t0
        })

        default_params = clf.__class__().get_params()
        current_params = clf.get_params()
        hyperparams_detail = {key: (current_params[key] if current_params[key] != default_params[key] else 'Default')
                              for key in current_params}

        param_records.append({
            'Model': name,
            'Hyperparameters': str(hyperparams_detail)
        })
            # 1D-CNN 학습을 위한 데이터 reshape (samples, timesteps, features=1) 또는 (samples, features, 1)
    X_train_cnn = np.expand_dims(X_train_scaled, axis=2)  # shape: (n, features, 1)
    X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

    cnn1d_model = build_1dcnn(input_shape=X_train_cnn.shape[1:])
    cnn1d_start = time.time()
    cnn1d_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    cnn1d_end = time.time()

    y_pred_cnn = cnn1d_model.predict(X_test_cnn).argmax(axis=1)
    metrics_cnn = evaluate_model(y_test, y_pred_cnn)

    # TabNet용 입력 데이터 float32로 변환 필요!
    X_train_tabnet = X_train_scaled.astype(np.float32)
    X_test_tabnet = X_test_scaled.astype(np.float32)

    tabnet_clf = TabNetClassifier(seed=config.SEED, verbose=0)
    t0 = time.time()
    tabnet_clf.fit(
        X_train_tabnet, y_train.values, 
        eval_set=[(X_test_tabnet, y_test.values)], 
        patience=10, max_epochs=100, batch_size=1024
    )
    t1 = time.time()
    y_pred_tabnet = tabnet_clf.predict(X_test_tabnet)
    metrics_tabnet = evaluate_model(y_test, y_pred_tabnet)

    model_records.append({
        'Model': 'TabNet',
        'Accuracy': metrics_tabnet['accuracy'],
        'Precision': metrics_tabnet['precision'],
        'Recall': metrics_tabnet['recall'],
        'F1_Score': metrics_tabnet['f1_score'],
        'Fit_Time_sec': t1 - t0
    })

    tabnet_params = tabnet_clf.get_params()
    param_records.append({
        'Model': 'TabNet',
        'Hyperparameters': str(tabnet_params)
    })
        # 결과 기록
    model_records.append({
        'Model': '1D-CNN',
        'Accuracy': metrics_cnn['accuracy'],
        'Precision': metrics_cnn['precision'],
        'Recall': metrics_cnn['recall'],
        'F1_Score': metrics_cnn['f1_score'],
        'Fit_Time_sec': cnn1d_end - cnn1d_start
    })
    param_records.append({
        'Model': '1D-CNN',
        'Hyperparameters': f"Conv1D(128,3)->Conv1D(64,3)->Dense(64), optimizer=Adam, batch_size=32, epochs=50, validation_split=0.2"
    })


    # 모델 성능과 하이퍼파라미터 Excel로 저장
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    excel_filename = os.path.join(report_dir, f"{now}_model_detail_report.xlsx")

    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        pd.DataFrame(model_records).to_excel(writer, sheet_name='Model Performance', index=False)
        pd.DataFrame(param_records).to_excel(writer, sheet_name='Model Hyperparameters', index=False)

    print(f"\n모델별 성능과 하이퍼파라미터가 '{excel_filename}' 파일로 저장되었습니다.")

    # 모델별 F1 Score 그래프 출력
    plt.figure(figsize=(10, 5))
    plt.bar([record['Model'] for record in model_records], [record['F1_Score'] for record in model_records], color='skyblue')
    plt.title("Model-wise F1 Score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()