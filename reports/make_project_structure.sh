#!/bin/bash

# 프로젝트 루트명 지정
ROOT_DIR="my_depression_project2"

# 디렉토리 생성
mkdir -p $ROOT_DIR/data $ROOT_DIR/preprocess $ROOT_DIR/models $ROOT_DIR/evaluations $ROOT_DIR/reports $ROOT_DIR/configs

# data
touch $ROOT_DIR/data/raw_data.csv

# preprocess
touch $ROOT_DIR/preprocess/__init__.py
cat << EOF > $ROOT_DIR/preprocess/data_loader.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)
EOF

cat << EOF > $ROOT_DIR/preprocess/preprocessing.py
# 결측치 처리, 이상치 제거, 레이블 인코딩, 다항특징 생성 등
def preprocess(df):
    pass  # 실제 함수 구현 필요
EOF

cat << EOF > $ROOT_DIR/preprocess/feature_selection.py
from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=11):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features, selector
EOF

# models
touch $ROOT_DIR/models/__init__.py
cat << EOF > $ROOT_DIR/models/dnn.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def build_dnn(input_dim, arch_layers, optimizer):
    model = Sequential()
    model.add(Dense(arch_layers[0], activation='relu', input_dim=input_dim, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    for units in arch_layers[1:]:
        model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax', dtype='float32'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
EOF

cat << EOF > $ROOT_DIR/models/traditional.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

def train_traditional_models(X, y, seed=42):
    models = {
        'LogisticRegression': LogisticRegression(random_state=seed, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': xgb.XGBClassifier(random_state=seed, eval_metric='mlogloss', use_label_encoder=False),
        'SVM': SVC(probability=True, random_state=seed),
        'AdaBoost': AdaBoostClassifier(random_state=seed),
        'LightGBM': lgb.LGBMClassifier(random_state=seed),
    }
    for model in models.values():
        model.fit(X, y)
    return models
EOF

touch $ROOT_DIR/models/utils.py

# evaluations
touch $ROOT_DIR/evaluations/__init__.py
cat << EOF > $ROOT_DIR/evaluations/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
EOF

touch $ROOT_DIR/evaluations/metrics.py

# reports
touch $ROOT_DIR/reports/__init__.py
cat << EOF > $ROOT_DIR/reports/generate_report.py
import pandas as pd

def save_report(data_dict, filename):
    with pd.ExcelWriter(filename) as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
EOF

# configs
cat << EOF > $ROOT_DIR/configs/config.py
DATA_PATH = 'data/raw_data.csv'
SEED = 42
SELECTED_FEATURES = 11
EOF

# run_experiment.py
cat << EOF > $ROOT_DIR/run_experiment.py
from preprocess.data_loader import load_data
from preprocess.preprocessing import preprocess
from preprocess.feature_selection import select_features
from models.dnn import build_dnn
from models.traditional import train_traditional_models
from evaluations.evaluation import evaluate_model
from configs import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("실험 코드 작성 위치")

if __name__ == "__main__":
    main()
EOF

echo "✅ my_depression_project2 디렉토리 및 파일 구조가 생성되었습니다."
