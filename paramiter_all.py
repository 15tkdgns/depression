import pandas as pd
import inspect
from datetime import datetime
import os

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
# DNN, 1DCNN은 keras/tensorflow import 필요 (여기선 구조상 예시 생략)

# 각 모델별 meta 정보 (튜닝팁, 권장값 등 필요한 만큼만 입력. 빈칸은 자동처리)
meta = {
    "XGBClassifier": {
        "learning_rate": ("0.01~0.3", "0.05~0.2", "학습률이 작을수록 안정적, n_estimators 조정", "항상 사용"),
        "scale_pos_weight": ("1~불균형비", "불균형비", "불균형 데이터에서만 사용", "불균형 분류에서만"),
        "gamma": ("0~5", "0~2", "클수록 분할 덜함, 과적합 방지", "과적합 방지시"),
        # ... (추가 파라미터 메타)
    },
    "LGBMClassifier": {
        "learning_rate": ("0.01~0.2", "0.05~0.15", "낮게, n_estimators 늘리기", "항상 사용"),
    },
    "RandomForestClassifier": {
        "n_estimators": ("10~1000", "100~300", "많을수록 안정적, 너무 많으면 느림", "항상 사용"),
    },
    "AdaBoostClassifier": {
        "n_estimators": ("10~1000", "100~200", "너무 크면 과적합, 속도↓", "항상 사용"),
    },
    "LogisticRegression": {
        "penalty": ("'l2', 'l1', 'elasticnet', 'none'", "'l2'", "L1은 변수선택, L2는 일반", "항상 사용"),
        "l1_ratio": ("0~1", "0.5", "L1/L2 혼합 비율", "penalty='elasticnet' and solver='saga'일 때만"),
    },
    "SVC": {
        "kernel": ("'linear','poly','rbf','sigmoid','precomputed'", "'rbf'", "비선형 문제에 주로 사용", "항상 사용"),
        "degree": ("2~5", "3", "poly 커널일 때만 적용", "kernel='poly'일 때만"),
        "gamma": ("'scale','auto',float", "'scale'", "rbf/poly/sigmoid에서만 의미 있음", "kernel in ['rbf','poly','sigmoid']일 때만"),
    },
    "PolynomialFeatures": {
        "degree": ("2~5", "2,3", "너무 크면 차원폭발", "항상 사용"),
    },
    "SelectKBest": {
        "score_func": ("f_classif, chi2, mutual_info_classif, f_regression", "문제 유형에 맞게", "분류/회귀에 맞춰 선택", "항상 사용"),
        "k": ("1~전체 특성수", "전체 특성의 5~50%", "최적 성능 위해 검증점수로 조절", "항상 사용"),
    },
    "SMOTE": {
        "sampling_strategy": ("0.1~1.0", "0.5~1.0", "1.0=동등, 0.5=언더샘플링", "불균형 클래스일 때만"),
        "k_neighbors": ("1~20", "3~8", "소수 클래스 샘플이 적으면 낮게", "소수 클래스 샘플 수보다 작아야 함"),
    },
    "TabNetClassifier": {
        "n_d": ("8~128", "32, 64", "표현력↑, 너무 크면 과적합 가능", "항상 사용"),
        "n_a": ("8~128", "n_d와 동일", "n_d와 동일하게 맞추는 것이 일반적", "항상 사용"),
        "n_steps": ("3~10", "5~7", "많을수록 깊이↑, 속도↓", "항상 사용"),
        "gamma": ("1.0~2.0", "1.2~1.5", "feature 재사용 정도 제어", "항상 사용"),
        "lambda_sparse": ("1e-5~1e-1", "0.0001~0.001", "클수록 sparsity↑, 희소성 조절", "항상 사용"),
        "optimizer_fn": ("함수", "Adam", "커스텀 optimizer 쓸 때만", "커스텀 옵티마이저 지정 시"),
        "scheduler_fn": ("함수", "None", "러닝레이트 스케줄러 쓸 때만", "커스텀 스케줄러 지정 시"),
        "batch_size": ("128~4096", "512~2048", "GPU 메모리 상황에 맞게", "fit()에서만 사용"),
        "patience": ("3~30", "10~20", "EarlyStopping용", "fit()에서만 사용"),
        "max_epochs": ("10~1000", "100~300", "학습 데이터 크기 따라 조절", "fit()에서만 사용"),
    },
}

# 모델 클래스와 human-readable name 매핑
models = [
    (XGBClassifier, "XGBClassifier"),
    (LGBMClassifier, "LGBMClassifier"),
    (RandomForestClassifier, "RandomForestClassifier"),
    (AdaBoostClassifier, "AdaBoostClassifier"),
    (LogisticRegression, "LogisticRegression"),
    (SVC, "SVC"),
    (PolynomialFeatures, "PolynomialFeatures"),
    (SelectKBest, "SelectKBest"),
    (SMOTE, "SMOTE"),
    (TabNetClassifier, "TabNetClassifier"),
]

rows = []

for cls, mdl_name in models:
    # 생성자 파라미터
    sig = inspect.signature(cls.__init__)
    for name, param in sig.parameters.items():
        if name == 'self': continue
        default = param.default if param.default is not inspect.Parameter.empty else ""
        rng, rec, tip, cond = meta.get(mdl_name, {}).get(name, ("", "", "", ""))
        rows.append([mdl_name, name, default, rng, rec, tip, cond])

    # fit 파라미터 (TabNet, XGBoost, LGBM 등만 유의미한 fit 인자 별도 있음)
    if hasattr(cls, 'fit'):
        fit_sig = inspect.signature(cls.fit)
        for name, param in fit_sig.parameters.items():
            if name in ['self', 'X', 'y']: continue
            # fit의 파라미터는 이름 겹치면 중복 방지
            fit_row = [mdl_name+"(fit)", name, param.default if param.default is not inspect.Parameter.empty else "", "", "", "", ""]
            # meta에 있으면 보강
            m = meta.get(mdl_name, {}).get(name, None)
            if m:
                fit_row[3:7] = m
            # 중복 파라미터 생략
            if name not in [r[1] for r in rows if r[0] == mdl_name]:
                rows.append(fit_row)

# DNN/1DCNN, keras 등은 자동추출 어려우므로 수작업 추가 가능

df = pd.DataFrame(rows, columns=["model","Hyperparameter","default","Range","Recommended Value","튜닝팁","Usage Condition"])

os.makedirs("reports", exist_ok=True)
now = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"reports/{now}_all_hyperparameters_full.xlsx"
df.to_excel(save_path, index=False)
print(f"\n✅ 모든 파라미터 엑셀 '{save_path}' 파일로 저장 완료!")
