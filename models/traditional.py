from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

def train_traditional_models(X, y, seed=42):
    models = {
        'LogisticRegression': LogisticRegression(random_state=seed, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed),
        'XGBoost': xgb.XGBClassifier(random_state=seed, eval_metric='mlogloss'),
        'SVM': SVC(probability=True, random_state=seed),
        'AdaBoost': AdaBoostClassifier(random_state=seed),
        'LightGBM': lgb.LGBMClassifier(random_state=seed),
    }
    for model in models.values():
        model.fit(X, y)
    return models
