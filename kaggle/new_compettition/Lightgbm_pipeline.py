import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from gbm_data_preprocessing import gbm_preprocessing_pipeline,new_columns,columns_mappings_dict,categorical_cols

def train_evaluate_lgbm(df,new_columns,columns_mappings_dict,categorical_cols):
    X_train, X_test, y_train, y_test = gbm_preprocessing_pipeline(
    df, 
    new_columns, 
    columns_mappings_dict, 
    categorical_cols
    )

    # モデルのパラメータ設定（分析目的ならデフォルトに近いものでもOK）
    params = {
        'objective': 'binary',             # 目的関数: 2値分類
        'metric': 'auc',                   # 評価指標: ROC AUC
        'boosting_type': 'gbdt',           # ブースティングタイプ
        'n_estimators': 1000,              # 試行回数
        'learning_rate': 0.05,             # 学習率
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,                     # ログ出力を抑制
    }

    # モデルのインスタンス化
    lgbm_clf = lgb.LGBMClassifier(**params)

    # モデルの学習
    # 早期停止（early stopping）を利用して過学習を防ぎ、効率的に学習させます
    lgbm_clf.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)], 
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # 予測確率を取得 (クラス1の確率)
    y_pred_proba = lgbm_clf.predict_proba(X_test)[:, 1]
    # デフォルトの閾値0.5でクラスを予測
    y_pred = lgbm_clf.predict(X_test)

    # ROC AUC スコア
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {auc:.4f}\n")

    # 分類レポート
    print(classification_report(y_test, y_pred))

    # 特徴量重要度の出力（分析のヒント）
    importances = pd.Series(lgbm_clf.feature_importances_, index=X_train.columns)
    top_20_importances = importances.sort_values(ascending=False).head(20)

    print("--- 特徴量重要度 Top 20 ---")
    print(top_20_importances)

