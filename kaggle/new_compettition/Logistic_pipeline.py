# modeling_pipeline.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def fit_predict_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, C: float = 1.0):
      
    # 1. モデルの初期化と学習（fit）
    model = LogisticRegression(
        random_state=42, 
        C=C, 
        solver='liblinear', # データセットのサイズに応じてsolverを選択
        max_iter=1000 # 収束しない場合のために反復回数を増加
    )
    
    print("--- モデル学習開始 ---")
    model.fit(X_train, y_train)
    print("--- 学習完了 ---")
    
    # 2. 予測（predict）
    y_pred = model.predict(X_test)
    
    # 3. 評価
    # 詳細な分類レポートを作成
    report = classification_report(y_test, y_pred, target_names=['キャンセルなし', 'キャンセルあり'])
    
    # 係数（洞察）の抽出
    # X_trainのカラム名とモデルの係数を結びつける
    coefficients_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    })
    coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
    
    # 結果を返す
    return model, report, coefficients_df

# --- ファイルが直接実行された場合のテストコード ---
if __name__ == '__main__':
    print("このファイルはモジュールとして設計されています。Notebookなどで利用してください。")
    # ここにダミーデータ生成と関数呼び出しのテストコードを追加できます。