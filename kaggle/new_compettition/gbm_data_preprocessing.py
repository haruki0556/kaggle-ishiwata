from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy as np

#カラムの翻訳
new_columns = {
"Booking_ID":"予約ID",
"no_of_adults":"大人の数",
"no_of_children":"子供の数",
"no_of_weekend_nights":"週末の宿泊数",
"no_of_week_nights":"平日の宿泊数",
"type_of_meal_plan":"食事プランの種類",
"required_car_parking_space":"駐車場の要不要",
"room_type_reserved":"予約された部屋のタイプ",
"lead_time":"予約日から到着日までの日数",
"arrival_year":"到着年",
"arrival_month":"到着月",
"arrival_date":"到着日",
"market_segment_type":"マーケットセグメント",
"repeated_guest":"リピーター客",
"no_of_previous_cancellations":"過去のキャンセル回数",
"no_of_previous_bookings_not_canceled":"過去のキャンセルされなかった予約数",
"avg_price_per_room":"一泊あたりの宿泊費",
"no_of_special_requests":"特別リクエストの数",
"booking_status":"予約ステータス",
}

#食事プランの種類の中身を日本語に変換
meal_plan_mapping = {
    'Meal Plan 1': 'プラン1',
    'Meal Plan 2': 'プラン2',
    'Meal Plan 3': 'プラン3',
    'Not Selected': '選択なし'
}

room_type_mapping = {
'Room_Type 1':"タイプ1", 
'Room_Type 4':'タイプ4',
'Room_Type 2':"タイプ2",
'Room_Type 6':"タイプ6",
'Room_Type 5':"タイプ5",
'Room_Type 7':"タイプ7", 
'Room_Type 3':"タイプ3",
}

segment_mapping = {
    "Offline":"オフライン予約",
    "Online":"オンライン予約",
    "Corporate":"法人予約",
    "Aviation":"航空",
    "Complementary":"招待",
}

status_mapping = {
    "Not_Canceled":0,
    "Canceled":1,
}

columns_mappings_dict = {
    "食事プランの種類":meal_plan_mapping,
    "予約された部屋のタイプ":room_type_mapping,
    "マーケットセグメント":segment_mapping,
    "予約ステータス":status_mapping,
}

categorical_cols = ["食事プランの種類", 
                    "予約された部屋のタイプ", 
                    'マーケットセグメント',
                    "到着年",
                    "到着月",
                    ] 


def gbm_preprocessing_pipeline(df,new_columns,columns_mappings_dict,categorical_cols):
    #カラム名の翻訳
    df_renamed = df.rename(columns=new_columns)

    #カテゴリの翻訳
    for col, maps in columns_mappings_dict.items():
        df_renamed[col] = df_renamed[col].replace(maps)

    #欠損値の削除
    df_renamed.dropna(subset=['到着月'], inplace=True)
    df_renamed.dropna(subset=['到着年'], inplace=True)

    #不要な列を削除
    df_renamed.drop(columns=["予約ID","到着日"],axis=1,inplace=True)
    
    #train_test_split
    X = df_renamed.drop(columns=["予約ステータス"],axis=1)
    y = df_renamed["予約ステータス"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,
        random_state=42,
        stratify=y, # 分類問題では非常に重要
    )

    #連続値の非線形変換
    log_cols = ['一泊あたりの宿泊費', '予約日から到着日までの日数', '過去のキャンセル回数']
    
    for col in log_cols:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    #整数型に変換
    X_train["到着月"] = X_train['到着月'].astype(int)
    X_train["到着年"] = X_train['到着年'].astype(int)
    X_test["到着月"] = X_test['到着月'].astype(int)
    X_test["到着年"] = X_test['到着年'].astype(int)

    # データ型を 'category' に変換
    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')


    return X_train,X_test,y_train,y_test







