from data.data_acquisition import fetch_data, save_data
from data.data_preprocessing import preprocess_data
from models.model import BigDataModel
import pandas as pd

def main():
    # 数据获取
    source_url = "https://example.com/bigdata.csv"
    raw_data_path = "data/raw/bigdata.csv"
    df = fetch_data(source_url)
    if df is not None:
        save_data(df, raw_data_path)
    
    # 数据预处理
    processed_data_path = "data/processed/preprocessed_data.csv"
    preprocess_data(raw_data_path, processed_data_path)
    
    # 模型训练
    df = pd.read_csv(processed_data_path)
    X = df.drop("target_column", axis=1)
    y = df["target_column"]
    
    model = BigDataModel()
    model.train(X, y)
    
    # 保存模型
    model_path = "data/models/big_data_model.joblib"
    model.save(model_path)

if __name__ == "__main__":
    main()
