from data.data_acquisition import fetch_data, save_data
from data.data_preprocessing import preprocess_data
from models.model import BigDataModel
from utils.helpers import setup_logging, get_timestamp, create_directory_if_not_exists
import pandas as pd
import logging
import os

def main():
    setup_logging()
    
    # 数据获取
    source_url = "https://example.com/bigdata.csv"
    raw_data_dir = "data/raw"
    create_directory_if_not_exists(raw_data_dir)
    raw_data_path = os.path.join(raw_data_dir, f"bigdata_{get_timestamp()}.csv")
    
    logging.info("Fetching data...")
    df = fetch_data(source_url)
    if df is not None:
        save_data(df, raw_data_path)
    
    # 数据预处理
    processed_data_dir = "data/processed"
    create_directory_if_not_exists(processed_data_dir)
    processed_data_path = os.path.join(processed_data_dir, f"preprocessed_data_{get_timestamp()}.csv")
    
    logging.info("Preprocessing data...")
    preprocess_data(raw_data_path, processed_data_path)
    
    # 模型训练
    logging.info("Training model...")
    df = pd.read_csv(processed_data_path)
    X = df.drop("target_column", axis=1)
    y = df["target_column"]
    
    model = BigDataModel()
    model.train(X, y)
    
    # 保存模型
    model_dir = "data/models"
    create_directory_if_not_exists(model_dir)
    model_path = os.path.join(model_dir, f"big_data_model_{get_timestamp()}.joblib")
    model.save(model_path)
    
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()
