from flask import Flask, jsonify
from data.data_acquisition import fetch_data, save_data
from data.data_preprocessing import preprocess_data
from models.model import BigDataModel
from utils.helpers import setup_logging, get_timestamp, create_directory_if_not_exists
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process_data():
    try:
        main()
        return jsonify({"status": "success", "message": "Data processed and model trained successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
    
    # 数据可视化
    logging.info("Generating data visualizations...")
    df = pd.read_csv(processed_data_path)
    create_directory_if_not_exists("visualizations")
    
    # 相关性热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    
    # 目标变量分布
    plt.figure(figsize=(8, 6))
    df['target'].value_counts().plot(kind='bar')
    plt.title('目标变量分布')
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('visualizations/target_distribution.png')
    plt.close()
    
    # 模型训练
    logging.info("Training model...")
    X = df.drop("target", axis=1)
    y = df["target"]
    
    model = BigDataModel()
    model.train(X, y)
    
    # 保存模型
    model_dir = "data/models"
    create_directory_if_not_exists(model_dir)
    model_path = os.path.join(model_dir, f"big_data_model_{get_timestamp()}.joblib")
    model.save(model_path)
    
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
