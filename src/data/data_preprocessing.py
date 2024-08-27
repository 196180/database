import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    """
    对数据进行预处理
    """
    # 读取数据
    df = pd.read_csv(input_path)
    
    # 处理缺失值
    df = df.dropna()
    
    # 标准化数值特征
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # 保存处理后的数据
    df.to_csv(output_path, index=False)
    print(f"预处理后的数据已保存到 {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/bigdata.csv"
    output_path = "data/processed/preprocessed_data.csv"
    preprocess_data(input_path, output_path)
