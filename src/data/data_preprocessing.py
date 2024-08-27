import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def merge_files(file_paths):
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_file_path = 'data/raw/merged_data.csv'
    merged_df.to_csv(merged_file_path, index=False)
    return merged_file_path

def preprocess_data(input_path, output_path, selected_variables=None):
    """
    对数据进行预处理
    """
    # 读取数据
    df = pd.read_csv(input_path)
    
    # 如果提供了选定的变量，只保留这些变量
    if selected_variables:
        df = df[selected_variables + ['target']]
    
    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # 标准化数值特征
    scaler = StandardScaler()
    numeric_columns = df_imputed.select_dtypes(include=['float64', 'int64']).columns
    df_imputed[numeric_columns] = scaler.fit_transform(df_imputed[numeric_columns])
    
    # 如果存在 'year' 列，将其转换为类别型
    if 'year' in df_imputed.columns:
        df_imputed['year'] = df_imputed['year'].astype('category')
    
    # 保存处理后的数据
    df_imputed.to_csv(output_path, index=False)
    print(f"预处理后的数据已保存到 {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/bigdata.csv"
    output_path = "data/processed/preprocessed_data.csv"
    preprocess_data(input_path, output_path)
