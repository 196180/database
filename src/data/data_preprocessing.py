import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(input_path, output_path):
    """
    对数据进行预处理
    """
    # 读取数据
    df = pd.read_csv(input_path)
    
    # 定义特征类型
    numeric_features = ['age', 'income', 'credit_score']
    categorical_features = ['education', 'occupation', 'marital_status']
    
    # 创建预处理管道
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 拟合和转换数据
    X = df.drop('target', axis=1)
    y = df['target']
    X_processed = preprocessor.fit_transform(X)
    
    # 创建特征名称
    feature_names = (numeric_features + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names(categorical_features).tolist())
    
    # 转换为DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # 添加目标变量
    X_processed_df['target'] = y
    
    # 特征工程
    X_processed_df['age_group'] = pd.cut(X_processed_df['age'], bins=[0, 18, 30, 50, 70, 100], labels=['0-18', '19-30', '31-50', '51-70', '71+'])
    X_processed_df['high_income'] = (X_processed_df['income'] > X_processed_df['income'].median()).astype(int)
    
    # 保存处理后的数据
    X_processed_df.to_csv(output_path, index=False)
    print(f"预处理后的数据已保存到 {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/bigdata.csv"
    output_path = "data/processed/preprocessed_data.csv"
    preprocess_data(input_path, output_path)
