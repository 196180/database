from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

class BigDataModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def train(self, X, y):
        """
        训练模型
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型准确率: {accuracy:.2f}")
    
    def save(self, model_path):
        """
        保存模型
        """
        joblib.dump(self.model, model_path)
        print(f"模型已保存到 {model_path}")
    
    @staticmethod
    def load(model_path):
        """
        加载模型
        """
        return joblib.load(model_path)

if __name__ == "__main__":
    # 加载预处理后的数据
    df = pd.read_csv("data/processed/preprocessed_data.csv")
    
    # 准备特征和目标变量
    X = df.drop("target_column", axis=1)
    y = df["target_column"]
    
    # 创建并训练模型
    model = BigDataModel()
    model.train(X, y)
    
    # 保存模型
    model.save("data/models/big_data_model.joblib")
