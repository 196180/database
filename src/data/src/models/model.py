from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class BigDataModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    def train(self, X, y):
        """
        训练模型
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"模型性能评估:")
        print(f"准确率: {accuracy:.2f}")
        print(f"精确率: {precision:.2f}")
        print(f"召回率: {recall:.2f}")
        print(f"F1分数: {f1:.2f}")
        print(f"AUC: {auc:.2f}")
        
        # 交叉验证
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"交叉验证分数: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # 特征重要性可视化
        self.plot_feature_importance(X.columns)
        
    def plot_feature_importance(self, feature_names):
        """
        绘制特征重要性图
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("特征重要性")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
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
    X = df.drop("target", axis=1)
    y = df["target"]
    
    # 创建并训练模型
    model = BigDataModel()
    model.train(X, y)
    
    # 保存模型
    model.save("data/models/big_data_model.joblib")
