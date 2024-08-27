import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from data.data_preprocessing import preprocess_data
from models.model import BigDataModel
from utils.helpers import setup_logging, get_timestamp, create_directory_if_not_exists
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        create_directory_if_not_exists(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"status": "success", "message": "File uploaded successfully", "path": file_path}), 200
    return jsonify({"status": "error", "message": "File type not allowed"}), 400

@app.route('/process', methods=['POST'])
def process_data():
    file_path = request.json.get('file_path')
    if not file_path:
        return jsonify({"status": "error", "message": "No file path provided"}), 400
    
    try:
        # 数据预处理
        processed_data_dir = "data/processed"
        create_directory_if_not_exists(processed_data_dir)
        processed_data_path = os.path.join(processed_data_dir, f"preprocessed_data_{get_timestamp()}.csv")
        
        preprocess_data(file_path, processed_data_path)
        
        # 模型训练
        df = pd.read_csv(processed_data_path)
        X = df.drop("target", axis=1)
        y = df["target"]
        
        model = BigDataModel()
        model.train(X, y)
        
        # 保存模型
        model_dir = "data/models"
        create_directory_if_not_exists(model_dir)
        model_path = os.path.join(model_dir, f"big_data_model_{get_timestamp()}.joblib")
        model.save(model_path)
        
        # 生成可视化
        visualizations = generate_visualizations(df)
        
        return jsonify({
            "status": "success", 
            "message": "Data processed and model trained successfully",
            "visualizations": visualizations
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def generate_visualizations(df):
    # 相关性热力图
    corr = df.corr()
    fig_heatmap = px.imshow(corr, labels=dict(color="Correlation"), x=corr.columns, y=corr.columns)
    fig_heatmap.update_layout(title='Feature Correlation Heatmap')
    
    # 目标变量分布
    fig_target = px.bar(df['target'].value_counts().reset_index(), x='index', y='target', labels={'index': 'Target', 'target': 'Count'})
    fig_target.update_layout(title='Target Variable Distribution')
    
    # 特征重要性（假设我们有特征重要性的数据）
    feature_importance = model.model.feature_importances_
    fig_importance = px.bar(x=df.columns[:-1], y=feature_importance, labels={'x': 'Features', 'y': 'Importance'})
    fig_importance.update_layout(title='Feature Importance')
    
    return {
        'heatmap': fig_heatmap.to_json(),
        'target_distribution': fig_target.to_json(),
        'feature_importance': fig_importance.to_json()
    }

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory('visualizations', filename)

if __name__ == "__main__":
    setup_logging()
    app.run(host='0.0.0.0', port=8000, debug=True)
