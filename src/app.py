import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from data.data_preprocessing import preprocess_data, merge_files
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
def upload_files():
    uploaded_files = request.files.getlist("file")
    if not uploaded_files:
        return jsonify({"status": "error", "message": "No files uploaded"}), 400
    
    file_paths = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            create_directory_if_not_exists(app.config['UPLOAD_FOLDER'])
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    
    if not file_paths:
        return jsonify({"status": "error", "message": "No valid files uploaded"}), 400
    
    merged_file_path = merge_files(file_paths)
    return jsonify({"status": "success", "message": "Files uploaded and merged successfully", "path": merged_file_path}), 200

@app.route('/process', methods=['POST'])
def process_data():
    file_path = request.json.get('file_path')
    selected_variables = request.json.get('selected_variables', [])
    if not file_path:
        return jsonify({"status": "error", "message": "No file path provided"}), 400
    
    try:
        # 数据预处理
        processed_data_dir = "data/processed"
        create_directory_if_not_exists(processed_data_dir)
        processed_data_path = os.path.join(processed_data_dir, f"preprocessed_data_{get_timestamp()}.csv")
        
        preprocess_data(file_path, processed_data_path, selected_variables)
        
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
        visualizations = generate_visualizations(df, selected_variables)
        
        return jsonify({
            "status": "success", 
            "message": "Data processed and model trained successfully",
            "visualizations": visualizations
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def generate_visualizations(df, selected_variables):
    visualizations = {}
    
    # 按年份分组的数据趋势
    if 'year' in df.columns:
        for var in selected_variables:
            if var in df.columns:
                fig = px.line(df.groupby('year')[var].mean().reset_index(), x='year', y=var)
                fig.update_layout(title=f'{var} Trend Over Years')
                visualizations[f'{var}_trend'] = fig.to_json()
    
    # 相关性热力图
    corr = df[selected_variables].corr()
    fig_heatmap = px.imshow(corr, labels=dict(color="Correlation"), x=corr.columns, y=corr.columns)
    fig_heatmap.update_layout(title='Feature Correlation Heatmap')
    visualizations['heatmap'] = fig_heatmap.to_json()
    
    # 目标变量分布
    if 'target' in df.columns:
        fig_target = px.bar(df['target'].value_counts().reset_index(), x='index', y='target', labels={'index': 'Target', 'target': 'Count'})
        fig_target.update_layout(title='Target Variable Distribution')
        visualizations['target_distribution'] = fig_target.to_json()
    
    # 特征重要性
    if hasattr(model.model, 'feature_importances_'):
        feature_importance = model.model.feature_importances_
        fig_importance = px.bar(x=df.columns[:-1], y=feature_importance, labels={'x': 'Features', 'y': 'Importance'})
        fig_importance.update_layout(title='Feature Importance')
        visualizations['feature_importance'] = fig_importance.to_json()
    
    return visualizations

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory('visualizations', filename)

if __name__ == "__main__":
    setup_logging()
    app.run(host='0.0.0.0', port=8000, debug=True)
