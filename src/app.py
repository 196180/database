import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from data.data_preprocessing import preprocess_data, merge_files_by_year, merge_files_by_variable
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
    
    return jsonify({"status": "success", "message": "Files uploaded successfully", "paths": file_paths}), 200

@app.route('/merge', methods=['POST'])
def merge_files():
    file_paths = request.json.get('file_paths')
    merge_type = request.json.get('merge_type')
    
    if not file_paths:
        return jsonify({"status": "error", "message": "No file paths provided"}), 400
    
    try:
        if merge_type == 'by_year':
            merged_file_path = merge_files_by_year(file_paths)
            visualizations = visualize_merge_by_year(merged_file_path)
        elif merge_type == 'by_variable':
            merged_file_path = merge_files_by_variable(file_paths)
            visualizations = visualize_merge_by_variable(merged_file_path)
        else:
            return jsonify({"status": "error", "message": "Invalid merge type"}), 400
        
        return jsonify({
            "status": "success", 
            "message": "Files merged successfully",
            "path": merged_file_path,
            "visualizations": visualizations
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def visualize_merge_by_year(file_path):
    df = pd.read_csv(file_path)
    visualizations = {}
    
    # 创建年度数据总量柱状图
    fig_yearly_data = px.bar(df.groupby('year').size().reset_index(name='count'), 
                             x='year', y='count', title='Yearly Data Count')
    visualizations['yearly_data_count'] = fig_yearly_data.to_json()
    
    # 为每个数值列创建年度趋势线图
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'year':
            fig_trend = px.line(df.groupby('year')[col].mean().reset_index(), 
                                x='year', y=col, title=f'{col} Yearly Trend')
            visualizations[f'{col}_yearly_trend'] = fig_trend.to_json()
    
    return visualizations

def visualize_merge_by_variable(file_path):
    df = pd.read_csv(file_path)
    visualizations = {}
    
    # 创建变量数据分布箱型图
    fig_distribution = px.box(df.melt(var_name='Variable', value_name='Value'), 
                              x='Variable', y='Value', title='Variable Distribution')
    visualizations['variable_distribution'] = fig_distribution.to_json()
    
    # 创建变量间相关性热力图
    corr = df.corr()
    fig_heatmap = px.imshow(corr, labels=dict(color="Correlation"), 
                            x=corr.columns, y=corr.columns, title='Variable Correlation Heatmap')
    visualizations['correlation_heatmap'] = fig_heatmap.to_json()
    
    return visualizations

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
