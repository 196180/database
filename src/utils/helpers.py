import os
import logging
from datetime import datetime

def setup_logging(log_file="big_data_model.log"):
    """设置日志"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )

def get_timestamp():
    """获取当前时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_directory_if_not_exists(directory):
    """如果目录不存在,则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def validate_file_path(file_path):
    """验证文件路径是否存在"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
