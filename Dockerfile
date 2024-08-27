# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 将当前目录内容复制到容器的 /app 中
COPY . /app

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 使端口 8000 可供此容器使用
EXPOSE 8000

# 定义环境变量
ENV NAME World

# 在容器启动时运行 main.py
CMD ["python", "src/main.py"]
