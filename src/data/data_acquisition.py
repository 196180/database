import pandas as pd

def fetch_data(source_url):
    """
    从给定的URL获取数据
    """
    try:
        df = pd.read_csv(source_url)
        return df
    except Exception as e:
        print(f"获取数据时出错: {e}")
        return None

def save_data(df, output_path):
    """
    将数据保存到指定路径
    """
    df.to_csv(output_path, index=False)
    print(f"数据已保存到 {output_path}")

if __name__ == "__main__":
    source_url = "https://example.com/bigdata.csv"
    output_path = "data/raw/bigdata.csv"
    
    df = fetch_data(source_url)
    if df is not None:
        save_data(df, output_path)
