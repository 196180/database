import unittest
import pandas as pd
import tempfile
import os
from src.data.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        # 创建一个测试数据集
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, 6, 7, 8],
            'C': ['a', 'b', 'c', 'd']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            output_path = os.path.join(tmpdir, "output.csv")
            
            # 保存输入数据
            df.to_csv(input_path, index=False)
            
            # 运行预处理
            preprocess_data(input_path, output_path)
            
            # 检查输出文件是否存在
            self.assertTrue(os.path.exists(output_path))
            
            # 读取预处理后的数据
            processed_df = pd.read_csv(output_path)
            
            # 检查是否删除了缺失值
            self.assertEqual(len(processed_df), 3)
            
            # 检查数值列是否已标准化
            self.assertTrue((processed_df['A'].mean() - 0).abs() < 1e-6)
            self.assertTrue((processed_df['A'].std() - 1).abs() < 1e-6)
            self.assertTrue((processed_df['B'].mean() - 0).abs() < 1e-6)
            self.assertTrue((processed_df['B'].std() - 1).abs() < 1e-6)

if __name__ == '__main__':
    unittest.main()
