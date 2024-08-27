import unittest
import pandas as pd
from src.data.data_acquisition import fetch_data, save_data
import os
import tempfile

class TestDataAcquisition(unittest.TestCase):
    def test_fetch_data(self):
        # 使用一个公开的CSV数据集URL进行测试
        url = "https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv"
        df = fetch_data(url)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_save_data(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.csv")
            save_data(df, output_path)
            self.assertTrue(os.path.exists(output_path))
            loaded_df = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(df, loaded_df)

if __name__ == '__main__':
    unittest.main()
