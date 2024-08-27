import unittest
import pandas as pd
import numpy as np
from src.models.model import BigDataModel
import tempfile
import os

class TestBigDataModel(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的测试数据集
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
        self.y = pd.Series(np.random.randint(0, 2, 100))

    def test_train(self):
        model = BigDataModel()
        model.train(self.X, self.y)
        self.assertIsNotNone(model.model)

    def test_save_and_load(self):
        model = BigDataModel()
        model.train(self.X, self.y)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.joblib")
            model.save(model_path)
            self.assertTrue(os.path.exists(model_path))

            loaded_model = BigDataModel.load(model_path)
            self.assertIsNotNone(loaded_model)

            # 检查原始模型和加载的模型的预测是否一致
            original_predictions = model.model.predict(self.X)
            loaded_predictions = loaded_model.predict(self.X)
            np.testing.assert_array_equal(original_predictions, loaded_predictions)

if __name__ == '__main__':
    unittest.main()
