import numpy as np
from planeclf.inference.predictor import predict_top1, predict_topk

def test_predict_axis_top1():
    class Dummy:
        def predict_proba(self, X):
            return np.array([[0.1, 0.7, 0.2],
                             [0.8, 0.1, 0.1]])
    model = Dummy()
    labels, conf = predict_top1(model, np.zeros((2, 1)), classes=["A", "B", "C"])
    assert labels == ["B", "A"]
    assert np.allclose(conf, [0.7, 0.8])

def test_predict_topk_ordering():
    class Dummy:
        def predict_proba(self, X):
            return np.array([[0.3, 0.5, 0.2]])  # 1x3
    model = Dummy()
    top3 = predict_topk(model, np.zeros((1, 1)), classes=["X", "Y", "Z"], k=3)
    assert top3[0] == [("Y", 0.5), ("X", 0.3), ("Z", 0.2)]
