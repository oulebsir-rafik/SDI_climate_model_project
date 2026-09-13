import numpy as np
import pandas as pd

from src.intake_forecast.model import fit_model


class RecordingStubModel:
    def __init__(self):
        self.seen_X = None
        self.seen_y = None

    def fit(self, X, y):
        self.seen_X = X
        self.seen_y = y
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_fit_model_drops_rows_with_nan_target_but_keeps_nan_features():
    X = pd.DataFrame({"feat": [1.0, np.nan, 3.0, 4.0]})
    y = pd.Series([10.0, 20.0, np.nan, 40.0])

    model = RecordingStubModel()
    fit_model(model, X, y)

    assert list(model.seen_y.index) == [0, 1, 3]
    assert list(model.seen_y) == [10.0, 20.0, 40.0]
    # row index 1 has a NaN feature but a valid target, so it must be kept, NaN untouched
    assert np.isnan(model.seen_X.loc[1, "feat"])
