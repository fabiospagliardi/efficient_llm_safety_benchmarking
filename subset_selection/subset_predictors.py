import numpy as np
# from sklearn.ensemble import RandomForestRegressor


# class RandomForestSubsetPredictor:
#     """
#     Predicts full-benchmark accuracy from per-item binary responses on a subset.
#
#     X : (n_models, K) binary response matrix — one column per subset item.
#     y : (n_models,)   full-dataset accuracy for each model.
#
#     Fit on training models (where both signatures and full accuracy are known),
#     then call predict() on test models evaluated only on the subset.
#     """
#
#     def __init__(self, random_state: int = 42, **rf_kwargs) -> None:
#         self._model = RandomForestRegressor(random_state=random_state, **rf_kwargs)
#
#     def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestSubsetPredictor":
#         self._model.fit(X_train, y_train)
#         return self
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         return self._model.predict(X)


class LinearSubsetPredictor:
    """
    Placeholder predictor: estimates full-dataset performance from subset performance.

    Fits an ordinary least-squares linear model on training models
    (where both subset and full-dataset scores are known), then applies
    it to test models evaluated only on the subset.

    Inputs and outputs are scalar per-model performance values (e.g. accuracy
    or ability). Fit a separate instance for each metric.
    """

    def __init__(self):
        self._coeffs = None  # (slope, intercept) from np.polyfit

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearSubsetPredictor":
        """
        X_train : subset performance for each training model, shape (n_train,)
        y_train : full-dataset performance for each training model, shape (n_train,)
        """
        if len(X_train) < 2:
            self._coeffs = np.array([1.0, 0.0])
        else:
            self._coeffs = np.polyfit(X_train, y_train, 1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X : subset performance for each model, shape (n_models,)
        Returns predicted full-dataset performance, shape (n_models,)
        """
        if self._coeffs is None:
            raise RuntimeError("Call fit() before predict().")
        return np.polyval(self._coeffs, X)
