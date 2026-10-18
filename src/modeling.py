import json
from pathlib import Path

import joblib
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


class ModelFactory:
    @staticmethod
    def univariate_polynomial(degree):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
                ("reg", LinearRegression(fit_intercept=False)),
            ]
        ).set_output(transform="pandas")

    @staticmethod
    def multivariate_polynomial(degree):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("reg", LinearRegression()),
            ]
        ).set_output(transform="pandas")

    @staticmethod
    def decision_tree(max_depth, min_samples_leaf=1, random_state=42):
        return DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
        )

    @staticmethod
    def knn(n_neighbors, weights="uniform"):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)),
            ]
        )

    @staticmethod
    def logistic(C=1.0, class_weight=None):
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)),
            ]
        )


class RegressionEvaluator:
    METRICS = {
        "R2": r2_score,
        "MAE": mean_absolute_error,
        "MAPE_%": lambda y, p: 100 * mean_absolute_percentage_error(y, p),
    }

    def __init__(self, splits):
        self.splits = splits
        self.models = {}

    def fit(self, models):
        X_train, y_train = self.splits["train"]
        for name, model in models.items():
            self.models[name] = model.fit(X_train, y_train)
        return self.models

    def scores(self, model, split):
        X, y = self.splits[split]
        pred = model.predict(X)
        return {metric: fn(y, pred) for metric, fn in self.METRICS.items()}

    def comparison_table(self, names=None, splits=("train", "validation")):
        names = names or list(self.models)
        rows = {}
        for name in names:
            row = {}
            for split in splits:
                for metric, value in self.scores(self.models[name], split).items():
                    row[(metric, split)] = value
            rows[name] = row
        table = pd.DataFrame(rows).T
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=["metric", "split"])
        table["R2", "gap"] = table["R2", "train"] - table["R2", "validation"]
        return table.sort_index(axis=1, level=0, sort_remaining=False).round(3)

    def cross_validation_table(self, names=None, folds=10, random_state=42):
        names = names or list(self.models)
        X, y = self.splits["train"]
        cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        rows = {}
        for name in names:
            result = cross_validate(
                clone(self.models[name]), X, y, cv=cv,
                scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
            )
            rows[name] = {
                "cv_MAE_mean": -result["test_MAE"].mean(),
                "cv_MAE_std": result["test_MAE"].std(),
                "cv_R2_mean": result["test_R2"].mean(),
            }
        return pd.DataFrame(rows).T.round(3)

    def best_model(self, names=None, metric="MAE", tolerance=0.0):
        names = names or list(self.models)
        val = {n: self.scores(self.models[n], "validation")[metric] for n in names}
        if metric == "R2":
            best = max(val.values())
            return next(n for n in names if val[n] >= best - tolerance)
        best = min(val.values())
        return next(n for n in names if val[n] <= best * (1 + tolerance))

    def test_report(self, name):
        return pd.DataFrame(
            {split: self.scores(self.models[name], split) for split in self.splits}
        ).T.round(3)

    def error_by_range(self, name, split="test", column=None, bins=4):
        X, y = self.splits[split]
        pred = self.models[name].predict(X)
        frame = pd.DataFrame({"actual": y.values, "predicted": pred})
        frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
        frame["ape_%"] = 100 * frame["abs_error"] / frame["actual"]
        frame["bias"] = frame["predicted"] - frame["actual"]
        key = X[column].values if column else frame["actual"]
        frame["band"] = pd.qcut(key, q=bins)
        return (
            frame.groupby("band", observed=True)
            .agg(
                n=("actual", "size"),
                MAE=("abs_error", "mean"),
                MAPE_pct=("ape_%", "mean"),
                mean_bias=("bias", "mean"),
            )
            .round(1)
        )

    def screening_table(self, threshold, names=None, split="validation"):
        names = names or list(self.models)
        X, y = self.splits[split]
        actual = (y.values >= threshold).astype(int)
        rows = {}
        for name in names:
            flagged = (self.models[name].predict(X) >= threshold).astype(int)
            rows[name] = {
                "recall": recall_score(actual, flagged, zero_division=0),
                "precision": precision_score(actual, flagged, zero_division=0),
                "missed_high_risk": int(((actual == 1) & (flagged == 0)).sum()),
            }
        return pd.DataFrame(rows).T.round(3)


class ClassificationEvaluator:
    def __init__(self, splits, threshold):
        self.threshold = threshold
        self.splits = {
            name: (X, (y.values >= threshold).astype(int)) for name, (X, y) in splits.items()
        }
        self.models = {}

    def fit(self, models):
        X_train, y_train = self.splits["train"]
        for name, model in models.items():
            self.models[name] = model.fit(X_train, y_train)
        return self.models

    def scores(self, model, split):
        X, y = self.splits[split]
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y, proba),
            "log_loss": log_loss(y, proba, labels=[0, 1]),
        }

    def comparison_table(self, splits=("train", "validation")):
        rows = {}
        for name, model in self.models.items():
            row = {}
            for split in splits:
                for metric, value in self.scores(model, split).items():
                    row[(metric, split)] = value
            rows[name] = row
        table = pd.DataFrame(rows).T
        table.columns = pd.MultiIndex.from_tuples(table.columns, names=["metric", "split"])
        return table.round(3)

    def best_model(self, metric="recall"):
        val = {n: self.scores(m, "validation")[metric] for n, m in self.models.items()}
        return max(val, key=val.get)

    def test_report(self, name):
        return pd.DataFrame(
            {split: self.scores(self.models[name], split) for split in self.splits}
        ).T.round(3)

    def confusion(self, name, split="test"):
        X, y = self.splits[split]
        pred = self.models[name].predict(X)
        return pd.crosstab(
            pd.Series(y, name="actual"), pd.Series(pred, name="predicted")
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)


class ModelInspector:
    @staticmethod
    def parameter_count(model):
        steps = dict(model.steps) if isinstance(model, Pipeline) else {}
        if "poly" in steps:
            reg = steps["reg"]
            n_terms = len(steps["poly"].get_feature_names_out())
            return n_terms + int(reg.fit_intercept)
        if isinstance(model, DecisionTreeRegressor):
            return model.tree_.node_count
        if "knn" in steps:
            return 0
        if "clf" in steps:
            clf = steps["clf"]
            return clf.coef_.size + clf.intercept_.size
        raise TypeError(f"Unsupported model: {type(model).__name__}")

    @staticmethod
    def parameter_breakdown(model):
        steps = dict(model.steps) if isinstance(model, Pipeline) else {}
        if "poly" in steps:
            names = list(steps["poly"].get_feature_names_out())
            if steps["reg"].fit_intercept:
                names = ["intercept"] + names
            return names
        if isinstance(model, DecisionTreeRegressor):
            leaves = model.get_n_leaves()
            return {
                "split_thresholds": model.tree_.node_count - leaves,
                "leaf_values": leaves,
                "depth": model.get_depth(),
            }
        if "knn" in steps:
            return {"stored_training_rows": steps["knn"].n_samples_fit_}
        if "clf" in steps:
            return {"weights": steps["clf"].coef_.size, "bias": steps["clf"].intercept_.size}
        return {}

    @staticmethod
    def raw_coefficients(model):
        coefs = model.named_steps["reg"].coef_
        scaler = model.named_steps.get("scale")
        if scaler is None:
            return coefs
        z = Polynomial([-scaler.mean_[0] / scaler.scale_[0], 1 / scaler.scale_[0]])
        expanded = sum((c * z**k for k, c in enumerate(coefs)), Polynomial([0.0]))
        return np.pad(expanded.coef, (0, len(coefs) - len(expanded.coef)))

    @staticmethod
    def polynomial_equation(model, variable="BMI", target="progression", precision=2):
        coefs = ModelInspector.raw_coefficients(model)
        terms = []
        for power, c in enumerate(coefs):
            value = f"{abs(c):.{precision}f}"
            if power == 0:
                body = value
            elif power == 1:
                body = f"{value}·{variable}"
            else:
                body = f"{value}·{variable}^{power}"
            sign = "-" if c < 0 else "+"
            terms.append((sign, body))
        first_sign, first_body = terms[0]
        text = ("-" if first_sign == "-" else "") + first_body
        for sign, body in terms[1:]:
            text += f" {sign} {body}"
        return f"{target} = {text}"

    @staticmethod
    def feature_importance(model, features):
        if isinstance(model, DecisionTreeRegressor):
            values = model.feature_importances_
        elif isinstance(model, Pipeline) and "clf" in model.named_steps:
            values = model.named_steps["clf"].coef_.ravel()
        else:
            raise TypeError("Importance is only defined here for trees and logistic models.")
        return pd.Series(values, index=features).sort_values(key=np.abs, ascending=False)


class ModelRegistry:
    def __init__(self, directory="models"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, model, name):
        path = self.directory / f"{name}.joblib"
        joblib.dump(model, path)
        return path

    def load(self, name):
        return joblib.load(self.directory / f"{name}.joblib")

    def write_metadata(self, metadata):
        path = self.directory / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2))
        return path

    def read_metadata(self):
        return json.loads((self.directory / "metadata.json").read_text())
