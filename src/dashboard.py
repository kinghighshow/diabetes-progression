import html
import json
import webbrowser
from pathlib import Path

import numpy as np


class RiskDashboard:
    LABELS = {
        "age": ("Age", "years"),
        "sex": ("Sex", "dataset code"),
        "bmi": ("Body mass index", "kg/m²"),
        "bp": ("Blood pressure", "mm Hg"),
        "hdl": ("HDL cholesterol", "mg/dL"),
        "tch": ("Total chol. / HDL", "ratio"),
        "ltg": ("Triglycerides", "log scale"),
        "glu": ("Blood sugar", "mg/dL"),
    }
    STEPS = {"bmi": 0.1, "tch": 0.1, "ltg": 0.01}

    def __init__(self, regressors, classifier, X_train, threshold):
        self.regressors = regressors
        self.classifier = classifier
        self.X_train = X_train
        self.threshold = float(threshold)

    @staticmethod
    def _scaler(pipeline):
        scaler = pipeline.named_steps["scale"]
        return scaler.mean_.round(6).tolist(), scaler.scale_.round(6).tolist()

    def _linear_spec(self, pipeline):
        mean, scale = self._scaler(pipeline)
        features = list(pipeline.feature_names_in_)
        reg = pipeline.named_steps["reg"]
        coef = dict(zip(pipeline.named_steps["poly"].get_feature_names_out(), reg.coef_))
        if set(coef) - set(features) - {"1"}:
            raise ValueError("Only degree 0 or 1 polynomial models can be exported.")
        return {
            "type": "linear",
            "features": features,
            "mean": mean,
            "scale": scale,
            "weights": [round(float(coef.get(f, 0.0)), 6) for f in features],
            "bias": round(float(coef.get("1", 0.0) + reg.intercept_), 6),
        }

    def _knn_spec(self, pipeline):
        mean, scale = self._scaler(pipeline)
        knn = pipeline.named_steps["knn"]
        return {
            "type": "knn",
            "features": list(pipeline.feature_names_in_),
            "mean": mean,
            "scale": scale,
            "k": int(knn.n_neighbors),
            "X": np.round(knn._fit_X, 6).tolist(),
            "y": np.asarray(knn._y, dtype=float).round(2).tolist(),
        }

    def _logistic_spec(self, pipeline):
        mean, scale = self._scaler(pipeline)
        clf = pipeline.named_steps["clf"]
        return {
            "type": "logistic",
            "features": list(pipeline.feature_names_in_),
            "mean": mean,
            "scale": scale,
            "weights": clf.coef_.ravel().round(6).tolist(),
            "bias": round(float(clf.intercept_[0]), 6),
        }

    def _spec(self, model):
        steps = model.named_steps
        if "knn" in steps:
            return self._knn_spec(model)
        if "clf" in steps:
            return self._logistic_spec(model)
        return self._linear_spec(model)

    def payload(self):
        inputs = []
        for feature in self.X_train.columns:
            label, unit = self.LABELS.get(feature, (feature, ""))
            column = self.X_train[feature]
            step = self.STEPS.get(feature, 1)
            inputs.append({
                "name": feature,
                "label": label,
                "unit": unit,
                "min": float(round(np.floor(column.min() / step) * step, 2)),
                "max": float(round(np.ceil(column.max() / step) * step, 2)),
                "value": round(round(float(column.median()) / step) * step, 2),
                "step": step,
            })
        return {
            "threshold": self.threshold,
            "inputs": inputs,
            "regressors": {name: self._spec(m) for name, m in self.regressors.items()},
            "classifier": self._spec(self.classifier),
        }

    def to_html(self):
        template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
        return template.replace("__PAYLOAD__", json.dumps(self.payload(), separators=(",", ":")))

    def save(self, path="dashboard.html"):
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path.resolve()

    def open(self, path="dashboard.html"):
        webbrowser.open(self.save(path).as_uri())

    def show(self, path="dashboard.html", height=660):
        from IPython.display import HTML, display

        saved = self.save(path)
        frame = f'<iframe srcdoc="{html.escape(self.to_html())}" style="width:100%;height:{height}px;border:0"></iframe>'
        display(HTML(frame))
        print(f"Saved to {saved}. Open it in any browser to use the dashboard outside the notebook.")
        return saved
