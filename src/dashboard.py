import ipywidgets as w
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display
from plotly.subplots import make_subplots


class RiskDashboard:
    LABELS = {
        "age": "Age (years)",
        "bmi": "BMI (kg/m²)",
        "bp": "Blood pressure (mm Hg)",
        "hdl": "HDL (mg/dL)",
        "tch": "Total chol. / HDL",
        "ltg": "Triglycerides (log)",
        "glu": "Blood sugar (mg/dL)",
    }
    STEPS = {"bmi": 0.1, "tch": 0.1, "ltg": 0.01}
    INK, BLUE, RED, TRACK = "#0b0b0b", "#2a78d6", "#e34948", "#f0efec"

    def __init__(self, regressors, classifier, X_train, threshold):
        self.regressors = regressors
        self.classifier = classifier
        self.threshold = threshold
        self.features = list(X_train.columns)
        self.ranges = X_train.agg(["min", "median", "max"])
        self.controls = self._build_controls()
        self.model_picker = w.Dropdown(options=list(regressors), description="Score model",
                                       style={"description_width": "90px"}, layout=w.Layout(width="360px"))
        self.button = w.Button(description="Predict", button_style="primary", icon="stethoscope")
        self.output = w.Output()
        self.button.on_click(lambda _: self.predict())

    def _build_controls(self):
        controls = {}
        style = {"description_width": "150px"}
        for feature in self.features:
            if feature == "sex":
                controls[feature] = w.ToggleButtons(options=[("Group 1", 1), ("Group 2", 2)], description="Sex",
                                                    style={**style, "button_width": "90px"})
                continue
            lo, mid, hi = self.ranges[feature]
            step = self.STEPS.get(feature, 1.0)
            controls[feature] = w.FloatSlider(
                value=round(mid / step) * step, min=lo, max=hi, step=step,
                description=self.LABELS.get(feature, feature), style=style,
                readout_format=".2f" if step < 0.1 else ".1f" if step < 1 else ".0f",
                continuous_update=False, layout=w.Layout(width="440px"),
            )
        return controls

    def patient(self):
        return pd.DataFrame([{f: self.controls[f].value for f in self.features}])

    def _figure(self, score, probability):
        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])
        fig.add_trace(go.Indicator(
            mode="number+delta", value=score,
            number=dict(valueformat=".0f", font=dict(size=56, color=self.INK)),
            delta=dict(reference=self.threshold, valueformat=".0f", increasing=dict(color=self.RED),
                       decreasing=dict(color=self.BLUE), suffix=" vs cut-off"),
            title=dict(text="Predicted progression score"),
        ), row=1, col=1)
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=100 * probability,
            number=dict(suffix="%", valueformat=".0f", font=dict(size=44, color=self.INK)),
            title=dict(text="Probability of high risk"),
            gauge=dict(axis=dict(range=[0, 100]), bgcolor=self.TRACK, borderwidth=0,
                       bar=dict(color=self.RED if probability >= 0.5 else self.BLUE, thickness=0.35),
                       threshold=dict(line=dict(color=self.INK, width=3), thickness=0.8, value=50)),
        ), row=1, col=2)
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))
        return fig

    def _verdict(self, probability):
        flagged = probability >= 0.5
        color = self.RED if flagged else self.BLUE
        text = "Flag for follow-up: likely high risk" if flagged else "Not flagged: below the high-risk line"
        return HTML(
            f'<div style="border-left:5px solid {color};padding:8px 14px;font-size:15px;'
            f'background:{self.TRACK};color:{self.INK}"><b>{text}</b> '
            f'(high risk = score of {self.threshold:.0f} or more)</div>'
        )

    def predict(self):
        patient = self.patient()
        model = self.regressors[self.model_picker.value]
        score = float(model.predict(patient[list(model.feature_names_in_)])[0])
        probability = float(self.classifier.predict_proba(patient[self.features])[0, 1])
        with self.output:
            self.output.clear_output(wait=True)
            self._figure(score, probability).show()
            display(self._verdict(probability))
        return score, probability

    def show(self):
        left = [self.controls[f] for f in ["age", "sex", "bmi", "bp"] if f in self.controls]
        right = [self.controls[f] for f in self.features if f not in ("age", "sex", "bmi", "bp")]
        panel = w.VBox([
            w.HTML("<h3 style='margin:0'>Diabetes progression screener</h3>"
                   "<span style='color:#52514e'>Sliders are limited to the range seen in training.</span>"),
            w.HBox([w.VBox(left), w.VBox(right)]),
            w.HBox([self.model_picker, self.button]),
            self.output,
        ])
        display(panel)
        self.predict()
