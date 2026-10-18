from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class DiabetesVisualizer:
    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    INK_SOFT = "#52514e"
    GRID = "#e6e5e0"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}
    DIVERGING = [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]]
    SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]

    def __init__(self, image_dir="images", export=True):
        self.image_dir = Path(image_dir)
        self.export = export
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._register_template()

    def _register_template(self):
        template = go.layout.Template()
        template.layout = go.Layout(
            font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=self.INK),
            title=dict(font=dict(size=19, color=self.INK), x=0.01, xanchor="left"),
            paper_bgcolor=self.SURFACE,
            plot_bgcolor=self.SURFACE,
            colorway=self.SERIES,
            margin=dict(l=60, r=30, t=80, b=55),
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=self.GRID),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right", title_text=""),
            xaxis=dict(gridcolor=self.GRID, zeroline=False, linecolor=self.GRID, tickfont=dict(color=self.INK_SOFT)),
            yaxis=dict(gridcolor=self.GRID, zeroline=False, linecolor=self.GRID, tickfont=dict(color=self.INK_SOFT)),
        )
        pio.templates["diabetes"] = template
        pio.templates.default = "diabetes"

    def save(self, fig, name, width=1100, height=600):
        if not self.export:
            return fig
        try:
            fig.write_image(self.image_dir / f"{name}.png", width=width, height=height, scale=2)
        except Exception as error:
            print(f"Skipped export of {name}.png ({error.__class__.__name__}). Install kaleido to save images.")
        return fig

    @staticmethod
    def _grid_shape(n, cols):
        return int(np.ceil(n / cols)), cols

    def target_distribution(self, y, threshold=None):
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=y, nbinsx=30, marker=dict(color=self.SERIES[0], line=dict(color=self.SURFACE, width=2)),
                hovertemplate="Progression %{x}<br>Patients %{y}<extra></extra>", name="patients",
            )
        )
        fig.add_vline(x=float(np.mean(y)), line=dict(color=self.INK_SOFT, dash="dot", width=2),
                      annotation_text=f"mean {np.mean(y):.0f}", annotation_position="top")
        if threshold is not None:
            fig.add_vline(x=threshold, line=dict(color=self.SERIES[7], width=2),
                          annotation_text=f"high-risk cut-off {threshold:.0f}", annotation_position="top right")
        fig.update_layout(title="Disease progression one year after baseline", showlegend=False,
                          xaxis_title="Progression score", yaxis_title="Patients", bargap=0.04)
        return self.save(fig, "target_distribution")

    def histograms(self, df, columns, cols=4):
        rows, cols = self._grid_shape(len(columns), cols)
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=columns, vertical_spacing=0.12, horizontal_spacing=0.06)
        for i, col in enumerate(columns):
            r, c = divmod(i, cols)
            fig.add_trace(
                go.Histogram(x=df[col], nbinsx=25, marker=dict(color=self.SERIES[0], line=dict(color=self.SURFACE, width=1)),
                             name=col, hovertemplate=f"{col} %{{x}}<br>count %{{y}}<extra></extra>"),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(title="Feature distributions", showlegend=False, height=260 * rows, bargap=0.03)
        return self.save(fig, "feature_histograms", height=260 * rows)

    def scatter_against_target(self, df, columns, target, cols=4):
        rows, cols = self._grid_shape(len(columns), cols)
        corr = df[columns].corrwith(df[target])
        titles = [f"{c}  (r = {corr[c]:.2f})" for c in columns]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.06)
        for i, col in enumerate(columns):
            r, c = divmod(i, cols)
            fig.add_trace(
                go.Scatter(x=df[col], y=df[target], mode="markers",
                           marker=dict(size=7, color=self.SERIES[0], opacity=0.45, line=dict(color=self.SURFACE, width=1)),
                           hovertemplate=f"{col} %{{x}}<br>{target} %{{y}}<extra></extra>"),
                row=r + 1, col=c + 1,
            )
            if df[col].nunique() > 2:
                slope, intercept = np.polyfit(df[col], df[target], 1)
                xs = np.linspace(df[col].min(), df[col].max(), 50)
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines",
                                         line=dict(color=self.SERIES[1], width=2), hoverinfo="skip"),
                              row=r + 1, col=c + 1)
        fig.update_layout(title=f"Each feature against {target}", showlegend=False, height=280 * rows)
        return self.save(fig, "feature_scatter", height=280 * rows)

    def correlation_heatmap(self, df):
        corr = df.corr().round(2)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        shown = corr.mask(mask)
        fig = go.Figure(
            go.Heatmap(
                z=shown.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=self.DIVERGING,
                text=shown.map(lambda v: "" if pd.isna(v) else f"{v:.2f}").values, texttemplate="%{text}",
                textfont=dict(size=11), xgap=2, ygap=2, colorbar=dict(title="r", thickness=12),
                hovertemplate="%{y} vs %{x}<br>r = %{z}<extra></extra>",
            )
        )
        fig.update_layout(title="Correlation matrix", yaxis=dict(autorange="reversed", showgrid=False),
                          xaxis=dict(showgrid=False), height=640, width=760)
        return self.save(fig, "correlation_matrix", width=760, height=640)

    def standardized_boxplots(self, df, columns):
        z = (df[columns] - df[columns].mean()) / df[columns].std()
        fig = go.Figure()
        for col in columns:
            fig.add_trace(go.Box(y=z[col], name=col, marker=dict(color=self.SERIES[0], size=5),
                                 line=dict(width=1.5), boxpoints="outliers", fillcolor="rgba(42,120,214,0.15)"))
        fig.update_layout(title="Spread and outliers on a common scale (z-scores)", showlegend=False,
                          yaxis_title="Standard deviations from mean")
        return self.save(fig, "outlier_boxplots")

    def polynomial_family(self, models, X, y, feature="bmi"):
        xs = pd.DataFrame({feature: np.linspace(X[feature].min(), X[feature].max(), 300)})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[feature], y=y, mode="markers", name="training data",
                                 marker=dict(size=7, color="#b4b2a9", opacity=0.6)))
        for i, (name, model) in enumerate(models.items()):
            fig.add_trace(go.Scatter(x=xs[feature], y=model.predict(xs), mode="lines", name=name,
                                     line=dict(width=2.5, color=self.SERIES[i % len(self.SERIES)])))
        fig.update_layout(title="Polynomial degrees 0 to 5 fitted on BMI", xaxis_title="BMI", yaxis_title="Progression")
        return self.save(fig, "bmi_polynomial_family")

    def metric_comparison(self, table, metric, title=None, name=None):
        fig = go.Figure()
        for split in ["train", "validation"]:
            fig.add_trace(go.Bar(x=table.index, y=table[metric, split], name=split,
                                 marker=dict(color=self.SPLIT_COLORS[split], line=dict(color=self.SURFACE, width=2)),
                                 hovertemplate=f"%{{x}}<br>{split} {metric}: %{{y:.3f}}<extra></extra>"))
        fig.update_layout(barmode="group", title=title or f"{metric}: training vs validation",
                          yaxis_title=metric, bargap=0.25, bargroupgap=0.05)
        return self.save(fig, name or f"compare_{metric.lower().replace('%', 'pct')}")

    def fit_over_splits(self, model, splits, feature="bmi", label="BMI"):
        lo = min(X[feature].min() for X, _ in splits.values())
        hi = max(X[feature].max() for X, _ in splits.values())
        xs = pd.DataFrame({feature: np.linspace(lo, hi, 200)})
        line = model.predict(xs)
        names = list(splits)
        fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=[f"{n} (n = {len(splits[n][0])})" for n in names])
        for i, name in enumerate(names):
            X, y = splits[name]
            fig.add_trace(go.Scatter(x=X[feature], y=y, mode="markers", name=name,
                                     marker=dict(size=8, color=self.SPLIT_COLORS[name], opacity=0.7,
                                                 line=dict(color=self.SURFACE, width=1)),
                                     hovertemplate=f"{label} %{{x}}<br>actual %{{y}}<extra>{name}</extra>"),
                          row=1, col=i + 1)
            fig.add_trace(go.Scatter(x=xs[feature], y=line, mode="lines", name="model fit", showlegend=i == 0,
                                     line=dict(color=self.INK, width=2.5), hoverinfo="skip"),
                          row=1, col=i + 1)
            fig.update_xaxes(title_text=label, row=1, col=i + 1)
        fig.update_yaxes(title_text="Progression", row=1, col=1)
        fig.update_layout(title="Chosen BMI model over train, validation and test data")
        return self.save(fig, "chosen_model_fit", width=1300, height=520)

    def actual_vs_predicted(self, y_true, y_pred, title, name, threshold=None):
        lo, hi = float(min(np.min(y_true), np.min(y_pred))), float(max(np.max(y_true), np.max(y_pred)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction",
                                 line=dict(color=self.INK_SOFT, dash="dot", width=2)))
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="patients",
                                 marker=dict(size=9, color=self.SERIES[0], opacity=0.75, line=dict(color=self.SURFACE, width=1)),
                                 hovertemplate="actual %{x:.0f}<br>predicted %{y:.0f}<extra></extra>"))
        if threshold is not None:
            fig.add_vline(x=threshold, line=dict(color=self.SERIES[7], width=1.5, dash="dash"))
            fig.add_hline(y=threshold, line=dict(color=self.SERIES[7], width=1.5, dash="dash"),
                          annotation_text="high-risk cut-off", annotation_position="bottom right")
        fig.update_layout(title=title, xaxis_title="Actual progression", yaxis_title="Predicted progression")
        return self.save(fig, name, width=800, height=620)

    def error_by_band(self, table, title, name):
        labels = [str(b) for b in table.index]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Mean absolute error", "Mean bias (predicted - actual)"],
                            horizontal_spacing=0.1)
        fig.add_trace(go.Bar(x=labels, y=table["MAE"], marker=dict(color=self.SERIES[0]),
                             hovertemplate="%{x}<br>MAE %{y}<extra></extra>"), row=1, col=1)
        colors = [self.SERIES[7] if v < 0 else self.SERIES[0] for v in table["mean_bias"]]
        fig.add_trace(go.Bar(x=labels, y=table["mean_bias"], marker=dict(color=colors),
                             hovertemplate="%{x}<br>bias %{y}<extra></extra>"), row=1, col=2)
        fig.update_xaxes(title_text="Actual progression band")
        fig.update_layout(title=title, showlegend=False)
        return self.save(fig, name, width=1200, height=500)

    def importance(self, series, title, name):
        s = series.iloc[::-1]
        colors = [self.SERIES[7] if v < 0 else self.SERIES[0] for v in s.values]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker=dict(color=colors),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title=title, xaxis_title="Weight", height=420)
        return self.save(fig, name, width=900, height=420)

    def confusion_matrix(self, cm, title, name):
        labels = ["not high risk", "high risk"]
        fig = go.Figure(go.Heatmap(
            z=cm.values, x=labels, y=labels, colorscale=self.SEQUENTIAL, showscale=False,
            text=cm.values, texttemplate="%{text}", textfont=dict(size=22), xgap=3, ygap=3,
            hovertemplate="actual %{y}<br>predicted %{x}<br>%{z} patients<extra></extra>",
        ))
        fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual",
                          yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(showgrid=False))
        return self.save(fig, name, width=620, height=520)

    def leaderboard(self, table, metric="MAE"):
        order = table.sort_values((metric, "validation"), ascending=metric != "R2").index
        fig = go.Figure()
        for split in ["train", "validation"]:
            fig.add_trace(go.Scatter(
                x=table.loc[order, (metric, split)], y=order, mode="markers", name=split,
                marker=dict(size=13, color=self.SPLIT_COLORS[split], line=dict(color=self.SURFACE, width=2)),
                hovertemplate=f"%{{y}}<br>{split} {metric}: %{{x:.2f}}<extra></extra>",
            ))
        for model in order:
            fig.add_shape(type="line", x0=table.loc[model, (metric, "train")], x1=table.loc[model, (metric, "validation")],
                          y0=model, y1=model, line=dict(color=self.GRID, width=3), layer="below")
        fig.update_layout(title=f"Part 3 leaderboard: {metric} on train vs validation", xaxis_title=metric,
                          yaxis=dict(autorange="reversed"), height=460)
        return self.save(fig, "part3_leaderboard", width=1000, height=460)
