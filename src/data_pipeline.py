import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DiabetesData:
    RENAME = {
        "s1": "tc",
        "s2": "ldl",
        "s3": "hdl",
        "s4": "tch",
        "s5": "ltg",
        "s6": "glu",
        "target": "progression",
    }

    DESCRIPTIONS = {
        "age": "Age (years)",
        "sex": "Sex (coded 1 / 2)",
        "bmi": "Body mass index (kg/m²)",
        "bp": "Average blood pressure (mm Hg)",
        "tc": "Total serum cholesterol (mg/dL)",
        "ldl": "Low-density lipoproteins (mg/dL)",
        "hdl": "High-density lipoproteins (mg/dL)",
        "tch": "Total cholesterol / HDL ratio",
        "ltg": "Log of serum triglycerides",
        "glu": "Blood sugar level (mg/dL)",
        "progression": "Disease progression one year after baseline",
    }

    TARGET = "progression"

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.raw = None
        self.df = None
        self.splits = {}

    def load(self):
        bunch = datasets.load_diabetes(as_frame=True, scaled=False)
        self.raw = bunch.frame.copy()
        self.df = self.raw.rename(columns=self.RENAME)
        self.df["sex"] = self.df["sex"].astype(int)
        return self.df

    @property
    def features(self):
        return [c for c in self.df.columns if c != self.TARGET]

    def dictionary(self):
        original = {new: old for old, new in self.RENAME.items()}
        return pd.DataFrame(
            {
                "column": list(self.DESCRIPTIONS),
                "sklearn_name": [original.get(c, c) for c in self.DESCRIPTIONS],
                "meaning": list(self.DESCRIPTIONS.values()),
            }
        ).set_index("column")

    def summary(self):
        stats = self.df.describe().T
        stats["skew"] = self.df.skew()
        stats["cv_%"] = 100 * stats["std"] / stats["mean"]
        return stats.round(2)

    def quality_report(self):
        q1 = self.df.quantile(0.25)
        q3 = self.df.quantile(0.75)
        iqr = q3 - q1
        outside = (self.df < q1 - 1.5 * iqr) | (self.df > q3 + 1.5 * iqr)
        return pd.DataFrame(
            {
                "dtype": self.df.dtypes.astype(str),
                "missing": self.df.isna().sum(),
                "unique": self.df.nunique(),
                "iqr_outliers": outside.sum(),
                "min": self.df.min(),
                "max": self.df.max(),
            }
        )

    def duplicate_count(self):
        return int(self.df.duplicated().sum())

    def outlier_rows(self, column):
        q1, q3 = self.df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (self.df[column] < q1 - 1.5 * iqr) | (self.df[column] > q3 + 1.5 * iqr)
        return self.df.loc[mask]

    def target_correlations(self):
        return (
            self.df.corr()[self.TARGET]
            .drop(self.TARGET)
            .sort_values(key=np.abs, ascending=False)
            .round(3)
        )

    def split(self, train=0.75, val=0.10, test=0.15):
        if not np.isclose(train + val + test, 1.0):
            raise ValueError("Split fractions must add up to 1.")
        X = self.df[self.features]
        y = self.df[self.TARGET]
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, train_size=train, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=test / (val + test), random_state=self.random_state
        )
        self.splits = {
            "train": (X_train, y_train),
            "validation": (X_val, y_val),
            "test": (X_test, y_test),
        }
        return self.splits

    def split_overview(self):
        rows = []
        for name, (X, y) in self.splits.items():
            rows.append(
                {
                    "split": name,
                    "rows": len(X),
                    "share_%": round(100 * len(X) / len(self.df), 1),
                    "target_mean": round(y.mean(), 1),
                    "target_std": round(y.std(), 1),
                    "bmi_mean": round(X["bmi"].mean(), 2),
                }
            )
        return pd.DataFrame(rows).set_index("split")

    def subset(self, columns):
        return {name: (X[columns], y) for name, (X, y) in self.splits.items()}

    def risk_threshold(self, quantile=0.75):
        return float(self.splits["train"][1].quantile(quantile))

    @staticmethod
    def to_risk_labels(y, threshold):
        return (np.asarray(y) >= threshold).astype(int)
