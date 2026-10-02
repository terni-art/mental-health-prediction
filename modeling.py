import re
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ACADEMIC_COLUMNS = ["Academic Pressure", "CGPA", "Study Satisfaction"]
WORK_COLUMNS = ["Work Pressure", "Job Satisfaction"]
ROLE_COLUMNS = ACADEMIC_COLUMNS + WORK_COLUMNS
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
MODEL_NAMES = ["dummy", "logistic_regression", "random_forest", "catboost", "lightgbm"]


UNKNOWN_CATEGORY = "Unknown"
MISSING_CATEGORY = "Missing"
RARE_CATEGORY = "Rare"
DEGREE_ALIASES = {
    "bsc": "BSc",
    "b.sc": "BSc",
    "b.pharm": "B.Pharm",
    "bpharm": "B.Pharm",
    "b.ed": "B.Ed",
    "bed": "B.Ed",
    "m.ed": "M.Ed",
    "med": "M.Ed",
}


def normalize_category(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else pd.NA


def normalize_degree(value):
    text = normalize_category(value)
    if pd.isna(text):
        return pd.NA
    return DEGREE_ALIASES.get(text.casefold(), text)


def classify_dietary_habits(value):
    text = normalize_category(value)
    if pd.isna(text):
        return pd.NA
    aliases = {
        "healthy": "Healthy",
        "moderate": "Moderate",
        "unhealthy": "Unhealthy",
        "more healthy": "More Healthy",
        "less healthy": "Less Healthy",
        "less than healthy": "Less Healthy",
    }
    return aliases.get(text.casefold(), UNKNOWN_CATEGORY)


def classify_sleep_duration(value):
    text = normalize_category(value)
    if pd.isna(text):
        return pd.NA
    text = text.casefold()
    aliases = {
        "less than 5 hours": "Less than 5 hours",
        "<5 hours": "Less than 5 hours",
        "more than 8 hours": "More than 8 hours",
        ">8 hours": "More than 8 hours",
    }
    if text in aliases:
        return aliases[text]
    match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?\s*hours?", text)
    if match is None:
        return UNKNOWN_CATEGORY
    lower = int(match.group(1))
    upper = int(match.group(2)) if match.group(2) is not None else lower
    if not 0 <= lower <= upper <= 24:
        return UNKNOWN_CATEGORY
    return f"{lower} hours" if lower == upper else f"{lower}-{upper} hours"


def prepare_after_eda(data):
    data = data.drop(
        columns=["Name", "Academic Data Available", "Work Data Available"],
        errors="ignore",
    ).copy()
    categorical = data.select_dtypes(include=["str", "object", "category"]).columns
    for column in categorical:
        data[column] = data[column].map(normalize_category)
    data["Sleep Duration"] = data["Sleep Duration"].map(classify_sleep_duration)
    data["Dietary Habits"] = data["Dietary Habits"].map(classify_dietary_habits)
    if "Degree" in data:
        data["Degree"] = data["Degree"].map(normalize_degree)
    return data


class ModelPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, model_name="catboost", rare_min_count=5):
        self.model_name = model_name
        self.rare_min_count = rare_min_count

    def fit(self, X, y=None):
        data = X.drop(columns=["id", "Name"], errors="ignore").copy()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.columns_ = data.columns.tolist()
        self.numeric_ = data.select_dtypes(include="number").columns.tolist()
        self.categorical_ = [column for column in data if column not in self.numeric_]
        self.frequent_ = {}
        self.seen_ = {}
        self.categories_ = {}
        for column in self.categorical_:
            values = data[column].fillna(MISSING_CATEGORY).astype(str)
            self.seen_[column] = set(values)
            counts = values.value_counts()
            self.frequent_[column] = counts[
                counts >= self.rare_min_count
            ].index.tolist()
            special = {MISSING_CATEGORY, UNKNOWN_CATEGORY, RARE_CATEGORY}
            cleaned = values.where(
                values.isin(self.frequent_[column]) | values.isin(special),
                RARE_CATEGORY,
            )
            self.categories_[column] = sorted(set(cleaned) | special)

        ordinary = [column for column in self.numeric_ if column not in ROLE_COLUMNS]
        self.ordinary_medians_ = data[ordinary].median().fillna(0)
        data[ordinary] = data[ordinary].fillna(self.ordinary_medians_)
        self.missing_columns_ = [
            column for column in self.numeric_ if data[column].isna().any()
        ]
        numeric = self._numeric_data(data)
        self.medians_ = numeric.median().fillna(0)
        self.scaler_ = None
        if self.model_name == "logistic_regression":
            self.scaler_ = StandardScaler().fit(numeric.fillna(self.medians_))
        self.output_columns_ = self.transform(X).columns.tolist()
        return self

    def _numeric_data(self, data):
        numeric = data[self.numeric_].copy()
        for column in self.missing_columns_:
            numeric[column + " missing"] = data[column].isna().astype(int)
        return numeric

    def transform(self, X):
        check_is_fitted(self, "medians_")
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"В данных отсутствуют признаки: {sorted(missing)}")
        data = X[self.columns_].copy()
        ordinary = self.ordinary_medians_.index
        data[ordinary] = data[ordinary].fillna(self.ordinary_medians_)
        for column in self.categorical_:
            values = data[column].fillna(MISSING_CATEGORY).astype(str)
            special = {MISSING_CATEGORY, UNKNOWN_CATEGORY, RARE_CATEGORY}
            cleaned = values.where(
                values.isin(self.frequent_[column]) | values.isin(special),
                RARE_CATEGORY,
            )
            cleaned = cleaned.where(
                values.isin(self.seen_[column]) | values.isin(special), UNKNOWN_CATEGORY
            )
            data[column] = cleaned.astype(object)

        if self.model_name in ["dummy", "logistic_regression", "random_forest"]:
            numeric = self._numeric_data(data).fillna(self.medians_).astype(float)
            if self.scaler_ is not None:
                numeric.loc[:, :] = self.scaler_.transform(numeric)
            categorical = data[self.categorical_].copy()
            for column in self.categorical_:
                categorical[column] = pd.Categorical(
                    categorical[column], categories=self.categories_[column]
                )
            encoded = pd.get_dummies(categorical, dtype=float)
            return pd.concat([numeric, encoded], axis=1)

        if self.model_name == "lightgbm":
            for column in self.categorical_:
                data[column] = pd.Categorical(
                    data[column], categories=self.categories_[column]
                )
        return data

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


def make_pipeline(config, X):
    name = config["model"]["name"]
    seed = config["split"]["seed"]
    categorical = tuple(
        column
        for column in X
        if column not in ["id", "Name"] and not pd.api.types.is_numeric_dtype(X[column])
    )
    defaults = {
        "dummy": {"strategy": "prior"},
        "logistic_regression": {"C": 1.0, "max_iter": 2000, "solver": "lbfgs"},
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 18,
            "min_samples_leaf": 2,
            "random_state": seed,
            "n_jobs": 4,
        },
        "catboost": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "cat_features": categorical,
            "random_seed": seed,
            "thread_count": 4,
            "verbose": False,
            "allow_writing_files": False,
        },
        "lightgbm": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": seed,
            "n_jobs": 4,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        },
    }
    constructors = {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "catboost": CatBoostClassifier,
        "lightgbm": LGBMClassifier,
    }
    params = defaults[name] | config["model"]["params"]
    if name == "catboost" and isinstance(params.get("cat_features"), list):
        params["cat_features"] = tuple(params["cat_features"])
    model = constructors[name](**params)
    return Pipeline(
        [
            (
                "preprocessing",
                ModelPreprocessor(name, config["preprocessing"]["rare_min_count"]),
            ),
            ("model", model),
        ]
    )


def calculate_metrics(y, probability, threshold=0.5):
    prediction = np.asarray(probability) >= threshold
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
    }


def summarize_cv(predictions, threshold):
    rows = []
    for fold, data in predictions.groupby("fold", sort=True):
        rows.append(
            {
                "fold": int(fold),
                **calculate_metrics(data["target"], data["probability"], threshold),
            }
        )
    folds = pd.DataFrame(rows)
    summary = {f"cv_mean_{metric}": float(folds[metric].mean()) for metric in METRICS}
    summary.update(
        {f"cv_std_{metric}": float(folds[metric].std()) for metric in METRICS}
    )
    return folds, summary


def run_training(estimator, X, y, config):
    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=True,
        random_state=config["cv"]["seed"],
    )
    splits = list(cv.split(X, y))
    threshold = config["cv"]["threshold"]
    predictions = []
    fit_seconds = []
    best_params = {}
    search_results = None

    if config["search"]["enabled"]:
        grid = {
            f"model__{key}": values for key, values in config["search"]["grid"].items()
        }
        captured = []

        def scoring(model, features, target):
            probability = model.predict_proba(features)[:, 1]
            scores = calculate_metrics(target, probability, threshold)
            captured.append(
                {
                    "params": {key: model.get_params()[key] for key in grid},
                    "predictions": pd.DataFrame(
                        {
                            "row": features.index,
                            "target": np.asarray(target),
                            "probability": probability,
                        }
                    ),
                }
            )
            print(
                f"Grid CV {len(captured)}: Accuracy={scores['accuracy']:.5f}",
                flush=True,
            )
            return scores

        search = GridSearchCV(
            estimator,
            grid,
            scoring=scoring,
            refit=config["search"]["scoring"],
            cv=splits,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X, y)
        fitted = search.best_estimator_
        best_params = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }
        best_predictions = [
            item["predictions"]
            for item in captured
            if item["params"] == search.best_params_
        ]
        if len(best_predictions) != len(splits):
            raise RuntimeError(
                "Не удалось восстановить вероятности лучшего кандидата на CV"
            )
        for fold, data in enumerate(best_predictions, start=1):
            predictions.append(data.assign(fold=fold))
        search_results = pd.DataFrame(search.cv_results_)
        fit_seconds.append(
            float(search_results.loc[search.best_index_, "mean_fit_time"])
        )
    else:
        for fold, (train_index, val_index) in enumerate(splits, start=1):
            model = clone(estimator)
            start = perf_counter()
            model.fit(X.iloc[train_index], y.iloc[train_index])
            fit_seconds.append(perf_counter() - start)
            probability = model.predict_proba(X.iloc[val_index])[:, 1]
            predictions.append(
                pd.DataFrame(
                    {
                        "row": val_index,
                        "fold": fold,
                        "target": y.iloc[val_index].to_numpy(),
                        "probability": probability,
                    }
                )
            )
            print(
                f"Fold {fold}: Accuracy={calculate_metrics(y.iloc[val_index], probability, threshold)['accuracy']:.5f}",
                flush=True,
            )
        fitted = clone(estimator).fit(X, y)

    oof = pd.concat(predictions, ignore_index=True).sort_values("row")
    folds, summary = summarize_cv(oof, threshold)
    summary["cv_mean_fit_seconds"] = float(np.mean(fit_seconds))
    return fitted, oof, folds, summary, best_params, search_results
