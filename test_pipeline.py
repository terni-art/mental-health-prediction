import ast
import copy
import json
import shutil
import subprocess
import sys

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal
from sklearn.base import clone

import pipeline
from modeling import (
    MODEL_NAMES,
    ModelPreprocessor,
    make_pipeline,
    prepare_after_eda,
    run_training,
    summarize_cv,
)


@pytest.fixture
def config():
    data = yaml.safe_load((pipeline.PROJECT_DIR / "params.yaml").read_text())
    data["data"].update({"directory": "data/raw", "mode": "split", "stage": "raw"})
    data["preprocessing"].update({"enabled": True, "rare_min_count": 2})
    data["cv"]["n_splits"] = 2
    data["model"] = {"name": "logistic_regression", "params": {"C": 1, "max_iter": 300}}
    data["search"].update({"enabled": False, "grid": {"C": [0.1, 1]}})
    data["mlflow"]["experiment"] = "pipeline-tests"
    return data


@pytest.fixture
def dataset():
    rng = np.random.default_rng(56)
    count = 120
    student = np.arange(count) % 2 == 0
    target = (rng.normal(size=count) > 0).astype(int)
    return pd.DataFrame(
        {
            "id": np.arange(count),
            "Name": [f"Person {i}" for i in range(count)],
            "Age": rng.integers(18, 60, count),
            "Gender": np.where(student, "Female", "Male"),
            "Working Professional or Student": np.where(
                student, "Student", "Working Professional"
            ),
            "Academic Pressure": np.where(student, 2 + target, np.nan),
            "CGPA": np.where(student, 7 + rng.random(count), np.nan),
            "Study Satisfaction": np.where(student, 3, np.nan),
            "Work Pressure": np.where(student, np.nan, 2 + target),
            "Job Satisfaction": np.where(student, np.nan, 3),
            "Financial Stress": rng.random(count) + target,
            "City": np.where(student, "A", "B"),
            "Sleep Duration": np.where(student, "5-6 hours", "More than 8 hours"),
            "Dietary Habits": np.where(student, "Healthy", "Moderate"),
            "Depression": target,
        }
    )


def save_source(project, data):
    source = project / "data/raw"
    source.mkdir(parents=True)
    data.to_csv(source / "train.csv", index=False)
    test = data.iloc[:12].drop(columns="Depression").copy()
    test["id"] += 10000
    test.to_csv(source / "test.csv", index=False)
    return source


def test_fixed_split_and_raw_split_are_preserved(config, dataset, tmp_path):
    source = save_source(tmp_path, dataset.iloc[:90])
    dataset.iloc[90:].to_csv(source / "val.csv", index=False)
    config["data"]["mode"] = "fixed"
    config["preprocessing"]["enabled"] = False
    train, val, _ = pipeline.prepare(config, tmp_path)
    assert_frame_equal(train, pd.read_csv(source / "train.csv"))
    assert_frame_equal(val, pd.read_csv(source / "val.csv"))
    dataset.to_csv(source / "train.csv", index=False)
    config["data"]["mode"] = "split"
    first_train, first_val, _ = pipeline.prepare(config, tmp_path)
    second_train, second_val, _ = pipeline.prepare(config, tmp_path)
    assert_frame_equal(first_train, second_train)
    assert_frame_equal(first_val, second_val)
    assert not set(first_train["id"]) & set(first_val["id"])


def test_duplicates_keep_val_unchanged(config, dataset, tmp_path):
    repeated = dataset.iloc[:24].copy()
    repeated["id"] += 1000
    save_source(tmp_path, pd.concat([dataset, repeated], ignore_index=True))
    train, val, info = pipeline.prepare(config, tmp_path)
    raw = pd.read_csv(tmp_path / "data/raw/train.csv")
    _, expected_val = pipeline.train_test_split(
        raw, test_size=0.2, random_state=56, stratify=raw["Depression"]
    )
    assert val["id"].tolist() == expected_val["id"].tolist()
    features = [column for column in raw if column not in ["id", "Name", "Depression"]]
    unique = pd.concat([val, train]).drop_duplicates(subset=features)
    assert len(unique) == len(train) + len(val)
    assert info["duplicates_removed"] > 0


def test_preprocessed_data_is_not_cleaned_twice(config, dataset, tmp_path):
    prepared = prepare_after_eda(dataset)
    save_source(tmp_path, prepared.iloc[:90])
    prepared.iloc[90:].to_csv(tmp_path / "data/raw/val.csv", index=False)
    config["data"].update({"mode": "fixed", "stage": "preprocessed"})
    train, _, _ = pipeline.prepare(config, tmp_path)
    assert_frame_equal(train, pd.read_csv(tmp_path / "data/raw/train.csv"))


def test_transformer_does_not_learn_from_val():
    train = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "Financial Stress": [1.0, 3.0, np.nan],
            "City": ["A", "A", "B"],
            "CGPA": [np.nan, 8, 9],
        }
    )
    val = pd.DataFrame(
        {
            "id": [4, 5],
            "Financial Stress": [999.0, np.nan],
            "City": ["new", "A"],
            "CGPA": [np.nan, 10],
        }
    )
    transformer = ModelPreprocessor("lightgbm", rare_min_count=2).fit(train)
    medians = transformer.ordinary_medians_.copy()
    output = transformer.transform(val)
    assert output.iloc[1]["Financial Stress"] == 2
    assert output.iloc[0]["City"] == "Unknown"
    assert pd.isna(output.iloc[0]["CGPA"])
    assert transformer.ordinary_medians_.equals(medians)
    assert transformer.frequent_["City"] == ["A"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_all_models_and_serialization(name, config, dataset, tmp_path):
    params = {
        "dummy": {},
        "logistic_regression": {"max_iter": 300},
        "random_forest": {"n_estimators": 8, "max_depth": 3, "n_jobs": 1},
        "catboost": {"iterations": 8, "depth": 3, "thread_count": 1},
        "lightgbm": {"n_estimators": 8, "min_child_samples": 2, "n_jobs": 1},
    }
    config["model"] = {"name": name, "params": params[name]}
    features = prepare_after_eda(dataset).drop(columns="Depression")
    estimator = clone(make_pipeline(config, features)).fit(
        features.iloc[:90], dataset["Depression"].iloc[:90]
    )
    expected = estimator.predict_proba(features.iloc[90:])
    path = tmp_path / "model.joblib"
    joblib.dump(estimator, path)
    np.testing.assert_allclose(
        joblib.load(path).predict_proba(features.iloc[90:]), expected
    )
    if name == "logistic_regression":
        features.iloc[90:].to_csv(tmp_path / "features.csv", index=False)
        code = "import joblib,pandas as pd,numpy as np,sys; np.save(sys.argv[3],joblib.load(sys.argv[1]).predict_proba(pd.read_csv(sys.argv[2])))"
        subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(path),
                str(tmp_path / "features.csv"),
                str(tmp_path / "probabilities.npy"),
            ],
            cwd=pipeline.PROJECT_DIR,
            check=True,
        )
        np.testing.assert_allclose(np.load(tmp_path / "probabilities.npy"), expected)


def test_grid_search_fits_preprocessing_inside_folds(config, dataset, monkeypatch):
    config["search"]["enabled"] = True
    features = prepare_after_eda(dataset).drop(columns="Depression")
    fit_sizes = []
    original = ModelPreprocessor.fit

    def observed_fit(self, X, y=None):
        fit_sizes.append(len(X))
        result = original(self, X, y)
        assert (
            self.ordinary_medians_["Financial Stress"] == X["Financial Stress"].median()
        )
        return result

    monkeypatch.setattr(ModelPreprocessor, "fit", observed_fit)
    _, predictions, folds, metrics, best, results = run_training(
        make_pipeline(config, features), features, dataset["Depression"], config
    )
    assert len(results) == 2
    assert fit_sizes == [60, 60, 60, 60, 120]
    assert predictions["row"].nunique() == len(dataset)
    assert len(folds) == 2 and best["C"] in [0.1, 1]
    assert 0 <= metrics["cv_mean_accuracy"] <= 1


def test_threshold_rescores_oof_without_fit():
    oof = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "target": [0, 1, 0, 1],
            "probability": [0.4, 0.6, 0.4, 0.6],
        }
    )
    _, before = summarize_cv(oof, 0.5)
    _, after = summarize_cv(oof, 0.7)
    assert before["cv_mean_accuracy"] == 1
    assert after["cv_mean_accuracy"] == 0.5
    assert before["cv_mean_roc_auc"] == after["cv_mean_roc_auc"]


def test_dvc_tracks_data_model_and_evaluation(config, dataset, tmp_path, monkeypatch):
    from dvc.repo import Repo

    save_source(tmp_path, dataset)
    for name in ["pipeline.py", "modeling.py", "requirements.txt", "dvc.yaml"]:
        shutil.copy2(pipeline.PROJECT_DIR / name, tmp_path / name)
    for name in [
        "data/processed",
        "artifacts/models",
        "artifacts/reports/pipeline",
        "artifacts/final",
    ]:
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / "placeholder.txt").write_text("synthetic fixture\n")
    config_path = tmp_path / "params.yaml"
    config_path.write_text(yaml.safe_dump(config))

    def status():
        with Repo(tmp_path) as repo:
            return repo.status()

    def commit():
        with Repo(tmp_path) as repo:
            repo.commit(force=True)

    monkeypatch.chdir(tmp_path)
    Repo.init(tmp_path, no_scm=True).close()
    commit()
    assert status() == {}
    config["model"]["params"]["C"] = 10
    config_path.write_text(yaml.safe_dump(config))
    assert set(status()) == {"train"}
    commit()
    config["evaluation"]["threshold"] = 0.7
    config_path.write_text(yaml.safe_dump(config))
    assert set(status()) == {"evaluate", "predict"}
    commit()
    dataset.iloc[:-1].to_csv(tmp_path / "data/raw/train.csv", index=False)
    assert set(status()) == {"prepare"}


def test_full_pipeline_logs_and_exports_model(config, dataset, tmp_path):
    config["search"]["enabled"] = True
    config["evaluation"]["importance"].update(
        {"permutation": True, "max_samples": 20, "n_repeats": 2}
    )
    save_source(tmp_path, dataset)
    for name in ["pipeline.py", "modeling.py", "requirements.txt", "dvc.yaml"]:
        shutil.copy2(pipeline.PROJECT_DIR / name, tmp_path / name)
    pipeline.prepare(config, tmp_path)
    metadata = pipeline.train(config, tmp_path)
    metrics = pipeline.evaluate(config, tmp_path)
    output = tmp_path / "artifacts/reports/pipeline"
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["training_run_id"] == metadata["training_run_id"]
    assert all(
        (output / f"{name}.png").exists()
        for name in ["confusion_matrix", "roc_curve", "pr_curve", "feature_importance"]
    )
    assert (output / "permutation_importance.png").exists()
    assert (tmp_path / "artifacts/models/grid_results.csv").exists()
    assert set(f"val_{metric}" for metric in METRICS_REQUIRED) <= set(metrics)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(evaluation["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.params["model.name"] == "logistic_regression"
    assert "train_sha256" in run.data.tags
    exported = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
    data = pd.read_csv(tmp_path / "data/processed/val.csv").drop(columns="Depression")
    np.testing.assert_allclose(
        exported.predict_proba(data),
        joblib.load(tmp_path / "artifacts/models/model.joblib").predict_proba(data),
    )
    stale = copy.deepcopy(config)
    stale["model"]["params"]["C"] = 10
    with pytest.raises(ValueError, match="другими настройками"):
        pipeline.evaluate(stale, tmp_path)


def test_cleaning_aliases_missing_values_and_idempotence(dataset):
    data = dataset.iloc[:8].copy()
    data["Degree"] = [
        " bsc ",
        "B.Sc",
        "bpharm",
        " M.Ed ",
        "PhD",
        None,
        "",
        "Odd degree",
    ]
    data["Sleep Duration"] = [
        "<5 hours",
        " 6 hours ",
        "6 - 8 hours",
        "More than 8 hours",
        None,
        "30 hours",
        "8-6 hours",
        "text",
    ]
    data["Dietary Habits"] = [
        "healthy",
        "More Healthy",
        "Less than Healthy",
        "Unhealthy",
        None,
        "",
        "junk",
        "Moderate",
    ]
    cleaned = prepare_after_eda(data)
    assert cleaned["Degree"].iloc[:5].tolist() == [
        "BSc",
        "BSc",
        "B.Pharm",
        "M.Ed",
        "PhD",
    ]
    assert cleaned["Sleep Duration"].iloc[:4].tolist() == [
        "Less than 5 hours",
        "6 hours",
        "6-8 hours",
        "More than 8 hours",
    ]
    assert cleaned["Sleep Duration"].iloc[5:].tolist() == ["Unknown"] * 3
    assert cleaned["Dietary Habits"].iloc[:4].tolist() == [
        "Healthy",
        "More Healthy",
        "Less Healthy",
        "Unhealthy",
    ]
    assert pd.isna(cleaned["Degree"].iloc[5]) and pd.isna(cleaned["Degree"].iloc[6])
    assert pd.isna(cleaned["Dietary Habits"].iloc[4]) and pd.isna(
        cleaned["Sleep Duration"].iloc[4]
    )
    assert "Name" not in cleaned
    assert_frame_equal(prepare_after_eda(cleaned), cleaned)


@pytest.mark.parametrize("name", ["catboost", "logistic_regression"])
def test_rare_missing_and_unknown_are_separate(name):
    data = pd.DataFrame(
        {"id": range(8), "City": ["A"] * 5 + ["B", None, "Unknown"], "Age": [20.0] * 8}
    )
    transformer = ModelPreprocessor(name, rare_min_count=5).fit(data)
    features = pd.DataFrame(
        {
            "id": [9, 10, 11, 12, 13],
            "City": ["A", "B", "new", None, "Unknown"],
            "Age": [30.0] * 5,
        }
    )
    out = transformer.transform(features)
    if name == "catboost":
        assert out["City"].tolist() == ["A", "Rare", "Unknown", "Missing", "Unknown"]
    else:
        for row, category in enumerate(["A", "Rare", "Unknown", "Missing", "Unknown"]):
            assert out.loc[row, f"City_{category}"] == 1
    assert transformer.frequent_["City"] == ["A"]


@pytest.mark.parametrize(
    "name,label",
    [("catboost", "CatBoost"), ("logistic_regression", "Logistic Regression")],
)
def test_notebook_and_module_preprocessing_match(name, label, dataset):
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.validation import check_is_fitted
    from modeling import ROLE_COLUMNS

    notebook = json.loads((pipeline.PROJECT_DIR / "2_models.ipynb").read_text())
    code = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "class NotebookPreprocessor(" in "".join(cell["source"])
    )
    tree = ast.parse(code)
    definitions = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ],
        type_ignores=[],
    )
    namespace = {
        "np": np,
        "pd": pd,
        "BaseEstimator": BaseEstimator,
        "TransformerMixin": TransformerMixin,
        "StandardScaler": StandardScaler,
        "check_is_fitted": check_is_fitted,
        "ROLE_SPECIFIC_COLUMNS": ROLE_COLUMNS,
    }
    exec(compile(definitions, "notebook_preprocessing", "exec"), namespace)
    features = prepare_after_eda(dataset).drop(columns="Depression")
    features["Financial Stress"] = np.nan
    features.loc[1, "City"] = None
    notebook_transformer = namespace["NotebookPreprocessor"](label, 5).fit(
        features.iloc[:90]
    )
    module_transformer = ModelPreprocessor(name, 5).fit(features.iloc[:90])
    val = features.iloc[90:].copy()
    val.loc[val.index[0], "City"] = "new city"
    assert_frame_equal(
        notebook_transformer.transform(val), module_transformer.transform(val)
    )


@pytest.mark.parametrize(
    "name,canonical,param",
    [("catboost", "catboost", "depth"), ("logreg", "logistic_regression", "C")],
)
def test_cli_model_and_yaml_profiles(name, canonical, param, tmp_path, monkeypatch):
    config_path = tmp_path / "params.yaml"
    original = (pipeline.PROJECT_DIR / "params.yaml").read_text()
    config_path.write_text(original)
    config = pipeline.read_config(config_path, name)
    assert config["model"]["name"] == canonical and param in config["model"]["params"]
    seen = []
    monkeypatch.setattr(pipeline, "run", lambda value: seen.append(value))
    monkeypatch.setattr(
        sys,
        "argv",
        ["pipeline.py", "run", "--model", name, "--config", str(config_path)],
    )
    pipeline.main()
    assert seen[0]["model"] == config["model"]
    assert config_path.read_text() == original


def test_legacy_flat_parameters(config, tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = pipeline.read_config(path)
    assert loaded["model"] == config["model"]


def test_missing_csv_is_reported_before_training(
    config, dataset, tmp_path, monkeypatch
):
    source = save_source(tmp_path, dataset)
    (source / "test.csv").unlink()
    monkeypatch.setattr(
        pipeline,
        "train",
        lambda *args: pytest.fail("Обучение не должно начинаться без test.csv"),
    )
    with pytest.raises(FileNotFoundError, match="Скачайте данные Kaggle вручную"):
        pipeline.run(config, tmp_path)


def test_deleted_experiment_is_restored(config, tmp_path):
    pipeline.setup_mlflow(config, tmp_path)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment"])
    client.delete_experiment(experiment.experiment_id)
    pipeline.setup_mlflow(config, tmp_path)
    assert client.get_experiment(experiment.experiment_id).lifecycle_stage == "active"


@pytest.mark.parametrize("name", ["catboost", "logistic_regression"])
def test_run_preserves_reports_and_exports_kaggle_submission(
    name, config, dataset, tmp_path, monkeypatch
):
    source = save_source(tmp_path, dataset)
    for filename in ["pipeline.py", "modeling.py", "requirements.txt", "dvc.yaml"]:
        shutil.copy2(pipeline.PROJECT_DIR / filename, tmp_path / filename)
    config["model"] = {
        "name": name,
        "params": {"iterations": 8, "depth": 3, "thread_count": 1}
        if name == "catboost"
        else {"C": 1, "max_iter": 300},
    }
    notebook_report = tmp_path / "artifacts/reports/notebook_search/figure.png"
    notebook_report.parent.mkdir(parents=True)
    notebook_report.write_bytes(b"existing notebook report")
    fit_sizes = []
    original = ModelPreprocessor.fit

    def observed(self, X, y=None):
        fit_sizes.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(ModelPreprocessor, "fit", observed)
    result = pipeline.run(config, tmp_path)
    assert fit_sizes[-1] == len(dataset)
    assert notebook_report.read_bytes() == b"existing notebook report"
    output = tmp_path / "artifacts/final"
    submission = pd.read_csv(output / result["submission"])
    test = pd.read_csv(source / "test.csv")
    assert submission.columns.tolist() == ["id", "Depression"]
    assert submission["id"].tolist() == test["id"].tolist()
    assert len(submission) == len(test) and set(submission["Depression"].unique()) <= {
        0,
        1,
    }
    assert not list(output.glob("*probabilit*"))
    fitted = joblib.load(output / "model.joblib")
    prepared_test = pd.read_csv(tmp_path / "data/processed/test.csv")
    expected = fitted.predict_proba(prepared_test)
    exported = mlflow.sklearn.load_model(f"runs:/{result['run_id']}/model")
    np.testing.assert_allclose(exported.predict_proba(prepared_test), expected)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.tags["training_run_id"] == result["training_run_id"]
    assert run.data.tags["evaluation_run_id"] == result["evaluation_run_id"]
    assert not any(key.startswith("val_") for key in run.data.metrics)


METRICS_REQUIRED = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]
