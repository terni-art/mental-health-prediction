import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from modeling import (
    ACADEMIC_COLUMNS,
    METRICS,
    MODEL_NAMES,
    WORK_COLUMNS,
    calculate_metrics,
    make_pipeline,
    prepare_after_eda,
    run_training,
    summarize_cv,
)

PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_SECTIONS = ["data", "split", "preprocessing", "model", "cv", "search"]


def read_config(path, model_name=None):
    with Path(path).open() as file:
        config = yaml.safe_load(file)
    for section in [*TRAINING_SECTIONS, "evaluation", "mlflow"]:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"В YAML нужен раздел {section}")
    selected = model_name or config["model"]["name"]
    aliases = {"logreg": "logistic_regression"}
    selected = aliases.get(selected, selected)
    params = config["model"]["params"]
    profile_names = {*MODEL_NAMES, "logreg"}
    if (
        params
        and set(params) <= profile_names
        and all(isinstance(value, dict) for value in params.values())
    ):
        profile = (
            "logreg"
            if selected == "logistic_regression" and "logreg" in params
            else selected
        )
        if profile not in params:
            raise ValueError(f"В model.params нет профиля для {selected}")
        params = params[profile]
    if not isinstance(params, dict):
        raise ValueError(
            "model.params должен быть словарем параметров или профилей моделей"
        )
    config["model"] = {"name": selected, "params": params}
    config["data"].setdefault("test_file", "test.csv")
    config["_config_path"] = str(Path(path).resolve())
    if config["data"]["mode"] not in ["fixed", "split"]:
        raise ValueError("data.mode должен быть fixed или split")
    if config["data"]["stage"] not in ["raw", "preprocessed"]:
        raise ValueError("data.stage должен быть raw или preprocessed")
    if config["model"]["name"] not in MODEL_NAMES:
        raise ValueError(f"Поддерживаемые модели: {MODEL_NAMES}")
    if not 0 < config["split"]["val_size"] < 1:
        raise ValueError("split.val_size должен быть между 0 и 1")
    if config["cv"]["n_splits"] < 2:
        raise ValueError("Для CV нужно не менее двух фолдов")
    for threshold in [config["cv"]["threshold"], config["evaluation"]["threshold"]]:
        if not 0 < threshold < 1:
            raise ValueError("Порог классификации должен быть между 0 и 1")
    if config["preprocessing"]["rare_min_count"] < 1:
        raise ValueError("rare_min_count должен быть положительным")
    if config["search"]["scoring"] not in METRICS:
        raise ValueError(f"Метрика поиска должна быть одной из {METRICS}")
    if config["search"]["enabled"]:
        grid = config["search"]["grid"]
        if not grid or any(
            not isinstance(values, list) or not values for values in grid.values()
        ):
            raise ValueError("В search.grid нужны непустые списки параметров")
    importance = config["evaluation"]["importance"]
    if min(importance["top_n"], importance["max_samples"], importance["n_repeats"]) < 1:
        raise ValueError("Настройки важности признаков должны быть положительными")
    return config


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        + "\n"
    )


def validate_data(data, target=None):
    required = {
        "id",
        "Sleep Duration",
        "Dietary Habits",
        *ACADEMIC_COLUMNS,
        *WORK_COLUMNS,
    }
    if target is not None:
        required.add(target)
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {sorted(missing)}")
    if data.empty:
        raise ValueError("CSV не должен быть пустым")
    if target is not None and (
        data[target].isna().any() or set(data[target].unique()) != {0, 1}
    ):
        raise ValueError(f"В {target} нужны оба класса 0 и 1 без пропусков")
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("id должен быть заполнен и уникален")


def check_sources(config, project_dir, require_test=False):
    source = project_dir / config["data"]["directory"]
    names = [config["data"]["train_file"]]
    if config["data"]["mode"] == "fixed":
        names.append(config["data"]["val_file"])
    if require_test:
        names.append(config["data"].get("test_file", "test.csv"))
    for name in names:
        if not name or not (source / name).is_file():
            raise FileNotFoundError(
                f"Нет исходного CSV: {source / str(name)}. Скачайте данные Kaggle вручную и положите файлы в {source}."
            )


def prepare(config, project_dir=PROJECT_DIR):
    check_sources(config, project_dir)
    source = project_dir / config["data"]["directory"]
    train_path = source / config["data"]["train_file"]
    data = pd.read_csv(train_path)
    target = config["data"]["target"]
    validate_data(data, target)
    source_hashes = {str(train_path): file_hash(train_path)}
    removed = 0
    if config["data"]["mode"] == "fixed":
        val_path = source / config["data"]["val_file"]
        val = pd.read_csv(val_path)
        validate_data(val, target)
        if set(data.columns) != set(val.columns):
            raise ValueError("Train и val должны иметь одинаковые столбцы")
        if set(data["id"]) & set(val["id"]):
            raise ValueError("Train и val пересекаются по id")
        train = data
        val = val[train.columns]
        source_hashes[str(val_path)] = file_hash(val_path)
    else:
        train, val = train_test_split(
            data,
            test_size=config["split"]["val_size"],
            random_state=config["split"]["seed"],
            stratify=data[target],
        )
        features = [column for column in data if column not in ["id", "Name", target]]
        unique = pd.concat([val, train]).drop_duplicates(subset=features)
        before = len(train)
        train = unique[unique["id"].isin(train["id"])].copy()
        removed = before - len(train)

    if config["data"]["stage"] == "raw" and config["preprocessing"]["enabled"]:
        train = prepare_after_eda(train)
        val = prepare_after_eda(val)
    test_path = source / config["data"].get("test_file", "test.csv")
    test = None
    if test_path.is_file():
        test = pd.read_csv(test_path)
        validate_data(test)
        if target in test or set(test.columns) != set(data.columns) - {target}:
            raise ValueError(
                "Kaggle test должен содержать признаки train без целевого столбца"
            )
        if set(test["id"]) & (set(train["id"]) | set(val["id"])):
            raise ValueError("Train/val и Kaggle test пересекаются по id")
        if config["data"]["stage"] == "raw" and config["preprocessing"]["enabled"]:
            test = prepare_after_eda(test)
        test = test[train.drop(columns=target).columns]
        source_hashes[str(test_path)] = file_hash(test_path)
    validate_data(train, target)
    validate_data(val, target)
    if train[target].value_counts().min() < config["cv"]["n_splits"]:
        raise ValueError(
            "В train недостаточно объектов каждого класса для выбранного числа фолдов"
        )
    output = project_dir / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    val.to_csv(output / "val.csv", index=False)
    if test is not None:
        test.to_csv(output / "test.csv", index=False)
    elif (output / "test.csv").exists():
        (output / "test.csv").unlink()
    info = {
        "source_sha256": source_hashes,
        "train_sha256": file_hash(output / "train.csv"),
        "val_sha256": file_hash(output / "val.csv"),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_positive_share": float(train[target].mean()),
        "val_positive_share": float(val[target].mean()),
        "val_fraction_actual": len(val) / (len(train) + len(val)),
        "duplicates_removed": removed,
        "preparation_config": {
            key: config[key] for key in ["data", "split", "preprocessing"]
        },
    }
    if test is not None:
        info.update(
            {"test_sha256": file_hash(output / "test.csv"), "test_rows": len(test)}
        )
    write_json(output / "split_info.json", info)
    print(
        f"Train: {len(train)}, val: {len(val)}, удалено повторов из train: {removed}",
        flush=True,
    )
    return train, val, info


def load_prepared(config, project_dir):
    directory = project_dir / "data/processed"
    info = json.loads((directory / "split_info.json").read_text())
    actual = {key: config[key] for key in ["data", "split", "preprocessing"]}
    if actual != info["preparation_config"]:
        raise ValueError("Настройки подготовки изменились. Выполните dvc repro")
    for name in ["train", "val"]:
        if file_hash(directory / f"{name}.csv") != info[f"{name}_sha256"]:
            raise ValueError("Подготовленные CSV изменились. Выполните этап prepare")
    if (
        "test_sha256" in info
        and file_hash(directory / "test.csv") != info["test_sha256"]
    ):
        raise ValueError("Подготовленный Kaggle test изменился. Выполните этап prepare")
    return (
        pd.read_csv(directory / "train.csv"),
        pd.read_csv(directory / "val.csv"),
        info,
    )


def setup_mlflow(config, project_dir):
    directory = (project_dir / config["mlflow"]["directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(directory / 'mlflow.db').as_posix()}")
    name = config["mlflow"]["experiment"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(
            name, artifact_location=(directory / "artifacts").as_uri()
        )
    elif experiment.lifecycle_stage == "deleted":
        mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(name)


def flatten_params(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_params(value, name))
        else:
            result[name] = (
                json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, list)
                else str(value)
            )
    return result


def log_context(config, info, project_dir):
    effective_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    mlflow.log_params(flatten_params(effective_config))
    mlflow.log_params(
        {
            key: info[key]
            for key in [
                "train_rows",
                "val_rows",
                "train_positive_share",
                "val_positive_share",
                "val_fraction_actual",
                "duplicates_removed",
            ]
        }
    )
    mlflow.set_tags(
        {
            "model": config["model"]["name"],
            "train_sha256": info["train_sha256"],
            "val_sha256": info["val_sha256"],
        }
    )
    mlflow.log_dict(info, "data/split_info.json")
    mlflow.log_text(
        yaml.safe_dump(effective_config, allow_unicode=True), "source/params.yaml"
    )
    source_config = Path(config.get("_config_path", project_dir / "params.yaml"))
    if source_config.is_file():
        mlflow.log_text(source_config.read_text(), "source/original_params.yaml")
    for name in [
        "pipeline.py",
        "modeling.py",
        "requirements.txt",
        "dvc.yaml",
        "dvc.lock",
        "data/raw.dvc",
        "data/preprocessed.dvc",
    ]:
        path = project_dir / name
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="source")
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "catboost",
        "lightgbm",
        "mlflow",
        "dvc",
        "PyYAML",
        "joblib",
    ]
    mlflow.log_dict(
        {package: version(package) for package in packages},
        "source/library_versions.json",
    )


def train(config, project_dir=PROJECT_DIR):
    data, _, info = load_prepared(config, project_dir)
    target = config["data"]["target"]
    X, y = data.drop(columns=target), data[target]
    setup_mlflow(config, project_dir)
    with mlflow.start_run(run_name=f"{config['model']['name']} - train") as run:
        mlflow.set_tag("stage", "train")
        log_context(config, info, project_dir)
        start = perf_counter()
        estimator = make_pipeline(config, X)
        model, oof, folds, summary, best_params, search_results = run_training(
            estimator, X, y, config
        )
        training_seconds = perf_counter() - start
        model_params = model.named_steps["model"].get_params()
        mlflow.log_params(
            flatten_params(
                {"effective_model": model_params, "best_params": best_params}
            )
        )
        mlflow.log_metrics(summary | {"training_seconds": training_seconds})
        output = project_dir / "artifacts/models"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        joblib.dump(model, output / "model.joblib")
        oof.to_csv(output / "cv_predictions.csv", index=False)
        folds.to_csv(output / "cv_folds.csv", index=False)
        if search_results is not None:
            search_results.to_csv(output / "grid_results.csv", index=False)
            mlflow.log_table(search_results, "tables/grid_results.json")
        metadata = {
            "training_run_id": run.info.run_id,
            "training_config": {key: config[key] for key in TRAINING_SECTIONS},
            "data_info": info,
            "model_params": model_params,
            "best_params": best_params,
            "cv_metrics": summary,
            "training_seconds": training_seconds,
            "input_columns": X.columns.tolist(),
        }
        write_json(output / "training.json", metadata)
        mlflow.log_table(folds, "tables/cv_folds.json")
        mlflow.log_artifacts(str(output), artifact_path="training")
        print(
            f"CV Accuracy: {summary['cv_mean_accuracy']:.5f}, F1: {summary['cv_mean_f1']:.5f}",
            flush=True,
        )
        print(f"Обучение сохранено в MLflow: {run.info.run_id}", flush=True)
    return metadata


def save_importance(values, names, directory, stem, title, top_n):
    table = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    table.to_csv(directory / f"{stem}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    top = table.head(top_n).sort_values("importance")
    ax.barh(top["feature"], top["importance"], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Важность")
    fig.tight_layout()
    fig.savefig(directory / f"{stem}.png", dpi=140)
    plt.close(fig)


def make_reports(model, X, y, probability, config, output):
    threshold = config["evaluation"]["threshold"]
    prediction = (probability >= threshold).astype(int)
    displays = [
        ("confusion_matrix", ConfusionMatrixDisplay.from_predictions, prediction),
        ("roc_curve", RocCurveDisplay.from_predictions, probability),
        ("pr_curve", PrecisionRecallDisplay.from_predictions, probability),
    ]
    for name, plotting, values in displays:
        fig, ax = plt.subplots(figsize=(6, 5))
        plotting(y, values, ax=ax)
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=140)
        plt.close(fig)

    importance = config["evaluation"]["importance"]
    estimator = model.named_steps["model"]
    if config["model"]["name"] == "dummy":
        (output / "feature_importance.txt").write_text(
            "Dummy использует частоту классов и не оценивает важность признаков.\n"
        )
        return
    names = model.named_steps["preprocessing"].get_feature_names_out()
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
        title = "Модуль коэффициентов Logistic Regression"
    else:
        values = estimator.feature_importances_
        title = "Встроенная важность признаков"
    save_importance(
        values, names, output, "feature_importance", title, importance["top_n"]
    )
    if importance["permutation"]:
        sample = X.sample(
            n=min(len(X), importance["max_samples"]), random_state=config["cv"]["seed"]
        )

        def scoring(fitted, features, target):
            return accuracy_score(
                target, fitted.predict_proba(features)[:, 1] >= threshold
            )

        result = permutation_importance(
            model,
            sample,
            y.loc[sample.index],
            scoring=scoring,
            n_repeats=importance["n_repeats"],
            random_state=config["cv"]["seed"],
            n_jobs=1,
        )
        meaningful = [column for column in X if column not in ["id", "Name"]]
        positions = [X.columns.get_loc(column) for column in meaningful]
        save_importance(
            result.importances_mean[positions],
            meaningful,
            output,
            "permutation_importance",
            "Permutation importance на val",
            importance["top_n"],
        )


def load_training(config, project_dir, info):
    model_dir = project_dir / "artifacts/models"
    metadata = json.loads((model_dir / "training.json").read_text())
    if (
        metadata["training_config"] != {key: config[key] for key in TRAINING_SECTIONS}
        or metadata["data_info"] != info
    ):
        raise ValueError(
            "Модель обучалась с другими настройками или данными. Выполните dvc repro"
        )
    model = joblib.load(model_dir / "model.joblib")
    return model, metadata


def evaluate(config, project_dir=PROJECT_DIR):
    _, data, info = load_prepared(config, project_dir)
    model, metadata = load_training(config, project_dir, info)
    model_dir = project_dir / "artifacts/models"
    target = config["data"]["target"]
    X, y = data.drop(columns=target), data[target]
    setup_mlflow(config, project_dir)
    with mlflow.start_run(run_name=f"{config['model']['name']} - evaluate") as run:
        mlflow.set_tags(
            {"stage": "evaluate", "training_run_id": metadata["training_run_id"]}
        )
        log_context(config, info, project_dir)
        mlflow.log_params(
            flatten_params(
                {
                    "effective_model": metadata["model_params"],
                    "best_params": metadata["best_params"],
                }
            )
        )
        start = perf_counter()
        probability = model.predict_proba(X)[:, 1]
        predict_seconds = perf_counter() - start
        metrics = calculate_metrics(y, probability, config["evaluation"]["threshold"])
        oof = pd.read_csv(model_dir / "cv_predictions.csv")
        folds, cv_metrics = summarize_cv(oof, config["evaluation"]["threshold"])
        all_metrics = {
            f"val_{key}": value for key, value in metrics.items()
        } | cv_metrics
        all_metrics.update(
            {
                "training_seconds": metadata["training_seconds"],
                "cv_mean_fit_seconds": metadata["cv_metrics"]["cv_mean_fit_seconds"],
                "val_predict_seconds": predict_seconds,
            }
        )
        output = project_dir / "artifacts/reports/pipeline"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        write_json(output / "metrics.json", all_metrics)
        write_json(
            output / "evaluation.json",
            {
                "run_id": run.info.run_id,
                "training_run_id": metadata["training_run_id"],
                "config": config,
            },
        )
        predictions = pd.DataFrame(
            {
                "id": data["id"],
                "target": y,
                "probability": probability,
                "prediction": (probability >= config["evaluation"]["threshold"]).astype(
                    int
                ),
            }
        )
        predictions.to_csv(output / "predictions.csv", index=False)
        folds.to_csv(output / "cv_folds.csv", index=False)
        make_reports(model, X, y, probability, config, output)
        mlflow.log_metrics(all_metrics)
        mlflow.log_table(folds, "tables/cv_folds.json")
        mlflow.log_artifacts(str(output), artifact_path="evaluation")
        requirements = (project_dir / "requirements.txt").read_text().splitlines()
        example = X.head(5).copy()
        numeric = example.select_dtypes(include="number").columns
        example[numeric] = example[numeric].astype(float)
        signature = infer_signature(example, model.predict(example))
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=example,
            code_paths=[str(project_dir / "modeling.py")],
            pip_requirements=requirements,
        )
        print(
            f"Val Accuracy: {metrics['accuracy']:.5f}, F1: {metrics['f1']:.5f}, ROC-AUC: {metrics['roc_auc']:.5f}",
            flush=True,
        )
        print(f"Оценка сохранена в MLflow: {run.info.run_id}", flush=True)
    return all_metrics


def predict(config, project_dir=PROJECT_DIR):
    check_sources(config, project_dir, require_test=True)
    train_data, val_data, info = load_prepared(config, project_dir)
    if "test_sha256" not in info:
        raise ValueError("Kaggle test не подготовлен. Выполните этап prepare")
    model, metadata = load_training(config, project_dir, info)
    evaluation = json.loads(
        (project_dir / "artifacts/reports/pipeline/evaluation.json").read_text()
    )
    if (
        evaluation["training_run_id"] != metadata["training_run_id"]
        or evaluation["config"]["evaluation"] != config["evaluation"]
    ):
        raise ValueError(
            "Оценка относится к другой модели или порогу. Выполните этап evaluate"
        )
    test = pd.read_csv(project_dir / "data/processed/test.csv")
    raw_ids = pd.read_csv(
        project_dir
        / config["data"]["directory"]
        / config["data"].get("test_file", "test.csv"),
        usecols=["id"],
    )["id"]
    if not test["id"].equals(raw_ids):
        raise ValueError("Изменился состав или порядок id Kaggle test")
    full_data = pd.concat([train_data, val_data], ignore_index=True)
    target = config["data"]["target"]
    X, y = full_data.drop(columns=target), full_data[target]
    if set(test.columns) != set(X.columns):
        raise ValueError("Признаки Kaggle test отличаются от train")
    test = test[X.columns]
    final_model = clone(model)
    setup_mlflow(config, project_dir)
    with mlflow.start_run(run_name=f"{config['model']['name']} - predict") as run:
        mlflow.set_tags(
            {
                "stage": "predict",
                "training_run_id": metadata["training_run_id"],
                "evaluation_run_id": evaluation["run_id"],
                "fit_data": "train+val",
                "test_sha256": info["test_sha256"],
            }
        )
        log_context(config, info, project_dir)
        mlflow.log_params(
            flatten_params(
                {
                    "effective_model": metadata["model_params"],
                    "best_params": metadata["best_params"],
                }
            )
        )
        mlflow.log_params(
            {
                "full_train_rows": len(X),
                "test_rows": len(test),
                "full_train_positive_share": float(y.mean()),
            }
        )
        start = perf_counter()
        final_model.fit(X, y)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        probability = final_model.predict_proba(test)[:, 1]
        predict_seconds = perf_counter() - start
        if (
            not np.isfinite(probability).all()
            or not ((probability >= 0) & (probability <= 1)).all()
        ):
            raise ValueError("Вероятности должны быть конечными числами от 0 до 1")
        prediction = (probability >= config["evaluation"]["threshold"]).astype(int)
        submission = pd.DataFrame({"id": test["id"], target: prediction})
        output = project_dir / "artifacts/final"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        alias = (
            "logreg"
            if config["model"]["name"] == "logistic_regression"
            else config["model"]["name"]
        )
        submission_path = output / f"submission_{alias}.csv"
        submission.to_csv(submission_path, index=False)
        joblib.dump(final_model, output / "model.joblib")
        result = {
            "run_id": run.info.run_id,
            "training_run_id": metadata["training_run_id"],
            "evaluation_run_id": evaluation["run_id"],
            "model": config["model"]["name"],
            "submission": submission_path.name,
            "full_train_rows": len(X),
            "test_rows": len(test),
            "full_train_sha256": hashlib.sha256(
                full_data.to_csv(index=False).encode()
            ).hexdigest(),
            "test_sha256": info["test_sha256"],
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
        write_json(output / "prediction.json", result)
        mlflow.log_metrics(
            {
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "predicted_positive_share": float(prediction.mean()),
            }
        )
        mlflow.log_artifact(str(submission_path), artifact_path="submissions")
        mlflow.log_artifact(str(output / "prediction.json"), artifact_path="prediction")
        example = test.head(5).copy()
        numeric = example.select_dtypes(include="number").columns
        example[numeric] = example[numeric].astype(float)
        mlflow.sklearn.log_model(
            final_model,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(example, final_model.predict(example)),
            input_example=example,
            code_paths=[str(project_dir / "modeling.py")],
            pip_requirements=(project_dir / "requirements.txt")
            .read_text()
            .splitlines(),
        )
        print(f"Submission: {submission_path}, строк: {len(submission)}", flush=True)
        print(f"Финальная модель сохранена в MLflow: {run.info.run_id}", flush=True)
    return result


def run(config, project_dir=PROJECT_DIR):
    check_sources(config, project_dir, require_test=True)
    prepare(config, project_dir)
    train(config, project_dir)
    evaluate(config, project_dir)
    return predict(config, project_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка данных, обучение и оценка Mental Health Prediction"
    )
    parser.add_argument(
        "stage", choices=["prepare", "train", "evaluate", "predict", "run"]
    )
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--model", choices=["catboost", "logreg"])
    arguments = parser.parse_args()
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = read_config(config_path, arguments.model)
    stages = {
        "prepare": prepare,
        "train": train,
        "evaluate": evaluate,
        "predict": predict,
        "run": run,
    }
    stages[arguments.stage](config)


if __name__ == "__main__":
    main()
