# Mental Health Prediction

Учебный проект бинарной классификации Depression. Источник - соревнование [Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11). Основная метрика - Accuracy. Также считаем Precision, Recall, F1, ROC-AUC и Average Precision.

В 1_eda.ipynb находятся исследование и обоснование подготовки данных. В 2_models.ipynb - сравнение четырех моделей, GridSearch для LogReg и CatBoost, анализ ошибок и результаты Kaggle. Скрипты воспроизводят подготовку, обучение выбранной модели, оценку на val и финальные предсказания.

## Окружение

Нужен Python 3.12. Команды создания окружения выполняются из родительской директории data_science:

```bash
python3.12 -m venv .venv
cd data_science_project
../.venv/bin/python -m pip install --no-compile -r requirements.txt
../.venv/bin/python -m pip check
```

Если .venv уже существует, используйте ее. requirements.txt содержит полный снимок установленных пакетов, включая зависимости MLflow, DVC, pytest и Ruff. Обновлять его после изменения окружения:

```bash
../.venv/bin/python -m pip freeze > requirements.txt
```

Все дальнейшие команды выполняются из data_science_project.

## Данные и запуск одной командой

Скачайте train.csv и test.csv со [страницы данных Kaggle](https://www.kaggle.com/competitions/playground-series-s4e11/data) и положите их в data/raw. При отсутствии CSV он останавливается до обучения с инструкцией.

```bash
../.venv/bin/python pipeline.py run --model catboost
../.venv/bin/python pipeline.py run --model logreg
```

Каждая команда выполняет четыре этапа для одной модели:

1. **prepare**: чтение CSV, стратифицированный split, фиксированная очистка и сохранение train/val/test.
2. **train**: CV на train и обучение Pipeline на всем train.
3. **evaluate**: оценка на val, сохранение метрик, матрицы ошибок, ROC/PR-кривых и важности.
4. **predict**: новый Pipeline с выбранными параметрами обучается на train + val и предсказывает Kaggle test.

run выполняет все этапы заново. При отсутствии --model используется model.name из YAML. --model выбирает профиль параметров и не изменяет YAML. Для другого файла настроек добавьте --config path/to/params.yaml.

Submission текущего запуска сохраняется в artifacts/final/submission_catboost.csv или submission_logreg.csv. Формат - ровно id,Depression, без индекса и вероятностей. Рядом находятся полный финальный Pipeline model.joblib и сведения о запуске prediction.json. artifacts/final содержит результаты текущей модели, предыдущие версии остаются в MLflow и кеше DVC. В artifacts/submissions сохранены ранее созданные файлы из ноутбука.

## Настройки YAML

Гиперпараметры задаются отдельно для каждой модели:

```yaml
model:
  name: catboost
  params:
    catboost:
      iterations: 500
      depth: 6
      learning_rate: 0.05
    logreg:
      C: 1
      max_iter: 2000
      solver: lbfgs
```

Остальные параметры конструктора берутся из базовых настроек modeling.py. Помимо двух профилей, скрипты сохраняют поддержку Dummy, Random Forest и LightGBM через YAML с соответствующим именем модели и плоскими параметрами.

По умолчанию используется data/raw, split 80/20 с seed=56, пять фолдов с seed=556, порог 0.5 и rare_min_count=5. Из train исключаем повторы по признакам с приоритетом val. Исходные CSV и готовые data/preprocessed не изменяются. data/processed содержит данные текущего скриптового запуска.

Фиксированная очистка удаляет Name, нормализует пробелы, исправляет понятные варианты Degree и проверяет значения сна и питания. Содержательные интервалы сна сохраняются. Общие индикаторы наличия учебных и рабочих данных не добавляются.

Обучаемый преобразователь находится внутри sklearn Pipeline. Медианы, редкие категории, кодирование и масштабирование определяются по обучающей части каждого фолда. Missing обозначает пропуск, Rare - встречавшуюся в train категорию с частотой менее 5, Unknown - неизвестное значение. Для LogReg/RF используем числовые признаки, индикаторы структурных пропусков и кодирование категорий. LogReg дополнительно масштабирует числовые признаки. CatBoost получает строковые категории, LightGBM - pandas category.

Для сохраненного разбиения:

```yaml
data:
  directory: data/preprocessed
  train_file: train.csv
  val_file: val.csv
  test_file: test.csv
  mode: fixed
  stage: preprocessed
  target: Depression
```

В режиме fixed состав и порядок train/val сохраняются, split.val_size не меняет готовые выборки. Для preprocessed фиксированная очистка повторно не применяется.

GridSearch по умолчанию выключен. Для поиска LogReg задайте search.enabled: true и search.grid: {C: [0.1, 1, 10]}. Для CatBoost используйте depth: [4, 6] и learning_rate: [0.03, 0.05]. Сетка должна соответствовать выбранной модели. search.scoring задает метрику выбора. Val не участвует в подборе параметров.

Дополнительная permutation importance включается через evaluation.importance.permutation. По умолчанию ограничиваем расчет 5000 строками и тремя повторениями. Для LogReg встроенная важность - модуль коэффициента после преобразований, для CatBoost - встроенная важность дерева. Эти значения не доказывают причинное влияние.

## DVC и воспроизводимость

```bash
../.venv/bin/dvc repro
../.venv/bin/dvc repro --force
```

DVC выполняет те же четыре этапа, используя model.name из YAML. Для переключения на LogReg измените model.name на logreg. Неизменный запуск использует кеш и пропускает готовые этапы. Изменение параметров модели запускает обучение, оценку и предсказания. Изменение evaluation.threshold запускает оценку и финальные предсказания. --force выполняет все этапы заново.

Подготовленные данные и результаты этапов учитываются в dvc.lock. Исходные raw и preprocessed зарегистрированы отдельными .dvc файлами. После ручной замены исходных CSV обновите версию:

```bash
../.venv/bin/dvc add data/raw
../.venv/bin/dvc push --run-cache
```

Локальный remote находится в storage/dvc. Для восстановления после получения кода и dvc.lock используйте dvc pull, затем dvc checkout. На другом компьютере локальное хранилище нужно перенести отдельно или заново выполнить этапы из исходных CSV. Для VirtualBox используется cache.type=copy.

## MLflow

```bash
../.venv/bin/mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db --host 127.0.0.1 --port 5000
```

Откройте [http://127.0.0.1:5000](http://127.0.0.1:5000). Текущий ноутбук использует mental-health-models, скриптовый пайплайн - mental-health-pipeline.

Обучение, оценка и финальные предсказания создают связанные записи с тегами stage=train/evaluate/predict.

В MLflow сохраняются исходный и фактический YAML, параметры модели, версии библиотек, копия кода, контрольные суммы, CV, метрики val, графики и модели. Final submission сохраняется как артефакт.

Локальная модель загружается:

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/final/model.joblib")
test = pd.read_csv("data/processed/test.csv")
probability = model.predict_proba(test)[:, 1]
```

Для переноса истории MLflow нужны tracking/mlflow.db и tracking/artifacts. В базе сохранены абсолютные пути к артефактам.

## Текущие результаты

На новой подготовке данных ноутбук получил:

| Модель | CV Accuracy | Val Accuracy | Val F1 | Среднее обучение на фолде, с |
|---|---:|---:|---:|---:|
| CatBoost | 0.93904 | 0.94101 | 0.83492 | 39.1 |
| Logistic Regression | 0.93822 | 0.94108 | 0.83509 | 5.1 |
| LightGBM | 0.93775 | 0.94083 | 0.83556 | 5.7 |
| Random Forest | 0.93279 | 0.93412 | 0.81170 | 11.8 |

GridSearch выбрал исходные C=1 для LogReg и depth=6, learning_rate=0.05 для CatBoost.

После финального обучения на train + val модели показали следующие результаты на Kaggle:

| Модель | Public Score | Private Score |
|---|---:|---:|
| Logistic Regression | 0.94109 | 0.93931 |
| CatBoost | 0.94205 | 0.94057 |

![Результаты Kaggle](screenshots/kaggle_submission_results.png)

CatBoost немного лучше по CV и обеим оценкам Kaggle. LogReg примерно в 7.7 раза быстрее в исходном сравнении. Для итогового варианта выбран CatBoost, LogReg остается простой и быстрой альтернативой.

## Основные файлы

- 1_eda.ipynb и 2_models.ipynb - исследование и результаты.
- pipeline.py и modeling.py - команды этапов, преобразования, модели и CV.
- params.yaml, dvc.yaml и dvc.lock - параметры и воспроизводимость.
- requirements.txt - зафиксированное окружение.
- data/raw, data/preprocessed и data/processed - исходные, исследовательские и текущие данные.
- artifacts/reports/pipeline и artifacts/final - отчеты и финальная модель текущего запуска.
- tracking и storage/dvc - локальная история MLflow и хранилище DVC.
