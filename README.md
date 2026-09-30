# Mental Health Prediction

Учебный проект бинарной классификации Depression. Источник - соревнование [Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11). Основная метрика - Accuracy. Также считаем Precision, Recall, F1, ROC-AUC и Average Precision.

В 1_eda.ipynb находятся исследование и обоснование подготовки данных.

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

## Данные

Скачайте train.csv и test.csv со [страницы данных Kaggle](https://www.kaggle.com/competitions/playground-series-s4e11/data) и положите их в data/raw.

Исходные CSV хранятся в data/raw, подготовленные train/val/test - в data/preprocessed. Подготовка и разбиение описаны в 1_eda.ipynb.

Фиксированная очистка удаляет Name, нормализует пробелы, исправляет понятные варианты Degree и проверяет значения сна и питания. Содержательные интервалы сна сохраняются. Общие индикаторы наличия учебных и рабочих данных не добавляются.

## Хранение данных в DVC

Исходные raw и подготовленные preprocessed зарегистрированы отдельными .dvc файлами. Локальный remote находится в storage/dvc. На другом компьютере локальное хранилище нужно перенести отдельно. Для VirtualBox используется cache.type=copy.
