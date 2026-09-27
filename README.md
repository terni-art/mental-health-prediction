# Mental Health Prediction

Учебный проект бинарной классификации Depression. Источник - соревнование [Exploring Mental Health Data](https://www.kaggle.com/competitions/playground-series-s4e11). Основная метрика - Accuracy. Также считаем Precision, Recall, F1, ROC-AUC и Average Precision.

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
