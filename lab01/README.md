# Лабораторная работа 1: классификация и Transfer Learning

Дисциплина «Современные методы искусственного интеллекта», Университет ИТМО.
Выполнила: Чурилина Полина Олеговна, группа K3320.

## Структура

| Папка / файл | Содержание |
|---|---|
| `part_a/task1_preprocessing.py` | 1. Предобработка: бинаризация, исключение среднего, MinMax, L1/L2 |
| `part_a/task2_label_encoding.py` | 2. Кодирование меток |
| `part_a/task3_logistic_regression.py` | 3. Логистический классификатор (`python task3_logistic_regression.py 100` для C=100) |
| `part_a/task4_naive_bayes.py` | 4. Наивный байесовский классификатор + перекрестная проверка |
| `part_a/task5_confusion_matrix.py` | 5. Матрица неточностей |
| `part_a/task6_svm_income.py` | 6. SVM: прогноз дохода, сравнение ядер |
| `part_a/task7_1_single_var_regression.py` | 7.1. Одномерная линейная регрессия |
| `part_a/task7_2_multi_var_regression.py` | 7.2. Многомерная линейная и полиномиальная регрессия |
| `part_a/task8_svr_housing.py` | 8. SVR: стоимость жилья (Boston Housing) |
| `part_b/transfer_learning.ipynb` | Часть 2: Transfer Learning (MobileNetV2, Oxford-IIIT Pet, 37 пород): Feature Extraction и Fine-tuning |

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd part_a
python task1_preprocessing.py      # и т.д.

cd ../part_b
jupyter notebook transfer_learning.ipynb
```

Ноутбук части 2 сам скачивает датасет (~800 МБ) и веса MobileNetV2. Работает на CUDA (Colab), Apple MPS или CPU.

Код из методических указаний адаптирован под актуальный scikit-learn: `sklearn.cross_validation` заменён на `sklearn.model_selection`, `load_boston()` заменён на `fetch_openml('boston')`.
