import os
import certifi
import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

# Python с python.org на macOS не видит системные SSL-сертификаты — берём их из certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())

# datasets.load_boston() удалён из sklearn начиная с версии 1.2,
# поэтому загружаем тот же набор данных с OpenML (кэшируется локально после первого запуска)
data = datasets.fetch_openml(name='boston', version=1, as_frame=False, parser='liac-arff', data_home='./sklearn_data')
X_all = data.data.astype(float)
y_all = data.target.astype(float)

# Перемешивание данных
X, y = shuffle(X_all, y_all, random_state=7)

# Разбивка данных на обучающий и тестовый наборы 80/20
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Тестовые точки данных (первая — из методички, остальные — реальные дома из датасета)
test_points = {
    'Точка из методички': [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27],
    'Дом №0 из датасета': list(X_all[0]),
    'Дом №283 из датасета': list(X_all[283]),
    'Дом №400 из датасета': list(X_all[400]),
}
true_prices = {'Дом №0 из датасета': y_all[0], 'Дом №283 из датасета': y_all[283], 'Дом №400 из датасета': y_all[400]}

# Различные комбинации параметров C и epsilon
params = [(1.0, 0.1), (10.0, 0.1), (0.1, 0.1), (1.0, 1.0), (1.0, 5.0)]

for C, epsilon in params:
    # Создание регрессионной модели на основе SVM
    sv_regressor = SVR(kernel='linear', C=C, epsilon=epsilon)

    # Обучение регрессора SVM
    sv_regressor.fit(X_train, y_train)

    # Оценка эффективности работы регрессора
    y_test_pred = sv_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    evs = explained_variance_score(y_test, y_test_pred)
    print(f"\n#### Performance: C={C}, epsilon={epsilon} ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))
    print("Number of support vectors =", len(sv_regressor.support_))

    # Тестирование регрессора на тестовых точках данных
    for name, point in test_points.items():
        pred = sv_regressor.predict([point])[0]
        true = f" (реальная цена: {true_prices[name]:.1f})" if name in true_prices else ""
        print(f"Predicted price [{name}]: {pred:.2f}{true}")
