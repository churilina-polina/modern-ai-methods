import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score  # раньше: sklearn.cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Входной файл
input_file = 'income_data.txt'

# Чтение данных
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Преобразование в массив numpy array
X = np.array(X)

# Преобразование строковых данных в числовые
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Перекрестная проверка: разбиение 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Создание и обучение SVM-классификатора (как в методичке)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Вычисление F-меры для SVM-классификатора
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100 * f1.mean(), 2)) + "%")


def encode_point(input_data):
    """Кодирование точки данных теми же кодировщиками, что и обучающая выборка."""
    input_data_encoded = [-1] * len(input_data)
    count = 0
    for i, item in enumerate(input_data):
        if item.isdigit():
            input_data_encoded[i] = int(input_data[i])
        else:
            # transform ожидает список, поэтому [item]
            input_data_encoded[i] = int(label_encoder[count].transform([item])[0])
            count += 1
    return np.array(input_data_encoded).reshape(1, -1)


# Тестовые точки данных (первая — из методички)
test_points = [
    ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners',
     'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States'],
    ['52', 'Self-emp-inc', '287927', 'Masters', '14', 'Married-civ-spouse', 'Exec-managerial',
     'Husband', 'White', 'Male', '15024', '0', '60', 'United-States'],
    ['23', 'Private', '122272', 'Some-college', '10', 'Never-married', 'Sales',
     'Own-child', 'Black', 'Female', '0', '0', '25', 'United-States'],
    ['45', 'Federal-gov', '170915', 'Doctorate', '16', 'Married-civ-spouse', 'Prof-specialty',
     'Husband', 'Asian-Pac-Islander', 'Male', '0', '1902', '50', 'India'],
]

print("\n--- Предсказания LinearSVC для тестовых точек ---")
for point in test_points:
    predicted_class = classifier.predict(encode_point(point))
    print(point[0], point[3], point[6], point[12] + 'h/week',
          '->', label_encoder[-1].inverse_transform(predicted_class)[0])


# Сравнение разных ядер и параметров.
# Ядерные SVC обучаются долго (квадратично от числа точек), поэтому
# для них берём подвыборку и масштабируем признаки.
print("\n--- Сравнение ядер и параметров (на тестовой выборке 20%) ---")
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
n_sub = 8000
rng = np.random.RandomState(0)
idx = rng.choice(len(X_train_s), n_sub, replace=False)

configs = [
    ("LinearSVC, без масштабирования", LinearSVC(random_state=0), False, False),
    ("LinearSVC, со StandardScaler",   LinearSVC(random_state=0), True, False),
    ("SVC linear, C=1",                SVC(kernel='linear', C=1), True, True),
    ("SVC rbf, C=1",                   SVC(kernel='rbf', C=1), True, True),
    ("SVC rbf, C=10",                  SVC(kernel='rbf', C=10), True, True),
    ("SVC poly (deg=3), C=1",          SVC(kernel='poly', degree=3, C=1), True, True),
    ("SVC sigmoid, C=1",               SVC(kernel='sigmoid', C=1), True, True),
]

print(f"{'Модель':34s} {'Accuracy':>9s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s}")
for name, model, scaled, subsample in configs:
    Xtr, Xte = (X_train_s, X_test_s) if scaled else (X_train, X_test)
    ytr = y_train
    if subsample:
        Xtr, ytr = Xtr[idx], ytr[idx]
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print(f"{name:34s} {100*accuracy_score(y_test, pred):8.2f}% "
          f"{100*precision_score(y_test, pred, average='weighted', zero_division=0):9.2f}% "
          f"{100*recall_score(y_test, pred, average='weighted'):7.2f}% "
          f"{100*f1_score(y_test, pred, average='weighted'):7.2f}%")

# Предсказания лучшей модели (rbf) для тех же точек
best = SVC(kernel='rbf', C=10).fit(X_train_s[idx], y_train[idx])
print("\n--- Предсказания SVC(rbf, C=10) для тестовых точек ---")
for point in test_points:
    predicted_class = best.predict(scaler.transform(encode_point(point)))
    print(point[0], point[3], point[6], point[12] + 'h/week',
          '->', label_encoder[-1].inverse_transform(predicted_class)[0])
