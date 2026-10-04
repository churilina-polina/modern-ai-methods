import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utilities import visualize_classifier

# Определение образца входных данных
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5],
              [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Значение C можно передать аргументом: python task3_logistic_regression.py 100
C = float(sys.argv[1]) if len(sys.argv) > 1 else 1

# Создание логистического классификатора
# (в новых версиях sklearn для liblinear нужно явно указать схему one-vs-rest)
from sklearn.multiclass import OneVsRestClassifier
classifier = OneVsRestClassifier(linear_model.LogisticRegression(solver='liblinear', C=C))

# Тренировка классификатора
classifier.fit(X, y)
print(f"C = {C}, accuracy on train = {classifier.score(X, y):.2f}")

# Визуализация работы классификатора
visualize_classifier(classifier, X, y, title=f'Logistic regression, C = {C:g}')
