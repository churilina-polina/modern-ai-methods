from sklearn import preprocessing

# Предоставление меток входных данных
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Создание кодировщика и установление соответствия между метками и числами
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Вывод отображения
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Преобразование меток с помощью кодировщика
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", encoded_values.tolist())

# Декодирование набора чисел с помощью декодера
# (в методичке здесь создаётся новый, необученный LabelEncoder — это ошибка,
#  он не сможет декодировать, поэтому используем уже обученный encoder)
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", decoded_list.tolist())
