import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Cargar el archivo CSV
file_path = 'ruta'
data = pd.read_csv(file_path)


data = data.head(100)

# Inicio para medir el tiempo
start_time = time.time()

# Codificar las variables categóricas
label_encoders = {}
for column in ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Definir características (X) y variable objetivo (y)
X = data.drop(columns=['PlayerID', 'EngagementLevel'])
y = data['EngagementLevel']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Obtener los rangos de las características originales para generar nuevos puntos en esos rangos
x_min, x_max = data['SessionsPerWeek'].min(), data['SessionsPerWeek'].max()
y_min, y_max = data['AvgSessionDurationMinutes'].min(), data['AvgSessionDurationMinutes'].max()

# Generar 3 datos aleatorios basados en los valores originales
new_sessions_per_week = np.random.uniform(x_min, x_max, size=2)  # Sesiones por semana
new_avg_session_duration = np.random.uniform(y_min, y_max, size=2)  # Duración promedio de la sesión

# Generar el resto de las características aleatorias (dentro de los rangos apropiados)
# 11 características, que es lo que el modelo espera
new_data_rest = np.random.randint(0, 19, size=(2, X_train.shape[1] - 2))  # Generar para las otras columnas

# Concatenar todas las columnas
new_dataX = np.column_stack((new_sessions_per_week, new_avg_session_duration, new_data_rest))

# Crear un modelo KNN
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)

# Predecir las clases para los nuevos datos
new_predictions = knn.predict(new_dataX)

# Predecir las clases para el conjunto de prueba
y_pred = knn.predict(X_test)

# Obtener las distancias y los índices de los vecinos más cercanos
distances, indices = knn.kneighbors(new_dataX)

# Elegir las características que quieras graficar (por ejemplo, 'SessionsPerWeek' y 'AvgSessionDurationMinutes')
x_column = 'SessionsPerWeek'  # Característica en el eje X
y_column = 'AvgSessionDurationMinutes'  # Característica en el eje Y

# Métrica de exactitud del modelo
accuracy = accuracy_score(y_test, y_pred)

# Validación cruzada con diferentes valores de K
k_values = np.arange(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Encontrar el índice del valor máximo en cv_scores
best_k_index = np.argmax(cv_scores)

# Obtener el mejor valor de K
best_k_value = k_values[best_k_index]

# Tiempo final para medir el tiempo de ejecución
end_time = time.time()

# Corregir la gráfica para que muestre correctamente los K vecinos más cercanos
plt.figure(figsize=(10, 6))

# Graficar los datos originales
sns.scatterplot(x=data[x_column], y=data[y_column], hue=data['EngagementLevel'], palette='viridis', s=60)

# Graficar los nuevos puntos generados aleatoriamente
plt.scatter(new_dataX[:, 0], new_dataX[:, 1], color='green', marker='x', s=200, label='New Data')

# Graficar los K vecinos más cercanos para cada punto nuevo generado
for i, (dist, idx) in enumerate(zip(distances, indices)):
    for neighbor_idx in idx:
        plt.scatter(X_train.iloc[neighbor_idx]['SessionsPerWeek'], 
                    X_train.iloc[neighbor_idx]['AvgSessionDurationMinutes'], 
                    color='red', marker='o', s=100, label=f'Neighbor of New Data {i+1}' if neighbor_idx == idx[0] else "")

# Personalizar la gráfica
plt.xlabel('Sesiones por semana')
plt.ylabel('Duración promedio de la sesión (minutos)')
plt.legend(bbox_to_anchor=(1, 0.3))  # Colocar la leyenda fuera del gráfico
plt.title(f'Gráfica de dispersión en 2D con los {knn.n_neighbors} vecinos más cercanos')

# Mostrar la gráfica
plt.show()

""" # Graficar los datos originales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[x_column], y=data[y_column], hue=data['EngagementLevel'], palette='viridis', s=60)

# Graficar los 3 nuevos datos generados aleatoriamente
plt.scatter(new_dataX[:, 0], new_dataX[:, 1], color='green', marker='x', label='New Data', s=100)

# Etiquetas y personalización de la gráfica
plt.xlabel('Sesiones por semana')
plt.ylabel('Duración promedio de la sesión (minutos)')
plt.legend(bbox_to_anchor=(1, 0.3))
plt.title('Gráfica de dispersión de datos con nuevos puntos aleatorios')
plt.show() """

# Generar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['EngagementLevel'].classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de Confusión')
plt.show()

# Graficar el rendimiento del modelo con diferentes valores de K
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', color='b')
plt.title('Rendimiento del modelo con diferentes valores de K')
plt.xlabel('Valor de K')
plt.ylabel('Precisión (accuracy)')
plt.grid(True)
plt.show()

# Gráfica 3D de los K vecinos más cercanos
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Definir colores para los niveles de 'EngagementLevel'
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}

# Graficar los datos de entrenamiento
ax.scatter(X_train['SessionsPerWeek'], 
           X_train['AvgSessionDurationMinutes'], 
           X_train['PlayerLevel'], 
           c=[colors[label] for label in y_train], 
           s=60, label='Training Data')

# Graficar el nuevo punto generado aleatoriamente
ax.scatter(new_dataX[:, 0], new_dataX[:, 1], new_dataX[:, 2], 
           c='black', marker='x', s=100, label='New Data')

# Graficar los K vecinos más cercanos para cada punto nuevo
for i, idx in enumerate(indices):
    # Los índices de los K vecinos más cercanos están en 'indices'
    for neighbor_idx in idx:
        ax.scatter(X_train.iloc[neighbor_idx]['SessionsPerWeek'], 
                   X_train.iloc[neighbor_idx]['AvgSessionDurationMinutes'], 
                   X_train.iloc[neighbor_idx]['PlayerLevel'], 
                   c='orange', marker='o', s=100, label=f'Neighbors of New Data {i+1}' if i == 0 and neighbor_idx == idx[0] else "")

# Etiquetas para los ejes
ax.set_xlabel('Sesiones por semana')
ax.set_ylabel('Duración promedio de la sesión (minutos)')
ax.set_zlabel('Nivel del jugador')
ax.set_title(f'Gráfica 3D de los {knn.n_neighbors} vecinos más cercanos al nuevo punto')

# Añadir leyenda
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Training Data',
                              markerfacecolor='gray', markersize=10),
                   plt.Line2D([0], [0], marker='x', color='black', label='New Data', markersize=10),
                   plt.Line2D([0], [0], marker='o', color='orange', label='Neighbors', markersize=10)]

ax.legend(handles=legend_elements, loc='upper left')

plt.show()

""" # Crear la gráfica de dispersión en 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Definir colores para los niveles de 'EngagementLevel'
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}

# Asegurarse de que las columnas existan en el conjunto de datos
if 'SessionsPerWeek' in X_train.columns and 'AvgSessionDurationMinutes' in X_train.columns and 'PlayerLevel' in X_train.columns:
    # Graficar los datos de entrenamiento
    scatter = ax.scatter(X_train['SessionsPerWeek'], 
                         X_train['AvgSessionDurationMinutes'], 
                         X_train['PlayerLevel'],
                         c=[colors[label] for label in y_train], 
                         s=60)

    # Añadir los nuevos datos a la gráfica 3D
    ax.scatter(new_dataX[:, 0], new_dataX[:, 1], new_dataX[:, 2], 
           c='black', marker='x', s=100, label='New Data')

    # Etiquetas para los ejes
    ax.set_xlabel('Sesiones por semana')
    ax.set_ylabel('Duración promedio de la sesión (minutos)')
    ax.set_zlabel('Nivel del jugador')
    ax.set_title('Gráfica de dispersión 3D de datos de juego')

    # Obtener las clases que están presentes en los datos
    unique_labels = np.unique(y_train)

    # Filtrar los colores para las etiquetas presentes en el conjunto de datos
    filtered_colors = {i: colors[i] for i in unique_labels if i in colors}

    # Añadir leyenda solo para las etiquetas presentes
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label_encoders['EngagementLevel'].inverse_transform([i])[0],
                                  markerfacecolor=color, markersize=10)
                       for i, color in filtered_colors.items()]

    # Añadir también la leyenda para los nuevos datos
    legend_elements.append(plt.Line2D([0], [0], marker='x', color='black', label='New Data', markersize=20))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1))

    # Ajustar el diseño para que la leyenda no se superponga
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()
else:
    print("Las columnas requeridas no están en el conjunto de entrenamiento.") """

# Calcular el tiempo de ejecución
training_time = end_time - start_time

print("Métricas del modelo KNN:")
print("---------------------------------------------------------")
print("Predicciones para los nuevos datos aleatorios: ", new_predictions)

print(f"El mejor valor de K encontrado es {best_k_value}")

print(f"Exactitud del modelo KNN: {accuracy * 100:.2f}%")

print(f"Tiempo de ejecución para entrenar el modelo: {training_time:.2f} segundos")

# Mostrar las distancias de los K vecinos más cercanos
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f"\nNuevos datos #{i+1}:")
    print(f"Distancias a los K vecinos más cercanos: {dist}")
    print(f"Índices de los K vecinos más cercanos: {idx}")
