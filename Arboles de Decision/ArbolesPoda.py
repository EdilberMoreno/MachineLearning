import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time

# Cargar el archivo CSV
file_path = 'ruta'
data = pd.read_csv(file_path)

# Seleccionar solo los primeros 50000 datos
data = data.head(5000)

# Preprocesamiento de los datos
# Convertir variables categóricas en numéricas
label_encoders = {}
for column in ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Seleccionar características y variable objetivo
X = data.drop(columns=['PlayerID', 'EngagementLevel'])
y = data['EngagementLevel']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

DatosPrueba= len(X_test)
print("Datos para la prueba:", DatosPrueba)

DatosEntreno=len(X_train)
print("Datos para el entrenamientos:", DatosEntreno)

""" print(X.head(100)) """

# Tomar el tiempo antes de entrenar el modelo
start_time = time.time()

# Entrenar el modelo
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10 , min_samples_split=20, max_depth=9, random_state=42)
clf.fit(X_train, y_train)

# Tomar el tiempo después de entrenar el modelo
end_time = time.time()

# Calcular el tiempo de ejecución
training_time = end_time - start_time

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Visualizar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoders['EngagementLevel'].classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()

# Determinar el índice de la característica en el nodo raíz
root_feature_index = clf.tree_.feature[0]

# Obtener el nombre de la característica correspondiente
root_feature_name = X.columns[root_feature_index]

print(f"El nodo raíz utiliza la característica: {root_feature_name}")

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoders['EngagementLevel'].classes_)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Mostrar el tiempo de ejecución
print(f"Tiempo de ejecución para entrenar el modelo: {training_time:.2f} segundos")

# Obtener la profundidad del árbol
tree_depth = clf.tree_.max_depth
print(f"El número de niveles (profundidad) del árbol es: {tree_depth}")

# Visualizar el árbol de decisión
plt.figure(figsize=(18,10), dpi=180)
plot_tree(clf, max_depth=3, filled=True, feature_names=X.columns, class_names=label_encoders['EngagementLevel'].classes_, rounded=True)
plt.title("Árbol de Decisión")
plt.show()

# Visualizar la importancia de las características
importances = clf.feature_importances_
features = X.columns

plt.figure(figsize=(13, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características en el Arbol de Decisión')
plt.tight_layout()  # Ajustar el diseño para evitar recortes
plt.show()
