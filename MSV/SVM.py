import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import time

# Cargar el archivo CSV
file_path = 'ruta'
data = pd.read_csv(file_path)

# Filtramos las primeras 1000 filas para trabajar con menos datos
Numerodatos = 1000
data = data.head(Numerodatos)
total_datos = len(data)


# Codificación de variables categóricas
label_encoders = {}
categorical_columns = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Seleccionar variables
X = data.drop(columns=['PlayerID','EngagementLevel']).values
y = data['EngagementLevel'].values

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

DatosEntrenamiento= len(X_train)

DatosPrueba= len(X_test)


# Función de clasificación con SVM y diferentes kernels
def Maquina():
    kernels = ['linear', 'poly', 'rbf']
    matrices_confusion = []
    accuracy_scores = []
    training_times = []  
    titles = ['SVM Lineal', 'SVM Polinomial', 'SVM Circular (RBF)']

    for kernel in kernels:
        print(f"\n--------SVM con kernel: {kernel}--------")
        svm = SVC(kernel=kernel, random_state=42)
        
        start_time = time.time()
        svm.fit(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        training_times.append(training_time)

        # Predicción
        y_svm = svm.predict(X_test)

        accuracy = accuracy_score(y_test, y_svm)
        accuracy_scores.append(accuracy * 100)
        cm = confusion_matrix(y_test, y_svm)
        matrices_confusion.append(cm)

        print(f"\nPrecisión kernel {kernel}: {accuracy:.4f}")
        print(f"Tiempo de entrenamiento {kernel}: {training_time:.4f} segundos")

        # Imprimir el reporte de clasificación
        report = classification_report(y_test, y_svm, target_names=['Low', 'Medium', 'High'])
        print(f"\nReporte de clasificación para el kernel {kernel}:\n{report}")

    # Graficar las matrices de confusión
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, matrix in enumerate(matrices_confusion):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Oranges', ax=axes[i])
        axes[i].set_title(f"Matriz de confusión {titles[i]}")
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # Reducción dimensional con PCA a 2 características
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    # Graficar las divisiones de datos para cada kernel
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, kernel in enumerate(kernels):
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train_pca, y_train)

        # Crear grid para mostrar los límites de decisión
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        # Predecir en el grid
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Graficar los límites de decisión y los puntos de entrenamiento
        axes[i].contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        axes[i].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', marker='o', s=60, cmap='coolwarm')
        axes[i].set_title(f"Límites de Decisión - {titles[i]}")

    plt.tight_layout()
    plt.show()

# Ejecutar la función
Maquina()


print(f'\nNumero de datos total: {total_datos}')
print(f'Numero de datos de entrenamiento: {DatosEntrenamiento}')
print(f'Numero de datos de entrenamiento: {DatosPrueba}')