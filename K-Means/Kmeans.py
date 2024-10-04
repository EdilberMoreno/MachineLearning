import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score, accuracy_score,classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import time

# =========================
# Cargar y Preprocesar Datos
# =========================

file_path = 'ruta'
data = pd.read_csv(file_path)

data = data.head(700)

total_datos = len(data)

# Aplicar LabelEncoder a las columnas categóricas
label_encoders = {}
categorical_columns = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
    
X = data.drop(columns=['PlayerID', 'EngagementLevel'])
y = data['EngagementLevel'].values

# Dividir los datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

DatosEntrenamiento= len(X_train)

DatosPrueba= len(X_test)


# Normalizar los datos
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# =========================
# Encontrar el número óptimo de clústeres
# =========================

inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=42)
    kmeans.fit(X_train_normalized)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.show()

# =========================
# Aplicar KMeans con el número óptimo de clústeres
# =========================

optimal_clusters = 3  # Ajusta este valor basado en el gráfico del codo

start_time = time.time()

kmeans = KMeans(n_clusters=optimal_clusters, max_iter=1000, random_state=42)
kmeans.fit(X_train_normalized)

end_time = time.time()
training_time = end_time - start_time

y_pred_train = kmeans.labels_ 
y_pred_test = kmeans.predict(X_test_normalized)

silhouette_avg = silhouette_score(X_train_normalized, y_pred_train)

# =========================
# Evaluación de precisión
# =========================

# Crear una matriz de confusión para observar la correspondencia entre etiquetas originales y predichas
conf_matrix = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(optimal_clusters))
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión para Clústeres')
plt.show()

accuracy = accuracy_score(y_test, y_pred_test)

report = classification_report(y_test, y_pred_test, target_names=['Low', 'Medium', 'High'])


# =========================
# Reducción de Dimensionalidad y Visualización 2D
# =========================

pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_normalized)
centroids_pca_2d = pca_2d.transform(kmeans.cluster_centers_)

df_train_pca_2d = pd.DataFrame(data=X_train_pca_2d, columns=['PCA1', 'PCA2'])
df_train_pca_2d['Cluster'] = y_pred_train

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_train_pca_2d, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=60, alpha=0.5)
plt.scatter(centroids_pca_2d[:, 0], centroids_pca_2d[:, 1], s=60, c='red', marker='X', label='Centroides')

plt.title('Gráfico 2D de Dispersión de Clústeres con Centroides')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# =========================
# Visualización 3D
# =========================

pca_3d = PCA(n_components=3) 
X_train_pca_3d = pca_3d.fit_transform(X_train_normalized)
centroids_pca_3d = pca_3d.transform(kmeans.cluster_centers_)

df_train_pca_3d = pd.DataFrame(data=X_train_pca_3d, columns=['PCA1', 'PCA2', 'PCA3'])
df_train_pca_3d['Cluster'] = y_pred_train

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df_train_pca_3d['PCA1'], df_train_pca_3d['PCA2'], df_train_pca_3d['PCA3'],
                     c=df_train_pca_3d['Cluster'], cmap='viridis', s=60, alpha=0.5)

ax.scatter(centroids_pca_3d[:, 0], centroids_pca_3d[:, 1], centroids_pca_3d[:, 2],
           s=60, c='red', marker='X', label='Centroides')

ax.set_title('Gráfico 3D de Dispersión de Clústeres con Centroides')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')

# Añadir una barra de color para los clusters
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')

plt.legend(title='Cluster')
plt.show()

print(f"Numero total de datos: {total_datos}")
print(f'Numero de datos de entrenamiento: {DatosEntrenamiento}')
print(f'Numero de datos de entrenamiento: {DatosPrueba}')
print(f"Tiempo de entrenamiento del modelo: {training_time:.3f} segundos")
print(f'Índice de Silhouette para el conjunto de entrenamiento: {silhouette_avg:.3f}')
print(f"Precisión KMeans: {accuracy*100:.3f}")
print(f"Reporte de Clasificación\n: {report}")