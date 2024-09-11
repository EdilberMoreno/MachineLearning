import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
import time

# Cargar el archivo CSV en un Dataset
ruta = 'ruta'
data = pd.read_csv(ruta)

data = data.head(5000)

total_datos = len(data)

# Preprocesamiento de los datos: convertir etiquetas categóricas en valores numéricos
label_encoders = {}
for column in ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']:
    le = LabelEncoder()  
    data[column] = le.fit_transform(data[column]) 
    label_encoders[column] = le  

X = data.drop(columns=['PlayerID', 'EngagementLevel'])  
y = data['EngagementLevel'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_entrenamiento = len(X_train)
num_prueba = len(X_test)

# Tomar el tiempo antes de entrenar el modelo
start_time = time.time()

rf_clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
rf_clf.fit(X_train, y_train)

end_time = time.time()
training_time = end_time - start_time

# Realizar predicciones sobre el conjunto de prueba
y_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)  
report = classification_report(y_test, y_pred, target_names=label_encoders['EngagementLevel'].classes_)  

conf_matrix = confusion_matrix(y_test, y_pred)

# Obtener la profundidad de cada árbol en el bosque
depths = [tree.tree_.max_depth for tree in rf_clf.estimators_]

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoders['EngagementLevel'].classes_, 
            yticklabels=label_encoders['EngagementLevel'].classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Matriz de Confusión')
plt.tight_layout()  
plt.show()

# Visualizar la importancia de las características
feature_importances = rf_clf.feature_importances_
features = X.columns

plt.figure(figsize=(13, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.title('Importancia de las características en el Random Forest')
plt.tight_layout() 
plt.show()

# Visualización árbol del bosque aleatorio
plt.figure(figsize=(13, 5))
plot_tree(rf_clf.estimators_[0], 
          feature_names=X.columns, 
          class_names=label_encoders['EngagementLevel'].classes_, 
          filled=True, 
          rounded=True, 
          fontsize=8, 
          max_depth=3)  
plt.title("Árbol Random Forest")
plt.tight_layout()  
plt.show()

print(X.head(10))
print(f"Total de datos: {total_datos}")
print(f"Número de datos de entrenamiento: {num_entrenamiento}")
print(f"Número de datos de prueba: {num_prueba}")
print(f"Accuracy: {accuracy * 100:.2f}%")  
print("Classification Report:")
print(report)  
print(f"Tiempo de ejecución para entrenar el modelo: {training_time:.2f} segundos")
# Mostrar las profundidades
print(f"Profundidades de cada árbol en el Random Forest: {depths}")
# Estadísticas adicionales sobre las profundidades
print(f"Profundidad máxima de los árboles: {max(depths)}")
print(f"Profundidad mínima de los árboles: {min(depths)}")
print(f"Profundidad promedio de los árboles: {np.mean(depths):.2f}")
