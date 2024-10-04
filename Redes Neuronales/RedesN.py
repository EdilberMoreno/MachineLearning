# Importar librerías necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Cargar el dataset
data = pd.read_csv('ruta')

data = data.tail(1000)

total_datos = len(data)

# Preprocesamiento de los datos: convertir etiquetas categóricas en valores numéricos
label_encoders = {}
for column in ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']:
    le = LabelEncoder()  
    data[column] = le.fit_transform(data[column]) 
    label_encoders[column] = le  

X = data.drop(columns=['PlayerID', 'EngagementLevel'])  
y = data['EngagementLevel'] 

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_entrenamiento = len(X_train)
num_prueba = len(X_test)
start_time = time.time()

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Función para graficar las 3 curvas de rendimiento en una gráfica
def plot_all_histories(histories, titles):
    plt.figure(figsize=(10, 6))
    
    for i, history in enumerate(histories):
        #plt.plot(history.history['accuracy'], label=f'{titles[i]} - Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'{titles[i]} - Precision')
    
    plt.title('Rendimiento de los modelos - Accuracy')
    plt.xlabel('Epocas')
    plt.ylabel('Presición')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función para graficar las matrices de confusión
def plot_confusion_matrices(models, X_test_scaled, y_test, titles):
    plt.figure(figsize=(18, 6))
    
    for i, model in enumerate(models):
        y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.subplot(1, 3, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matriz de confusión - {titles[i]}')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
    
    plt.tight_layout()
    plt.show()

# Modelo 1: Red neuronal de 3 capas (10-6-3)
def model_1():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 2: Red neuronal de 4 capas (9-6-4-3)
def model_2():
    model = Sequential()
    model.add(Dense(9, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modelo 3: Red neuronal de 3 capas (6-9-3)
def model_3():
    model = Sequential()
    model.add(Dense(6, activation='sigmoid', input_shape=(X_train.shape[1],)))
    model.add(Dense(9, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Entrenar y evaluar los modelos
models = [model_1(), model_2(), model_3()]
titles = ['Model 1: 10-6-3', 'Model 2: 9-6-4-3', 'Model 3: 6-9-3']

histories = []

for i, model in enumerate(models):
    print(f'Entrenando {titles[i]}')
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=0)
    histories.append(history)
    
    
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print(f'Reporte para el modelo {titles[i]}:\n', classification_report(y_test, y_pred))
    print(f'Precisión para el modelo {titles[i]}: {accuracy_score(y_test, y_pred)*100}\n')

# Graficar historial de entrenamiento de los 3 modelos
plot_all_histories(histories, titles)

# Graficar matrices de confusión
plot_confusion_matrices(models, X_test_scaled, y_test, titles)

end_time = time.time()
training_time = end_time - start_time
print(f"Tiempo de ejecución para entrenar los modelos: {training_time:.2f} segundos")
print(f"Total de datos: {total_datos}")
print(f"Número de datos de entrenamiento: {num_entrenamiento}")
print(f"Número de datos de prueba: {num_prueba}")