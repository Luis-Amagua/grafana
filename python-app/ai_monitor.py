import pandas as pd
import numpy as np
import time
import threading
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prometheus_client import start_http_server, Summary, Gauge, Counter, Histogram
from flask import Flask, request, jsonify

# Fijamos la semilla para obtener resultados reproducibles
np.random.seed(42)

# Número de registros
n = 500

# Generación de datos
areas = np.random.randint(50, 500, n)  # Área en m²
antiguedad = np.random.randint(0, 51, n)  # Antigüedad en años

# Precios relacionados con las variables
precios = (areas * 300) - (antiguedad * 1000)
precios += np.random.normal(0, 20000, n)  # Agregar ruido aleatorio

# Crear DataFrame
df = pd.DataFrame({'Área': areas, 'Antigüedad (años)': antiguedad, 'Precio': precios})

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df[['Área', 'Antigüedad (años)']]
y = df['Precio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión múltiple
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Guardar el modelo entrenado
with open('model.pkl', 'wb') as f:
    pickle.dump(modelo, f)

# Evaluar el modelo
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

# ---------------------- MÉTRICAS PROMETHEUS ----------------------

# Definir métricas
prediction_time = Summary('prediction_time_seconds', 'Time taken for prediction')
prediction_error = Gauge('prediction_error', 'Prediction error')
prediction_count = Counter('prediction_count', 'Total number of predictions made')
prediction_latency = Histogram('prediction_latency_seconds', 'Latency of predictions', buckets=[0.1, 0.2, 0.5, 1, 2, 5])
average_price = Gauge('average_price', 'Average predicted price')
max_prediction_price = Gauge('max_prediction_price', 'Maximum predicted price')
min_prediction_price = Gauge('min_prediction_price', 'Minimum predicted price')
prediction_timeout = Counter('prediction_timeout', 'Number of timeouts during predictions')

@prediction_time.time()  # Medir el tiempo de predicción
def make_prediction(features):
    start_time = time.time()
    try:
        predicted_price = modelo.predict(features)  # Predicción
        error = np.abs(predicted_price - np.mean(y_test))  # Error aproximado
        prediction_error.set(error)  # Registrar error
        prediction_count.inc()  # Contar predicciones

        # Actualizar métricas
        average_price.set(np.mean(predicted_price))
        max_prediction_price.set(np.max(predicted_price))
        min_prediction_price.set(np.min(predicted_price))
        prediction_latency.observe(time.time() - start_time)  # Latencia

        return predicted_price

    except Exception as e:
        prediction_timeout.inc()
        print(f"Prediction failed: {str(e)}")
        return None

# ---------------------- SERVIDOR PROMETHEUS ----------------------
def start_prometheus_server():
    start_http_server(8000)  # Iniciar servidor en puerto 8000
    while True:
        test_features = np.array([[150, 10]])  # Valores de prueba
        make_prediction(test_features)  # Hacer predicción y actualizar métricas
        time.sleep(10)

# ---------------------- API FLASK ----------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" key'}), 400

        features = np.array(data['data']).reshape(1, -1)
        if features.shape[1] != 2:
            return jsonify({'error': 'Invalid input size. Expected 2 features'}), 400

        prediction = make_prediction(features)
        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Iniciar el servidor Prometheus en un hilo
thread = threading.Thread(target=start_prometheus_server, daemon=True)
thread.start()

# Iniciar la aplicación Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
