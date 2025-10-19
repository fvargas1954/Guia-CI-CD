import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print("🚀 Iniciando entrenamiento del modelo...")

# Forzar tracking URI con ruta absoluta del directorio actual
tracking_dir = os.path.join(os.getcwd(), "mlruns")
os.makedirs(tracking_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{tracking_dir}")
mlflow.set_experiment("ci-cd-mlflow-local")

print(f"📁 MLflow tracking URI: {tracking_dir}")

# Cargar datos
print("📊 Cargando dataset de diabetes...")
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📦 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Entrenar modelo
print("🤖 Entrenando modelo de regresión lineal...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"🎯 MSE: {mse:.4f}")

# Registrar en MLflow
with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.sklearn.log_model(model, "model")
    
    print("💾 Modelo y métricas registrados en MLflow")

print("✅ Entrenamiento completado con éxito!")