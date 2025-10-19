import os
import mlflow
import sys

print("🔍 Iniciando validación del modelo...")

# Forzar tracking URI con ruta absoluta del directorio actual
tracking_dir = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{tracking_dir}")

print(f"📁 MLflow tracking URI: {tracking_dir}")

# Obtener el experimento
experiment = mlflow.get_experiment_by_name("ci-cd-mlflow-local")

if experiment is None:
    print("❌ No se encontró el experimento. Ejecuta train.py primero.")
    sys.exit(1)

# Buscar el último run
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

if runs.empty:
    print("❌ No hay runs disponibles para validar.")
    sys.exit(1)

# Obtener MSE del último run
mse = runs.iloc[0]["metrics.mse"]

print(f"📊 MSE del modelo: {mse:.4f}")

# Validar umbral
THRESHOLD = 3000

if mse > THRESHOLD:
    print(f"❌ MSE ({mse:.4f}) supera el umbral ({THRESHOLD})")
    print("❌ Modelo NO apto para producción")
    sys.exit(1)
else:
    print(f"✅ MSE ({mse:.4f}) es aceptable (< {THRESHOLD})")
    print("✅ Modelo validado exitosamente")
    sys.exit(0)