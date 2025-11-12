from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

print("Cargando dataset de Vinos...")
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

# De las 13 características, seleccionamos las 4 más importantes.
features_importantes = [
    'proline', 
    'flavanoids', 
    'color_intensity', 
    'alcohol'
]

X_subset = X[features_importantes]

print(f"Entrenando modelo usando solo estas 4 características: {features_importantes}")

# Entrenar el modelo solo con el subconjunto de características
clf = LogisticRegression(random_state=0, max_iter=2000).fit(X_subset, y)

# Guardar el nuevo modelo
with open('modelo_wine.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("¡Modelo de Vinos guardado en modelo_wine.pkl!")
print("\nValores de ejemplo para los sliders:")
print(X_subset.describe())