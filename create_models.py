import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Crear un modelo básico
model = RandomForestClassifier()
model.fit(np.array([[1,2,3,4]]), np.array([0]))
model.classes_ = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# Guardar el modelo
with open('models/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Crear un scaler estándar (en lugar de DummyScaler)
scaler = StandardScaler()
scaler.fit(np.array([[1,2,3,4]]))

# Guardar el scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Modelo y scaler guardados correctamente")