import torch
import math
import numpy as np

class Tarjet:
    def __init__(self, condiciones, k_values, user_define):
        self.condiciones = condiciones
        self.k_values = k_values
        self.user_define = user_define
        

    def configure_targets(self, manual = False):
        self.tarjets = {}
        if self.user_define:
            for i in range(1, self.condiciones + 1):
                k_value = self.k_values[i - 1]  # Usar el valor de k de PhysicsParams
                reflection_values = np.zeros(k_value)  # Inicialización con ceros
                if manual:
                    for j in range(k_value):
                        reflection_values[j] = float(input(f"Ingrese el valor de reflexión para tarjet_{i} en la posición {j+1}: "))
                        target_reflection = torch.from_numpy(reflection_values).unsqueeze(1).unsqueeze(1).unsqueeze(0)
                else:
                    target_reflection = torch.from_numpy(reflection_values).unsqueeze(1).unsqueeze(1).unsqueeze(0)

                self.tarjets[f"tarjet_{i}"] = target_reflection
        else:
            for i in range(1, self.condiciones + 1):
                reflection_values = np.zeros(self.k_values[0]) 
                if manual:
                    for j in range(k_value):
                        reflection_values[j] = float(input(f"Ingrese el valor de reflexión para tarjet_{i} en la posición {j+1}: "))
                        target_reflection = torch.from_numpy(reflection_values).unsqueeze(1).unsqueeze(1).unsqueeze(0)
                else:
                    target_reflection = torch.from_numpy(reflection_values).unsqueeze(1).unsqueeze(1).unsqueeze(0)

                self.tarjets[f"tarjet_{i}"] = target_reflection          