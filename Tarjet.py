import torch
import math
import numpy as np


class Tarjet:
    def __init__(self, condiciones, k_values, user_define):
        self.condiciones = condiciones
        self.k_values = k_values
        self.user_define = user_define

    def configure_targets(self, manual=False):
        self.tarjets_reflections = {}
        self.tarjets_transmisions = {}
        
        for i in range(1, self.condiciones + 1):
            if self.user_define:
                k_value = self.k_values[i - 1]
            else:
                k_value = self.k_values[0]

            reflection_values = np.zeros(k_value)
            transmisions_values = np.zeros(k_value)

            if manual:
                for j in range(k_value):
                    reflection_values[j] = float(input(f"Ingrese el valor de reflexión para tarjet_{i} en la posición {j+1}: "))
                    transmisions_values[j] = float(input(f"Ingrese el valor de transmisión para tarjet_{i} en la posición {j+1}: "))

            target_reflection = torch.from_numpy(reflection_values).unsqueeze(1).unsqueeze(1).unsqueeze(0)
            target_transmision = torch.from_numpy(transmisions_values).unsqueeze(1).unsqueeze(1).unsqueeze(0)

            self.tarjets_reflections[f"tarjet_{i}"] = target_reflection
            self.tarjets_transmisions[f"tarjet_{i}"] = target_transmision
