import torch
import math

class PhysicsParams:
    def __init__(self, condiciones, user_define):
        self.condiciones = condiciones
        self.user_define = user_define
        self.k_values = []  # Lista para almacenar los valores de k

    def collect_user_input(self):
        if self.user_define:
            for i in range(1, self.condiciones + 1):
                while True:
                    try:
                        setattr(self, f"n_bot_{i}", float(input(f"Ingrese el valor para n_bot_{i}: ")))
                        setattr(self, f"n_top_{i}", float(input(f"Ingrese el valor para n_top_{i}: ")))
                        k_value = int(input(f"Ingrese el valor entero para k_{i}: "))
                        setattr(self, f"k_{i}", k_value)
                        self.k_values.append(k_value)  # Almacenar el valor de k
                        setattr(self, f"theta_{i}", float(input(f"Ingrese el valor para theta_{i} (en grados): ")))
                        setattr(self, f"lambda_min_{i}", float(input(f"Ingrese el valor para lambda_min_{i} (en um): ")))
                        setattr(self, f"lambda_max_{i}", float(input(f"Ingrese el valor para lambda_max_{i} (en um): ")))
                        break
                    except ValueError:
                        print("Por favor, ingrese un valor válido.")

                while True:
                    try:
                        pol = str(input(f"Ingrese el valor para pol_{i} u cualquier otro para ambos: "))
                        if pol in ["TE", "TM"]:
                            setattr(self, f"pol_{i}", pol)
                        else:
                            setattr(self, f"pol_{i}", "(TE and TM)")
                        break
                    except ValueError:
                        print("Por favor, ingrese un valor válido para pol_{i}.")

        else:
            setattr(self, "lambda_min", float(input(f"Ingrese el valor para lambda_min (en um): ")))
            setattr(self, "lambda_max", float(input(f"Ingrese el valor para lambda_max (en um): ")))
            k_value = int(input(f"Ingrese el valor entero para la discretizacion k: "))
            self.k_values.append(k_value)
            setattr(self, "k", k_value)

            for i in range(1, self.condiciones + 1):
                while True:
                    try:
                        setattr(self, f"n_bot_{i}", float(input(f"Ingrese el valor para n_bot_{i}: ")))
                        setattr(self, f"n_top_{i}", float(input(f"Ingrese el valor para n_top_{i}: ")))

                        pol = str(input(f"Ingrese el valor para pol_{i} u cualquier otro para ambos: "))
                        if pol in ["TE", "TM"]:
                            setattr(self, f"pol_{i}", pol )
                        else:
                            setattr(self, f"pol_{i}", "(TE and TM)")

                        setattr(self, f"theta_{i}", float(input(f"Ingrese el valor para theta_{i} (en grados): ")))
                        break
                    except ValueError:
                        print("Por favor, ingrese un valor válido.")

    def generate_physics_params(self):
        if self.user_define:
            for i in range(1, self.condiciones + 1):
                k_value = getattr(self, f"k_{i}")
                lambda_min_nm = getattr(self, f"lambda_min_{i}")
                lambda_max_nm = getattr(self, f"lambda_max_{i}")

                lambda_min_um = lambda_min_nm / 1000
                lambda_max_um = lambda_max_nm / 1000
                
                setattr(self, f"k_{i}", 2 * math.pi / torch.linspace(lambda_min_um, lambda_max_um, k_value))
                theta_radians = getattr(self, f"theta_{i}") * math.pi / 180
                setattr(self, f"n_bot_{i}", torch.tensor([getattr(self, f"n_bot_{i}")]))
                setattr(self, f"n_top_{i}", torch.tensor([getattr(self, f"n_top_{i}")]))
                setattr(self, f"theta_{i}", torch.tensor([theta_radians]))
        else:
            lambda_min_nm = getattr(self, "lambda_min")
            lambda_max_nm = getattr(self, "lambda_max")

            lambda_min_um = lambda_min_nm / 1000
            lambda_max_um = lambda_max_nm / 1000

            k_value = getattr(self, "k")
            setattr(self, 'ks', 2 * math.pi / torch.linspace(lambda_min_um, lambda_max_um, k_value))

            for i in range(1, self.condiciones + 1):
                setattr(self, f"k_{i}", self.ks ) # inicializa con todos los mismas discretizaciones
                setattr(self, f"n_bot_{i}", torch.tensor([getattr(self, f"n_bot_{i}")]))
                setattr(self, f"n_top_{i}", torch.tensor([getattr(self, f"n_top_{i}")]))
                theta_radians = getattr(self, f"theta_{i}") * math.pi / 180
                setattr(self, f"theta_{i}", torch.tensor([theta_radians]))

    def view_attributes(self):
        for attr, value in self.__dict__.items():
            if attr != "condiciones":
                print(f"{attr}: {value}")