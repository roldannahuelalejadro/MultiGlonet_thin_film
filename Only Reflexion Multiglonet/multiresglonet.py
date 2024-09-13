import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from MTMM import *
from net import *

class GLOnet():
    def __init__(self, params, physicsparams, tarjet):
        # GPU 
        self.cpu = torch.cpu
        self.dtype = torch.FloatTensor
        self.physicsparams = physicsparams
        self.tarjet = tarjet
        # Construct the generator network
        if params.net == 'Res':
            self.generator = ResGenerator(params)
        else:
            self.generator = Generator(params)
        
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr, betas=(params.beta1, params.beta2), weight_decay=params.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params.step_size, gamma=params.step_size)
        
        # Training parameters
        self.noise_dim = params.noise_dim
        self.numIter = params.numIter
        self.batch_size = params.batch_size
        self.sigma = params.sigma
        self.alpha_sup = params.alpha_sup
        self.iter0 = 0
        self.alpha = 0.1
    
        # Simulation parameters
        self.user_define = params.user_define
        if params.user_define:
            self.n_database = params.n_database
        else:
            self.materials = params.materials
            self.matdatabase = params.matdatabase
        
        # Physical parameters
        self.condiciones = params.condiciones
        
        # Import characteristics from physicsparams
        for attr_name, attr_value in physicsparams.__dict__.items():
            if hasattr(attr_value, 'type'):
                setattr(self, attr_name, attr_value.type(self.dtype))
            else:
                setattr(self, attr_name, attr_value)

        # Import target reflections as a dictionary
        self.target_reflections = tarjet.tarjets
        
        # Training history
        self.loss_training = []
        self.refractive_indices_training = []
        self.thicknesses_training = []
        
    def train(self, show_update = True):
        self.generator.train()
            
        with tqdm(total=self.numIter) as t:
            it = self.iter0  
            while True:
                it += 1 
                normIter = it / self.numIter
                self.update_alpha(normIter)
                if it > self.numIter:
                    return 
                z = self.sample_z(self.batch_size)
                thicknesses, refractive_indices, _ = self.generator(z, self.alpha) # quiero que tengan las mismas dimensiones
                resultados_reflection = MTMM_solver(self.condiciones, thicknesses, refractive_indices, self) # y cuyos resultados sean coherenetes
                self.optimizer.zero_grad()
                perdidas = self.calcular_perdidas_globales(resultados_reflection)
                g_loss = self.sumar_perdidas(perdidas)
                self.record_history(g_loss, thicknesses, refractive_indices)
                g_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if show_update:
                    t.update()
 
    def evaluate(self, num_devices, kvector = None, inc_angles = None, grayscale=True):
        if kvector is None:
            kvector = self.k
        if inc_angles is None:
            inc_angles = self.theta

        self.generator.eval()
        z = self.sample_z(num_devices)
        thicknesses, refractive_indices, P = self.generator(z, self.alpha)
        result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer

        if not grayscale:
            if self.user_define:
                n_database = self.n_database # do not support dispersion
            else:
                n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)
            
            one_hot = torch.eye(len(self.materials)).type(self.dtype)
            ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
        else:
            if self.user_define:
                ref_idx = refractive_indices
            else:
                n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)
                ref_idx = torch.sum(P.unsqueeze(-1) * n_database, dim=2)

        
        return (thicknesses, ref_idx, result_mat)

    def optimizer_evaluate(self, num_devices, kvector=None, inc_angles=None, grayscale=True):
        thicknesses, ref_idx, result_mat = self.evaluate(num_devices, kvector, inc_angles, grayscale)

        reflex = MTMM_solver(self.condiciones, thicknesses, ref_idx, self.physicsparams)
        optimal_reflections = {}

        # Inicializa FoM_total como tensor de ceros del mismo tamaño que el primer error cuadrático
        FoM_total = torch.zeros_like(torch.pow(reflex[f'reflexion_1'] - self.target_reflections[f'tarjet_1'], 2))

        for i in range(1, self.physicsparams.condiciones + 1):
            reflex_key = f'reflexion_{i}'
            tarjet_key = f'tarjet_{i}'
            r = reflex[reflex_key]
            FoM_total += torch.pow(r - self.target_reflections[tarjet_key], 2)

        FoM = FoM_total.mean(dim=[1, 2, 3])
        _, indices = torch.sort(FoM)
        opt_idx = indices[0]

        optimal_materials = [self.materials[result_mat[opt_idx, i]] for i in range(result_mat.size(1))]

        optimal_thicknesses = thicknesses[opt_idx]
        optimal_ref_idx = ref_idx[opt_idx]
        
        for i in range(1, self.physicsparams.condiciones + 1):
            reflex_key = f'reflexion_{i}'
            optimal_reflections[reflex_key] = reflex[reflex_key][opt_idx]

        return optimal_thicknesses, optimal_ref_idx, optimal_reflections, optimal_materials

    def update_alpha(self, normIter):
        self.alpha = round(normIter/0.05) * self.alpha_sup + 1.
        
    def sample_z(self, batch_size):
        return (torch.randn(batch_size, self.noise_dim, requires_grad=True)).type(self.dtype)

    def global_loss_function(self, reflection):
        return -torch.mean(torch.exp(-torch.mean(torch.pow(reflection - self.target_reflections, 2), dim=(1, 2, 3)) / self.sigma))

    def sumar_perdidas(self, perdidas_globales):
        primer_perdida = next(iter(perdidas_globales.values()))
        suma_total = torch.zeros_like(primer_perdida)
        for perdida in perdidas_globales.values():
            suma_total += perdida
        return suma_total

    def record_history(self, loss, thicknesses, refractive_indices):
        self.loss_training.append(loss.detach())
        self.thicknesses_training.append(thicknesses.mean().detach())
        self.refractive_indices_training.append(refractive_indices.mean().detach())

    def calcular_perdidas_globales(self, resultados_reflexion):
        perdidas_globales = {}
        for key, tensor_reflexion in resultados_reflexion.items():
            # Convertir la clave de 'reflexion_{i}' a 'tarjet_{i}'
            tarjet_key = key.replace('reflexion', 'tarjet')

            perdida = -torch.mean(torch.exp(-torch.mean(torch.pow(tensor_reflexion - self.target_reflections[tarjet_key], 2), dim=(1, 2, 3)) / self.sigma))

            perdidas_globales[key] = perdida
        return perdidas_globales
    
    def viz_training(self):
        plt.figure(figsize = (20, 5))
        plt.subplot(131)
        plt.plot(self.loss_training)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Iterations', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)