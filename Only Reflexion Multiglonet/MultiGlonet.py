from matplotlib import rcParams, rc
rcParams.update({'figure.autolayout': True})

import csv
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import pandas as pd

from Tarjet import *
from Phisicsparams import *
from utils import *
from MTMM import *
from tqdm import tqdm

from multiresglonet import GLOnet
from material_database import MatDatabase

from typing import TypeVarTuple


params = Params()
params.thickness_sup = 0.2
params.N_layers = 25

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params.materials = ['Al2O3','TiO2', 'SiO2']
params.user_define = True

if params.user_define:
  params.n_min = 1.09
  params.n_max = 2.6
  params.M_discretion_n = 200
  params.M_materials = params.M_discretion_n
  params.n_database = torch.tensor(np.array([np.linspace(params.n_min,params.n_max,params.M_discretion_n)]))
else:
  pass # definirlo en otro lado
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

params.alpha_sup =  5
params.numIter = 300
params.sigma = 0.035
params.batch_size = 75
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params.net = 'Res'
params.res_layers = 16                                                                             # Cantidad de bloques Residuales del bloque ResNet
params.res_dim = 256                                                                               # Cantidad de neuronas en la capa de entrada al bloque ResNet
params.noise_dim = 26                                                                              # Dimension de la Capa de entrada
params.lr = 0.05                                                                                   # Tasa de aprendizaje del optimizador Adam (learning rate)
params.beta1 = 0.9                                                                                 # Coeficiente de decaimiento para el momento del primer orden del optimizador Adam
params.beta2 = 0.99                                                                                # Coeficiente de decaimiento para el momento del segundo orden del optimizador Adam
params.weight_decay = 0.001                                                                        # Termino de decaimiento del peso para regularizar los pesos del generador durante la optimizacion
params.step_size = 40000                                                                           # Numero de epicas despues de las cuales se reduce la tasa de aprendizaje
params.gamma = 0.5                                                                                 # El factor de reduccion para la tasa de aprendizaje. Despues de cada step_size epocas, la tasa de aprendizaje se multiplica por gamma
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
params.condiciones = 2
physicsparams = PhysicsParams(params.condiciones, user_define=True)

n_interna = 1.2

physicsparams.n_bot_1 = n_interna
physicsparams.n_top_1 = 1
physicsparams.k_1 = 370
physicsparams.k_values.append(physicsparams.k_1)
physicsparams.theta_1 = 45
physicsparams.lambda_min_1 = 380
physicsparams.lambda_max_1 = 750
physicsparams.pol_1 = "TE"

physicsparams.n_bot_2 = 1
physicsparams.n_top_2 = n_interna
physicsparams.k_2 = 190
physicsparams.k_values.append(physicsparams.k_2)
physicsparams.theta_2 = 45
physicsparams.lambda_min_2 = 380
physicsparams.lambda_max_2 = 570
physicsparams.pol_2 = "TE"

physicsparams.generate_physics_params()

tarjet = Tarjet(params.condiciones, physicsparams.k_values, params.user_define)
tarjet.configure_targets()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if not params.user_define:
    params.matdatabase =  MatDatabase(params.materials)
    params.n_database = params.matdatabase.interp_wv(2 * math.pi/physicsparams.ks, params.materials, True)
    params.M_materials =  params.n_database.size(0)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
tarjet.tarjets["tarjet_1"].view(-1)[190:] = 1
tarjet.tarjets["tarjet_2"].view(-1)[115:] = 1

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

thicknesses_list = []
ref_idx_list = []

figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

histogram_dir = "Histogramas"
os.makedirs(histogram_dir, exist_ok=True)
    
Loss_dir =  "Losses"
if not os.path.exists(Loss_dir):
    os.makedirs(Loss_dir)


xlim = (300, 800)
ylim = (-0.5, 1.5)
vertical_lines = [380, 495, 570, 750]


 
for seed in range(2):                                                
  params.seed = seed

  torch.manual_seed(seed)
  glonet = GLOnet(params, physicsparams, tarjet)
  glonet.train()
  
  plt.figure(figsize = (20, 5))
  plt.subplot(131)
  plt.plot(glonet.loss_training)
  plt.ylabel('Loss', fontsize=18)
  plt.xlabel('Iterations', fontsize=18)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  
  plt.savefig(f"{Loss_dir}/Loss_seed_{seed}.png", bbox_inches='tight')
  plt.close()
  
  print(f"iteration{seed + 1 }")

  with torch.no_grad():
      params.k_test = 2 * math.pi / torch.linspace(0.3, 2.5, 50)
      params.theta_test = torch.linspace(0, math.pi/2.25, 50)
      (thicknesses, ref_index, result_mat) = glonet.evaluate(150, kvector=params.k_test, inc_angles=params.theta_test, grayscale=True)
      # Optimizacion
      
      reflex = MTMM_solver(params.condiciones, thicknesses, ref_index, physicsparams)
      FoM_reflex_total = sum(torch.pow(reflex[f'reflexion_{i}'] - tarjet.tarjets[f'tarjet_{i}'], 2).mean(dim=[1, 2, 3]) for i in range(1, 3))
      _, indices = torch.sort(FoM_reflex_total)
      opt_idx = indices[0]

      # Visualización del FoM total
      
      plt.figure(figsize=(10, 2))
      plt.subplot(131)
      plt.hist(FoM_reflex_total.cpu().detach().numpy(), alpha=0.5)
      plt.xlabel(f"FoM (n' = {n_interna})", fontsize=18)
      plt.xticks(fontsize=14)
      plt.yticks(fontsize=14)
      plt.savefig(f"{histogram_dir}/histograma_seed_{seed}.png", bbox_inches='tight')
      plt.close()
      
      # Encontrar el índice óptimo
      _, indices = torch.sort(FoM_reflex_total)
      opt_idx = indices[0]

      optimal_thicknesses = thicknesses[opt_idx]
      optimal_ref_idx = ref_index[opt_idx]
      
      thicknesses_list.append(optimal_thicknesses.view(-1).cpu().numpy().tolist())
      ref_idx_list.append(optimal_ref_idx.view(-1).cpu().numpy().tolist())

          
 
  fig, axs = plt.subplots(1, 2, figsize=(13, 3))  # 1 fila, 2 columnas
  fig.subplots_adjust(wspace=0.4)
  
  # Definir optimal_reflections
  optimal_reflections = {}
  
  for i in range(1, 3):  # Solo 1 y 2
      reflex_key = f'reflexion_{i}'
      optimal_reflections[reflex_key] = reflex[reflex_key][opt_idx]
      
      # Gráfico de reflexión óptima
      axs[i-1].plot(2 * math.pi / getattr(physicsparams, f'k_{i}') * 1000,
                    optimal_reflections[f'reflexion_{i}'][:, 0, 0].detach().numpy(),
                    "-", color="violet", label="Optimal Reflexion")
  
      # Gráfico de reflexión de tarjeta
      tarjet_color = "red" if i == 1 else "green"
      axs[i-1].plot(2 * math.pi / getattr(physicsparams, f'k_{i}') * 1000,
                    tarjet.tarjets[f"tarjet_{i}"].view(-1),
                    ".-", color=tarjet_color, label=f"Tarjet Reflexion {i}", markersize=2.5)
  
      axs[i-1].set_xlabel("Wavelength (nm)", fontsize=16)
      axs[i-1].set_ylabel("Reflection", fontsize=16)
      axs[i-1].legend(fontsize=10)
      axs[i-1].set_title(f"Reflexion {i}", fontsize=20)
      axs[i-1].tick_params(axis='both', which='major', labelsize=14)
      axs[i-1].set_xlim(xlim)
      axs[i-1].set_ylim(ylim)
  
      # Agregar líneas verticales punteadas para reflexiones
      for line_pos in vertical_lines:
          axs[i-1].axvline(x=line_pos, linestyle='--', color='gray')
          
  fig.savefig(os.path.join(figures_dir, f"reflexion_seed_{seed}.png"))
  plt.close(fig)

with open('optimal_thicknesses.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(thicknesses_list)

# Archivo CSV para los índices de refracción
with open('optimal_ref_idx.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(ref_idx_list)
    

    