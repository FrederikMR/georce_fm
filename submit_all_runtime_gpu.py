#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit jobs

def submit_job():
    
    os.system("bsub < submit_runtime.sh")
    
    return

#%% Generate jobs

def generate_job(manifold, d, m, geometry, tol, batch_size, N_data):

    batch_name = f"{batch_size}".replace('.', '')
    with open ('submit_runtime.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {geometry[0]}{manifold}{d}_{batch_name}_{m}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -env "LSB_JOB_REPORT_MAIL=N"
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    module swap python3/3.10.12
    
    python3 runtime.py \\
        --manifold {manifold} \\
        --geometry {geometry} \\
        --dim {d} \\
        --batch_size {batch_size} \\
        --N_data {N_data} \\
        --T 100 \\
        --v0 1.5 \\
        --method {m} \\
        --jax_lr_rate 0.01 \\
        --tol {tol} \\
        --max_iter 1000 \\
        --number_repeats 5 \\
        --timing_repeats 5 \\
        --save_path timing_gpu/ \\
    ''')
    
    return
                            
#%% Loop fixed jobs

def loop_fixed_jobs(wait_time = 1.0):
    
    geomtries = ['Riemannian', 'Finsler'] #, "Lorentz"]
    N_data = 100
    batch_size = 1.0

    runs = {"Sphere": [[2,3,5,10,20,50,100],1e-4],
            "Ellipsoid": [[2,3,5,10,20,50,100],1e-4],
            "SPDN": [[2,3],1e-4],
            "T2": [[2],1e-4],
            "H2": [[2],1e-4],
            "Paraboloid": [[2], 1e-4],
            "HyperParaboloid": [[2], 1e-4],
            "Gaussian": [[2],1e-4],
            "Frechet": [[2],1e-4],
            "Cauchy": [[2],1e-4],
            "Pareto": [[2],1e-4],
            "celeba": [[32],1e-3],
            "svhn": [[32],1e-3],
            "mnist": [[8],1e-3],
            }
    
    runs = {
            "HyperbolicParaboloid": [[2], 1e-4],
            }
    
    jax_methods = ["sgd", "rmsprop_momentum", "rmsprop", "adamax", "adam", "adagrad"] #JAX
    methods = ['GEORCE_FM', 'euclidean'] + jax_methods

    for geo in geomtries:
        for man, vals in runs.items():
            dims, tol = vals[0], vals[1]
            for d in dims:
                for m in methods:
                    time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                    generate_job(man, d, m, geo, tol, batch_size, N_data)
                    try:
                        submit_job()
                    except:
                        time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                        try:
                            submit_job()
                        except:
                            print(f"Job script with {geo}, {man}, {d}, {tol} failed!")
                            
#%% Loop jobs

def loop_adaptive_jobs(wait_time = 1.0):
    
    geomtries = ['Riemannian', 'Finsler'] #, "Lorentz"]
    batches = [0.01, 0.1, 0.25]
    N_data = 1_000

    runs = {"Sphere": [[2,3,5,10,20,50,100],1e-4],
            "Ellipsoid": [[2,3,5,10,20,50,100],1e-4],
            "SPDN": [[2,3],1e-4],
            "T2": [[2],1e-4],
            "H2": [[2],1e-4],
            "Paraboloid": [[2], 1e-4],
            "HyperParaboloid": [[2], 1e-4],
            "Gaussian": [[2],1e-4],
            "Frechet": [[2],1e-4],
            "Cauchy": [[2],1e-4],
            "Pareto": [[2],1e-4],
            "celeba": [[32],1e-3],
            "svhn": [[32],1e-3],
            "mnist": [[8],1e-3],
            }
    
    runs = {
            "HyperbolicParaboloid": [[2], 1e-4],
            }
    
    jax_methods = ["sgd", "rmsprop_momentum", "rmsprop", "adamax", "adam", "adagrad"] #JAX
    ada_jax_methods = [''.join(("ADA", jm)) for jm in jax_methods]
    methods = ['GEORCE_AdaFM'] + ada_jax_methods

    for geo in geomtries:
        for man, vals in runs.items():
            dims, tol = vals[0], vals[1]
            for d in dims:
                for batch in batches:
                    for m in methods:
                        time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                        generate_job(man, d, m, geo, tol, batch, N_data)
                        try:
                            submit_job()
                        except:
                            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                            try:
                                submit_job()
                            except:
                                print(f"Job script with {geo}, {man}, {d}, {tol} failed!")

#%% main

if __name__ == '__main__':
    
    loop_fixed_jobs(1.0)
    loop_adaptive_jobs(1.0)