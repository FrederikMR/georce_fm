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

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_runtime.sh")
    
    return

#%% Generate jobs

def generate_job(manifold, d, method, geometry, tol, batch_size, N_data):

    with open ('submit_runtime.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {method}_{geometry[0]}{manifold}{d}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
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
        --method {method} \\
        --jax_lr_rate 0.01 \\
        --tol {tol} \\
        --max_iter 1000 \\
        --number_repeats 5 \\
        --timing_repeats 5 \\
        --save_path timing_gpu/ \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    geomtries = ['Riemannian', 'Finsler'] #, "Lorentz"]
    jax_methods = ["sgd", "rmsprop_momentum", "rmsprop", "adamax", "adam", "adagrad"] #JAX
    methods = ['GEORCE_FM', 'euclidean'] + jax_methods

    N_data = 100
    runs = {"Sphere": [[2,3,5,10,20,50,100,250,500,1000],1e-3],
            "Ellipsoid": [[2,3,5,10,20,50,100,250,500,1000],1e-3],
            "SPDN": [[2,3],1e-3],
            "T2": [[2],1e-3],
            "H2": [[2],1e-3],
            "Gaussian": [[2],1e-3],
            "Frechet": [[2],1e-3],
            "Cauchy": [[2],1e-3],
            "Pareto": [[2],1e-3],
            "celeba": [[32],1e-3],
            "svhn": [[32],1e-3],
            "mnist": [[8],1e-3],
            }

    for geo in geomtries:
        for man, vals in runs.items():
            dims, tol = vals[0], vals[1]
            for d in dims:
                for m in methods:
                    time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                    generate_job(man, d, m, geo, tol, 1.0, N_data)
                    try:
                        submit_job()
                    except:
                        time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                        try:
                            submit_job()
                        except:
                            print(f"Job script with {geo}, {man}, {m}, {d}, {tol}, {N_data} failed!")
                            
#%% Loop adaptive jobs

def loop_adaptive_jobs(wait_time = 1.0):
    
    geomtries = ['Riemannian', 'Finsler'] #, "Lorentz"]
    jax_methods = ["sgd", "rmsprop_momentum", "rmsprop", "adamax", "adam", "adagrad"] #JAX
    jax_methods = [''.join(("ADA", jm)) for jm in jax_methods]
    methods = ['GEORCE_AdaFM', 'euclidean'] + jax_methods
    
    N_data = 1_000
    batches = [0.01, 0.05, 0.1, 0.5, 1.0]

    runs = {"Sphere": [[2,3,5,10,20,50,100,250,500,1000],1e-3],
            "Ellipsoid": [[2,3,5,10,20,50,100,250,500,1000],1e-3],
            "SPDN": [[2,3],1e-3],
            "T2": [[2],1e-3],
            "H2": [[2],1e-3],
            "Gaussian": [[2],1e-3],
            "Frechet": [[2],1e-3],
            "Cauchy": [[2],1e-3],
            "Pareto": [[2],1e-3],
            "celeba": [[32],1e-3],
            "svhn": [[32],1e-3],
            "mnist": [[8],1e-3],
            }

    for geo in geomtries:
        for batch in batches:
            for man, vals in runs.items():
                dims, tol = vals[0], vals[1]
                for d in dims:
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
                                print(f"Job script with {geo}, {man}, {m}, {d}, {tol}, {N_data} failed!")

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)
    loop_adaptive_jobs(1.0)