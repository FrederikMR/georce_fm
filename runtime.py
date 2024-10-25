#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:52:36 2024

@author: fmry
"""

#%% Sources

#https://jax.readthedocs.io/en/latest/faq.html

#%% Modules

import jax.numpy as jnp
from jax import jit, vmap

import timeit

import os

import pickle

#argparse
import argparse

from typing import Dict

#JAX Optimization
from jax.example_libraries import optimizers

from load_manifold import load_manifold
from geometry.finsler.manifolds import RiemannianNavigation
from geometry.riemannian.geodesics import GEORCE as RGEORCE
from geometry.riemannian.frechet_mean import GEORCE_FM as RGEORCE_FM
from geometry.riemannian.frechet_mean import JAXOptimization as RJAXOptimization
from geometry.riemannian.frechet_mean import ScipyOptimization as RScipyOptimization

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Sphere",
                        type=str)
    parser.add_argument('--geometry', default="Riemannian",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--N_data', default=100,
                        type=int)
    parser.add_argument('--sigma', default=1.0,
                        type=float)
    parser.add_argument('--T', default=1000,
                        type=int)
    parser.add_argument('--v0', default=1.5,
                        type=float)
    parser.add_argument('--method', default="GEORCE_FM",
                        type=str)
    parser.add_argument('--jax_lr_rate', default=0.01,
                        type=float)
    parser.add_argument('--tol', default=1e-5,
                        type=float)
    parser.add_argument('--max_iter', default=1000,
                        type=int)
    parser.add_argument('--sub_iter', default=10,
                        type=int)
    parser.add_argument('--line_search_iter', default=100,
                        type=int)
    parser.add_argument('--number_repeats', default=1,
                        type=int)
    parser.add_argument('--timing_repeats', default=1,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--save_path', default='timing_local/',
                        type=str)

    args = parser.parse_args()
    return args

#%% Timing

def estimate_method(FrechetMean, Geodesic, z_obs, M):
    
    args = parse_args()
    print("Computing method...")
    method = {} 
    z_mu, zt, sum_dist, grad, idx = FrechetMean(z_obs)
    print("\t-Estimate Computed")
    timing = []
    timing = timeit.repeat(lambda: FrechetMean(z_obs)[0].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    print("\t-Timing Computed")
    timing = jnp.stack(timing)
    
    zt = vmap(Geodesic, in_axes=(0,None))(z_obs, z_mu)
    sum_dist = M.length_frechet(zt, z_obs, z_mu)
    print("\t-Geodesics computed")
    
    method['mu'] = z_mu
    method['iterations'] = idx
    method['max_iter'] = args.max_iter
    method['tol'] = args.tol
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['sum_dist'] = sum_dist
    method['grad_norm'] = jnp.linalg.norm(grad.reshape(-1))
    
    return method

#%% Save times

def save_times(methods:Dict, save_path:str)->None:
    
    with open(save_path, 'wb') as f:
        pickle.dump(methods, f)
    
    return

#%% Force Field for Randers manifold

def force_fun(z, M):
    
    val = jnp.cos(z)
    
    val2 = jnp.sqrt(jnp.einsum('i,ij,j->', val, M.G(z), val))
    
    return jnp.sin(z)*val/val2

#%% Riemannian Run Time code

def riemannian_runtime()->None:
    
    args = parse_args()
    
    save_path = ''.join((args.save_path, f'riemannian/{args.manifold}/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path, args.method, 
                         f'_{args.manifold}', 
                         f'_d={args.dim}', 
                         f'_T={args.T}.pkl',
                         ))
    if os.path.exists(save_path):
        os.remove(save_path)
    
    z_obs, M, rho = load_manifold(args.manifold, 
                                  args.dim, 
                                  sigma=args.sigma, 
                                  N_data=args.N_data,
                                  seed=args.seed,
                                  )
    
    Geodesic = RGEORCE(M=M,
                       init_fun=None,
                       T=args.T,
                       max_iter=args.sub_iter,
                       line_search_method="soft",
                       line_search_params = {'rho':rho},
                       )
    
    methods = {}
    if args.method == "euclidean":
        euclidean_mean = jnp.mean(z_obs, axis=0)
        euclidean = {}
        euclidean['mu'] = euclidean_mean
        euclidean['iterations'] = None
        euclidean['max_iter'] = args.max_iter
        euclidean['tol'] = args.tol
        euclidean['mu_time'] = None
        euclidean['std_time'] = None
        euclidean['sum_dist'] = None
        euclidean['grad_norm'] = None
        methods['Euclidean'] = euclidean
        save_times(methods, save_path)
    elif args.method == "GEORCE_FM":
        FrechetMean = RGEORCE_FM(M=M,
                                 init_fun=None,
                                 T=args.T,
                                 line_search_method="soft",
                                 max_iter=args.max_iter,
                                 tol = args.tol,
                                 line_search_params={'rho': rho},
                                 )    
        methods['GEORCE_FM'] = estimate_method(jit(FrechetMean), jit(Geodesic), z_obs, M)
    elif args.method == "ADAM":
        FrechetMean = RJAXOptimization(M = M,
                                       init_fun=None,
                                       lr_rate=args.jax_lr_rate,
                                       optimizer=optimizers.adam,
                                       T=args.T,
                                       max_iter=args.max_iter,
                                       sub_iter = args.sub_iter,
                                       tol=args.tol,
                                       rho=rho,
                                       )
        methods['ADAM'] = estimate_method(jit(FrechetMean), jit(Geodesic), z_obs, M)
    elif args.method == "SGD":
        FrechetMean = RJAXOptimization(M = M,
                                       init_fun=None,
                                       lr_rate=args.jax_lr_rate,
                                       optimizer=optimizers.sgd,
                                       T=args.T,
                                       max_iter=args.max_iter,
                                       sub_iter = args.sub_iter,
                                       tol=args.tol,
                                       rho=rho,
                                       )
        methods['SGD'] = estimate_method(jit(FrechetMean), jit(Geodesic), z_obs, M)
    else:
        try:
            FrechetMean = RScipyOptimization(M = M,
                                             T=args.T,
                                             tol=args.tol,
                                             max_iter=args.max_iter,
                                             method=args.method,
                                             )
            methods[args.method] = estimate_method(FrechetMean, jit(Geodesic), z_obs, M)
        except:
            raise ValueError(f"Method, {args.method}, not defined!")
    save_times(methods, save_path)
    
    print(methods)
    print(z_obs[0])
    
    return

#%% Finsler Run Time code

def finsler_runtime()->None:
    
    args = parse_args()
    
    save_path = ''.join((args.save_path, f'finsler/{args.manifold}/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path, args.method, 
                         f'_{args.manifold}', 
                         f'_d={args.dim}', 
                         f'_T={args.T}.pkl',
                         ))
    if os.path.exists(save_path):
        os.remove(save_path)
    
    z_obs, RM, rho = load_manifold(args.manifold, 
                                   args.dim, 
                                   sigma=args.sigma, 
                                   N_data=args.N_data,
                                   seed=args.seed,
                                   )
    
    M = RiemannianNavigation(RM=RM,
                             force_fun=lambda z: force_fun(z, RM),
                             v0=args.v0,
                             )
    
    methods = {}
    
    if args.method == "euclidean":
        euclidean_mean = jnp.mean(z_obs, axis=0)
        euclidean = {}
        euclidean['mu'] = euclidean_mean
        euclidean['iterations'] = None
        euclidean['max_iter'] = args.max_iter
        euclidean['tol'] = args.tol
        euclidean['mu_time'] = None
        euclidean['std_time'] = None
        euclidean['sum_dist'] = None
        euclidean['grad_norm'] = None
        methods['Euclidean'] = euclidean
        save_times(methods, save_path)
        
    return
    
#%% main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.geometry == "Riemannian":
        riemannian_runtime()
    elif args.geometry == "Finsler":
        finsler_runtime()
    else:
        raise ValueError("Invalid geometry for runtime comparison.")