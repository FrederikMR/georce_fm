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
from geometry.finsler.manifolds import RiemannianNavigation as FRiemannianNavigation
from geometry.lorentz.manifolds import RiemannianNavigation as LRiemannianNavigation

from geometry.riemannian.geodesics import GEORCE as RGEORCE
from geometry.riemannian.frechet_mean import GEORCE_AdaFM as RGEORCE_AdaFM
from geometry.riemannian.frechet_mean import JAXAdaOptimization as RJAXAdaOptimization

from geometry.finsler.geodesics import GEORCE as FGEORCE
from geometry.finsler.frechet_mean import GEORCE_AdaFM as FGEORCE_AdaFM
from geometry.finsler.frechet_mean import JAXAdaOptimization as FJAXAdaOptimization

from geometry.lorentz.geodesics import GEORCE as LGEORCE
from geometry.lorentz.frechet_mean import GEORCE_AdaFM as LGEORCE_AdaFM
from geometry.lorentz.frechet_mean import JAXAdaOptimization as LJAXAdaOptimization

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--manifold', default="Ellipsoid",
                        type=str)
    parser.add_argument('--geometry', default="Riemannian",
                        type=str)
    parser.add_argument('--dim', default=2,
                        type=int)
    parser.add_argument('--batch_size', default=1.0,
                        type=float)
    parser.add_argument('--N_data', default=100,
                        type=int)
    parser.add_argument('--sigma', default=1.0,
                        type=float)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--v0', default=1.5,
                        type=float)
    parser.add_argument('--method', default="GEORCE_FM",
                        type=str)
    parser.add_argument('--jax_lr_rate', default=0.01,
                        type=float)
    parser.add_argument('--tol', default=1e-3,
                        type=float)
    parser.add_argument('--max_iter', default=100,
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
    z_mu, zt, stop_criteria = FrechetMean(z_obs)
    print("\t-Estimate Computed")
    timing = []
    timing = timeit.repeat(lambda: FrechetMean(z_obs)[0].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    print("\t-Timing Computed")
    timing = jnp.stack(timing)
    
    zt = vmap(Geodesic, in_axes=(0,None))(z_obs, z_mu)
    sum_geodesic_dist = M.length_frechet(zt, z_obs, z_mu)
    print("\t-Geodesics computed")
    
    method['mu'] = z_mu
    method['zt'] = zt
    method['iterations'] = idx
    method['max_iter'] = args.max_iter
    method['tol'] = args.tol
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['sum_geodesic_dist'] = sum_geodesic_dist
    method['grad_norm'] = jnp.linalg.norm(grad.reshape(-1))
    
    return method

#%% Timing

def estimate_lorentz(FrechetMean, Geodesic, z_obs, M):
    
    args = parse_args()
    print("Computing method...")
    method = {} 
    z_mu, ts, zs, grad, idx = FrechetMean(0.0,z_obs)
    print("\t-Estimate Computed")
    timing = []
    timing = timeit.repeat(lambda: FrechetMean(0.0, z_obs)[0].block_until_ready(), 
                           number=args.number_repeats, 
                           repeat=args.timing_repeats)
    print("\t-Timing Computed")
    timing = jnp.stack(timing)
    
    ts, zs = vmap(Geodesic, in_axes=(None,0,None))(0.0,z_obs, z_mu)
    sum_geodesic_dist = M.length_frechet(0.0, ts, zs, z_obs, z_mu)
    print("\t-Geodesics computed")
    
    method['mu'] = z_mu
    method['ts'] = ts
    method['zs'] = zs
    method['iterations'] = idx
    method['max_iter'] = args.max_iter
    method['tol'] = args.tol
    method['mu_time'] = jnp.mean(timing)
    method['std_time'] = jnp.std(timing)
    method['sum_geodesic_dist'] = sum_geodesic_dist
    method['grad_norm'] = jnp.linalg.norm(grad.reshape(-1))
    
    return method

#%% Save times

def save_times(methods:Dict, save_path:str)->None:
    
    with open(save_path, 'wb') as f:
        pickle.dump(methods, f)
    
    return

#%% Force Field for Randers manifold

def force_fun(t, z, M):
    
    val = jnp.cos(t*z)
    
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
    
    z_obs, M = load_manifold(args.manifold, 
                             args.dim, 
                             N_data=args.N_data,
                             )
    
    Geodesic = RGEORCE(M=M,
                       init_fun=None,
                       T=args.T,
                       max_iter=10,
                       line_search_method="soft",
                       line_search_params = {'rho':0.5},
                       )
    
    batch_size = int(args.batch_size*args.N_data)
    
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
    elif args.method == "GEORCE_AdaFM":
        FrechetMean = RGEORCE_AdaFM(M=M,
                                    init_fun=None,
                                    alpha=0.9,
                                    sub_iter=2,
                                    T=args.T,
                                    max_iter=args.max_iter,
                                    tol = args.tol,
                                    line_search_params={'rho': 0.95},
                                    )    
        methods['GEORCE_AdaFM'] = estimate_method(jit(lambda z: FrechetMean(z,batch_size)), 
                                                  jit(Geodesic), 
                                                  z_obs, 
                                                  M,
                                                  )
    elif args.method == "ADAM":
        FrechetMean = RJAXAdaOptimization(M = M,
                                          init_fun=None,
                                          lr_rate=args.jax_lr_rate,
                                          optimizer=optimizers.adam,
                                          T=args.T,
                                          max_iter=args.max_iter,
                                          tol=args.tol,
                                          )
        methods['AdaADAM'] = estimate_method(jit(lambda z: FrechetMean(z,batch_size)), 
                                             jit(Geodesic), 
                                             z_obs, 
                                             M)
    elif args.method == "SGD":
        FrechetMean = RJAXAdaOptimization(M = M,
                                          init_fun=None,
                                          lr_rate=args.jax_lr_rate,
                                          optimizer=optimizers.sgd,
                                          T=args.T,
                                          max_iter=args.max_iter,
                                          tol=args.tol,
                                          )
        methods['AdaSGD'] = estimate_method(jit(lambda z: FrechetMean(z,batch_size)), 
                                            jit(Geodesic), 
                                            z_obs, 
                                            M,
                                            )
    else:
        raise ValueError(f"Method, {args.method}, not defined!")
    save_times(methods, save_path)
    
    print(methods)
    print(jnp.mean(z_obs, axis=0))
    
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
    
    z_obs, RM = load_manifold(args.manifold, 
                             args.dim, 
                             N_data=args.N_data,
                             )
    
    M = FRiemannianNavigation(RM=RM,
                              force_fun=lambda z: force_fun(1.0, z, RM),
                              v0=args.v0,
                              )
    
    Geodesic = FGEORCE(M=M,
                       init_fun=None,
                       T=args.T,
                       max_iter=10,
                       line_search_method="soft",
                       line_search_params = {'rho':0.5},
                       )
    
    batch_size = int(args.batch_size*args.N_data)
    
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
    elif args.method == "GEORCE_AdaFM":
        FrechetMean = FGEORCE_AdaFM(M=M,
                                 init_fun=None,
                                 alpha=0.9,
                                 sub_iters=2,
                                 T=args.T,
                                 max_iter=args.max_iter,
                                 tol = args.tol,
                                 line_search_params={'rho': 0.95},
                                 )    
        methods['GEORCE_AdaFM'] = estimate_method(jit(lambda z: FrechetMean(z,batch_size)), 
                                                  jit(Geodesic), 
                                                  z_obs, 
                                                  M,
                                                  )
    elif args.method == "AdaADAM":
        FrechetMean = FJAXAdaOptimization(M = M,
                                          init_fun=None,
                                          lr_rate=args.jax_lr_rate,
                                          optimizer=optimizers.adam,
                                          T=args.T,
                                          max_iter=args.max_iter,
                                          tol=args.tol,
                                          )
        methods['AdaADAM'] = estimate_method(jit(lambda z: FrechetMean(z,batch_size)), 
                                             jit(Geodesic), 
                                             z_obs, 
                                             M,
                                             )
    elif args.method == "SGD":
        FrechetMean = FJAXAdaOptimization(M = M,
                                          init_fun=None,
                                          lr_rate=args.jax_lr_rate,
                                          optimizer=optimizers.sgd,
                                          T=args.T,
                                          max_iter=args.max_iter,
                                          tol=args.tol,
                                          )
        methods['AdaSGD'] = estimate_method(jit(lambda z: FrechetMean(z,batch_size)), 
                                         jit(Geodesic), 
                                         z_obs, 
                                         M,
                                         )
    else:
        raise ValueError(f"Method, {args.method}, not defined!")
    save_times(methods, save_path)
    
    print(methods)
    print(jnp.mean(z_obs, axis=0))
        
    return

#%% Lorentz Run Time code

def lorentz_runtime()->None:

    args = parse_args()
    
    save_path = ''.join((args.save_path, f'lorentz/{args.manifold}/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_path = ''.join((save_path, args.method, 
                         f'_{args.manifold}', 
                         f'_d={args.dim}', 
                         f'_T={args.T}.pkl',
                         ))
    if os.path.exists(save_path):
        os.remove(save_path)
    
    z_obs, RM = load_manifold(args.manifold, 
                             args.dim, 
                             N_data=args.N_data,
                             )
    
    M = LRiemannianNavigation(RM=RM,
                              force_fun=lambda t,z: force_fun(t, z, RM),
                              v0=args.v0,
                              )
    
    Geodesic = LGEORCE(M=M,
                       init_fun=None,
                       T=args.T,
                       max_iter=10,
                       line_search_params = {'rho':0.5},
                       )
    
    batch_size = int(args.batch_size*args.N_data)
    
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
        FrechetMean = LGEORCE_AdaFM(M=M,
                                    init_fun=None,
                                    alpha=0.9,
                                    sub_iters=2,
                                    T=args.T,
                                    max_iter=args.max_iter,
                                    tol = args.tol,
                                    line_search_params={'rho': 0.95},
                                    )    
        methods['GEORCE_FM'] = estimate_lorentz(jit(lambda t,z: FrechetMean(t,z,batch_size)), 
                                                jit(Geodesic), 
                                                z_obs, 
                                                M,
                                                )
    elif args.method == "ADAM":
        FrechetMean = LJAXAdaOptimization(M = M,
                                          init_fun=None,
                                          lr_rate=args.jax_lr_rate,
                                          optimizer=optimizers.adam,
                                          T=args.T,
                                          max_iter=args.max_iter,
                                          tol=args.tol,
                                          )
        methods['AdaADAM'] = estimate_lorentz(jit(lambda t,z: FrechetMean(t,z,batch_size)), 
                                           jit(Geodesic), 
                                           z_obs,
                                           M,
                                           )
    elif args.method == "SGD":
        FrechetMean = LJAXAdaOptimization(M = M,
                                          init_fun=None,
                                          lr_rate=args.jax_lr_rate,
                                          optimizer=optimizers.sgd,
                                          T=args.T,
                                          max_iter=args.max_iter,
                                          tol=args.tol,
                                          )
        methods['AdaSGD'] = estimate_lorentz(jit(lambda t,z: FrechetMean(t,z,batch_size)), 
                                             jit(Geodesic), 
                                             z_obs, 
                                             M,
                                             )
    else:
        raise ValueError(f"Method, {args.method}, not defined!")
    save_times(methods, save_path)
    
    print(methods)
    print(jnp.mean(z_obs, axis=0))
        
    return
    
#%% main

if __name__ == '__main__':
    
    args = parse_args()
    
    if args.geometry == "Riemannian":
        riemannian_runtime()
    elif args.geometry == "Finsler":
        finsler_runtime()
    elif args.geometry == "Lorentz":
        lorentz_runtime()
    else:
        raise ValueError("Invalid geometry for runtime comparison.")