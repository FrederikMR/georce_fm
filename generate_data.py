#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.random as jrandom
import jax.numpy as jnp
from jax import jit, lax

import haiku as hk

import os

from geometry.riemannian.manifolds import nSphere, nEllipsoid, nEuclidean, \
    nParaboloid, HyperbolicParaboloid, SPDN, H2, Cylinder, Landmarks, T2, \
        LatentSpaceManifold, FisherRaoGeometry

from vae.model_loader import mnist_generator, svhn_generator, celeba_generator, load_model

from vae.models import mnist_encoder
from vae.models import mnist_decoder
from vae.models import mnist_vae

from vae.models import svhn_encoder
from vae.models import svhn_decoder
from vae.models import svhn_vae

from vae.models import celeba_encoder
from vae.models import celeba_decoder
from vae.models import celeba_vae

#%% Load manifolds

def generate_data(manifold:str="celeba", 
                  dim:int = 2,
                  N_data:int=100,
                  sigma:float=1.0,
                  seed:int=2712,
                  data_path:str = 'data/',
                  svhn_path:str = "../../../Data/SVHN/",
                  celeba_path:str = "../../../Data/CelebA/",
                  ):
    
    key = jrandom.PRNGKey(seed)
    
    save_path = ''.join((data_path, f'{manifold}/'))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if manifold == "Euclidean":
        M = nEuclidean(dim=dim)
        
        z0 = jnp.zeros(dim)
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))

    if manifold == "SPDN":
        M = SPDN(N=dim)
        
        x0 = jnp.eye(dim)*10.0
        z0 = M.invf(x0)

        z_obs = z0+jnp.abs(jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim)))

    elif manifold == "Paraboloid":
        M = nParaboloid(dim=dim)
        
        z0 = jnp.zeros(dim)+1.0
        z_obs = z0+jnp.sqrt(0.1)*jrandom.normal(key, shape=(N_data, M.dim))
        
    elif manifold == "HyperbolicParaboloid":
        M = HyperbolicParaboloid()
        
        z0 = jnp.zeros(dim)+1.0
        z_obs = z0+jnp.sqrt(0.1)*jrandom.normal(key, shape=(N_data, M.dim))

    elif manifold == "Sphere":
        M = nSphere(dim=dim)
        
        z0 = -0.5*jnp.linspace(0,1,dim)
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))

    elif manifold == "Ellipsoid":
        params = jnp.linspace(0.5,1.0,dim+1)
        M = nEllipsoid(dim=dim, params=params)
        
        z0 = -0.5*jnp.linspace(0,1,dim)
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))

    elif manifold == "H2":
        M = H2()
        
        z0 = jnp.zeros(2)#+jnp.pi#jnp.array([1.0,1.0])
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))

    elif manifold == "Cylinder":
        M = Cylinder()
        
        z0 = jnp.array([-5*jnp.pi/4,1.0])
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))

    elif manifold == "T2":
        M = T2(R=3.0, r=1.0)
        
        z0 = jnp.zeros(2)#+jnp.pi
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))

    elif manifold == "Landmarks":
        M = Landmarks(N=dim, m=2, k_alpha=0.1)
        
        z0 = jnp.vstack((jnp.linspace(-5.0,5.0,M.N),jnp.linspace(0.0,0.0,M.N))).T.flatten()
        z_obs = z0+jnp.sqrt(sigma)*jrandom.normal(key, shape=(N_data, M.dim))
        
    elif manifold == "Gaussian":
        M = FisherRaoGeometry(distribution='Gaussian')
        z0 = jnp.array([-1.0, 0.5])
        z1 = jnp.array([1.0, 1.0])
        
        eps = 0.1*jrandom.normal(key, shape=(N_data//2, dim))
        
        data1 = z0 + jnp.abs(eps)
        data2 = z1 + jnp.abs(eps)
        z_obs = jnp.vstack((data1, data2))
        
    elif manifold == "Frechet":
        M = FisherRaoGeometry(distribution='Frechet')
        
        z0 = jnp.array([0.5, 0.5])
        z1 = jnp.array([1.0, 1.0])
        
        eps = 0.1*jrandom.normal(key, shape=(N_data//2, dim))
        
        data1 = z0 + jnp.abs(eps)
        data2 = z1 + jnp.abs(eps)
        z_obs = jnp.vstack((data1, data2))

    elif manifold == "Cauchy":
        M = FisherRaoGeometry(distribution='Cauchy')
        z0 = jnp.array([-1.0, 0.5])
        z1 = jnp.array([1.0, 1.0])
        
        eps = 0.1*jrandom.normal(key, shape=(N_data//2, dim))
        
        data1 = z0 + jnp.abs(eps)
        data2 = z1 + jnp.abs(eps)
        z_obs = jnp.vstack((data1, data2))

    elif manifold == "Pareto":
        M = FisherRaoGeometry(distribution='Pareto')
        
        z0 = jnp.array([0.5, 0.5])
        z1 = jnp.array([1.0, 1.0])
        
        eps = 0.5*jrandom.normal(key, shape=(N_data//2, dim))
        
        data1 = z0 + jnp.abs(eps)
        data2 = z1 + jnp.abs(eps)
        z_obs = jnp.vstack((data1, data2))

    elif manifold == "celeba":
        
        celeba_state = load_model(''.join(('models/', f'celeba_{dim}/')))
        celeba_dataloader = celeba_generator(data_dir=celeba_path,
                                             batch_size=N_data,
                                             seed=seed,
                                             split=0.8,
                                             )
        @hk.transform
        def celeba_tvae(x):

            vae = celeba_vae(
                        encoder=celeba_encoder(latent_dim=dim),
                        decoder=celeba_decoder(),
            )
         
            return vae(x)
        
        celeba_vae_fun = jit(lambda x: celeba_tvae.apply(lax.stop_gradient(celeba_state.params),
                                                         celeba_state.rng_key,
                                                         x))
       
       
        celeba_data = next(celeba_dataloader).x
        celeba_rec = celeba_vae_fun(celeba_data)
        
        z_obs = celeba_rec.mu_zx
   
    elif manifold == "svhn":
        
        svhn_state = load_model(''.join(('models/', f'svhn_{dim}/')))
        svhn_dataloader = svhn_generator(data_dir=svhn_path,
                                         batch_size=N_data,
                                         seed=seed,
                                         split='train[:80%]',
                                         )
        @hk.transform
        def svhn_tvae(x):

            vae = svhn_vae(
                        encoder=svhn_encoder(latent_dim=dim),
                        decoder=svhn_decoder(),
            )
         
            return vae(x)

        svhn_vae_fun = jit(lambda x: svhn_tvae.apply(lax.stop_gradient(svhn_state.params),
                                                     svhn_state.rng_key,
                                                     x))
       
       
        svhn_data = next(svhn_dataloader).x
        svhn_rec = svhn_vae_fun(svhn_data)
        
        z_obs = svhn_rec.mu_zx
        
    elif manifold == "mnist":
        
        mnist_state = load_model(''.join(('models/', f'mnist_{dim}/')))
        mnist_dataloader = mnist_generator(seed=seed,
                                           batch_size=N_data,
                                           split='train[:80%]')
       
        @hk.transform
        def mnist_tvae(x):
       
            vae = mnist_vae(
                        encoder=mnist_encoder(latent_dim=dim),
                        decoder=mnist_decoder(),
            )
       
            return vae(x)
        mnist_vae_fun = lambda x: mnist_tvae.apply(mnist_state.params,
                                                   mnist_state.rng_key,
                                                   x)
       
       
        mnist_data = next(mnist_dataloader).x
        mnist_rec = mnist_vae_fun(mnist_data)
        
        z_obs = mnist_rec.mu_zx

    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
            
    jnp.save(''.join((save_path, f'z_obs_{dim}_{N_data}.npy')), z_obs)
        
    return

#%% Generate Data

def generate_all_data(sigma:float=1.0,
                      seed:int=2712,
                      data_path:str = 'data/',
                      svhn_path:str = "../../../Data/SVHN/",
                      celeba_path:str = "../../../Data/CelebA/",
                      )->None:

    N_data = [100, 1_000]
    #sphere
    runs = {"Sphere": [2,3,5,10,20,50,100,250,500,1000],
            "Ellipsoid": [2,3,5,10,20,50,100,250,500,1000],
            "Paraboloid": [2],
            "HyperbolicParaboloid": [2],
            "SPDN": [2,3],
            "T2": [2],
            "H2": [2],
            "Gaussian": [2],
            "Frechet": [2],
            "Cauchy": [2],
            "Pareto": [2],
            "mnist": [8],
            "celeba": [32],
            "svhn": [32],
            }
    
    for m,dims in runs.items():
        for dim in dims:
            print(f"Generating data for {m} with dimension {dim}")
            for N in N_data:
                generate_data(manifold=m,
                              dim=dim,
                              N_data=N,
                              sigma=sigma,
                              seed=seed,
                              data_path=data_path,
                              svhn_path=svhn_path,
                              celeba_path=celeba_path,
                              )
    
    return