    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J celeba_32
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
    
    python3 train_vae.py \
        --model celeba \
        --svhn_dir /work3/fmry/Data/SVHN/ \
        --celeba_dir /work3/fmry/Data/CelebA/ \
        --lr_rate 0.0002 \
        --con_training 0 \
        --split 0.8 \
        --batch_size 100 \
        --latent_dim 32 \
        --epochs 500000 \
        --save_step 10000 \
        --save_path models/ \
        --seed 2712
    