#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH -o ./results/results/TGCN/TGCN_sparsesoft_result(0.08-0.1-0.2-0.3-0.4-0.5).out


source /home/liuyu/miniconda3/etc/profile.d/conda.sh
conda activate RTGCN

module purge
module load 2021
module load CUDA/11.3.1

for noise in 0.08 0.1 0.2 0.3 0.4 0.5
do
python main.py  --model_name TGCN --noise_type missing --max_epochs 5000 --attack --soft --sparse --alpha 3 --num_iter 1 --lamda 2 --learning_rate 0.0005 --weight_decay 5e-4 --batch_size 64 --hidden_dim 64 --loss mse_with_regularizer --pre_len 3 --settings supervised --gpus 1  --data losloop --noise_test  --noise_ratio_test $noise --noise --noise_ratio 0.04
done

