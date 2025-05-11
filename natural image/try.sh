#!/bin/bash -x
#SBATCH -p czhangcn_rent   # 替换为你的GPU分区名称
#SBATCH -J test
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1     # 每个节点启动1个任务
#SBATCH --gres=gpu:0     # 每个节点分配8块GPU
#SBATCH --cpus-per-task=16 # 每个任务使用32个CPU核心（适配每个GPU 4个线程）
#SBATCH -o swinunet.out     # 标准输出日志
#SBATCH -e swinunet.err      # 错误输出日志

# 设置路径
export PATH=/hpc2hdd/home/xingmu/miniconda3/bin:$PATH  # 替换为你实际的安装路径

# 初始化conda
eval "$(/hpc2hdd/home/xingmu/miniconda3/condabin/conda shell.bash hook)"
module load cuda/11.7
# 激活你的conda环境
conda activate UNI_breast  # 替换为你实际的环境

# 环境变量设置（确保分布式通信正常）




# 运行训练脚本
python /hpc2hdd/home/xingmu/IJCNN/test_metrics.py
