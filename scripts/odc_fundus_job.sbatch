#!/bin/bash
#SBATCH --job-name=odc_job         # job name
#SBATCH --account=ki-css           # account name
#SBATCH --output=logs/odc_%j.out   # output filei
#SBATCH --error=logs/odc_%j.err    # error file
#SBATCH --partition=a100ai         # partition
#SBATCH --nodes=1                  # number of nodes
#SBATCH --ntasks=1                 # numper of tasks
#SBATCH --cpus-per-task=8          # cpu kernel
#SBATCH --mem=64G                  # RAM
#SBATCH --time=24:00:00            # max. runningtime  (HH:MM:SS)
#SBATCH --gres=gpu:1               # gpus
#SBATCH --mail-type=ALL       # automated mail notification
#SBATCH --mail-user=mpaul01@students.uni-mainz.de

# --- Load Module  ---
module purge
module load lang/Python/3.12.3-GCCcore-13.3.0

# --- Data preperation ---
mkdir -p /localscratch/${SLURM_JOB_ID}/dataset/fundus
cp /lustre/project/ki-css/data-opthalmic-classification-thesis/big-fundus-dataset.zip /localscratch/${SLURM_JOB_ID}/
unzip /localscratch/${SLURM_JOB_ID}/big-fundus-dataset.zip -d /localscratch/${SLURM_JOB_ID}/dataset/fundus

echo "--- Control of the directory structure ---"
find /localscratch/${SLURM_JOB_ID}/dataset -type d
echo "-------------------------------------"
# --- Activate Venv ---
VENV_DIR="$HOME/bachelor-thesis/ophthalmic-diffusion-classifier/venv-odc"

source "$VENV_DIR/bin/activate"

# --- Open project & script folder  ---
PROJECT_DIR="$HOME/bachelor-thesis/ophthalmic-diffusion-classifier"
cd "$PROJECT_DIR"/scripts

# ---Start Job  ---
bash run.sh

# --- Deactivate venv ---
deactivate
