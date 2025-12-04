#!/bin/bash
#SBATCH --job-name=probing_experiment
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load miniconda3/24.1.2-py310 cuda/12.4.1
conda init bash
conda activate nanogpt_cu124  # TODO change this to your personal environment

echo "Python: $(which python) ($(python --version))"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"

model_type="normal"
config_file="config/default.json"
for arg in "$@"; do
  case $arg in
    --model-type=*) model_type="${arg#*=}";;
    --config=*) config_file="${arg#*=}";;
    *)
      echo "unknown argument: $arg"
      exit 1
      ;;
  esac
done

repo_name="ipa-gpt-interpret"
repo_address="git@github.com:aaron-jencks/$repo_name.git"
repo_dir="$scratch_github_prefix/$repo_name"
cd "$repo_dir"

echo "Working directory: $(pwd)"
echo "Model type: $model_type"
echo "Configs: $config_file"

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

# Run the probing experiment
python probing-exp-preextracted.py \
  $config_file \
  --model-type "$model_type" \
  --num-layers 12 \
  --hidden-dim 768 \
  --output-log "probing_results_preextracted_$model_type"

echo "===== [$(date)] JOB COMPLETED ====="