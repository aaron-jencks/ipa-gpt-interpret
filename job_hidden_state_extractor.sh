#!/bin/bash
#SBATCH --job-name=probing_dataset_hidden_state_extractor
#SBATCH --account=PAS3150
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=01:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus-per-node=1

echo "===== [$(date)] JOB STARTED ====="

# Load required modules
module load miniconda3/24.1.2-py310 cuda/12.4.1
conda init bash
conda activate nanogpt_cu124  # TODO change this to your personal environment

echo "Python: $(which python) ($(python --version))"

# setup paths
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
storage_prefix="/fs/ess/PAS2836/ipa_gpt"
checkpoints_prefix="$storage_prefix/checkpoints"
scratch_datasets_prefix="$scratch_prefix/tokens"
scratch_github_prefix="$scratch_prefix/github"
scratch_hf_cache_prefix="$scratch_prefix/cache"
mkdir -pv $scratch_datasets_prefix $scratch_github_prefix $checkpoints_prefix $scratch_hf_cache_prefix

repo_name="ipa-gpt-interpret"
repo_address="git@github.com:aaron-jencks/$repo_name.git"
repo_dir="$scratch_github_prefix/$repo_name"
cd "$repo_dir"
git pull

model_type="normal"
batch_size=512
accumulation_size=60000
for arg in "$@"; do
  case $arg in
    --model-type=*) model_type="${arg#*=}";;
    --batch-size=*) batch_size="${arg#*=}";;
    --accumulation-size=*) accumulation_size="${arg#*=}";;
    *)
      echo "unknown argument: $arg"
      exit 1
      ;;
  esac
done

echo "Model type: $model_type"
echo "Batch size: $batch_size"
echo "Accumulation size: $accumulation_size"

echo "===== [$(date)] RUNNING PYTHON SCRIPT ====="

# Run the actual script
python hidden_state_extractor.py \
  config/default.json \
  --cpus 16 --debug --model-type "$model_type" \
  --batch-size "$batch_size" --accumulation-size "$accumulation_size"

echo "===== [$(date)] JOB COMPLETED ====="