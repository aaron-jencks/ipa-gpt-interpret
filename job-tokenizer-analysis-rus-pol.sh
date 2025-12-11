#!/bin/bash
#SBATCH --job-name=tokenizer_overlap_analysis
#SBATCH --account=PAS3150
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda3/24.1.2-py310 cuda/12.4.1
conda init bash # Initialize conda
conda_env="nanogpt_cu124"
conda activate "$conda_env" # Activate the conda environment
echo "conda environment: $conda_env"

free --giga
export PYTHONPATH=. # Set the python path

# setup paths

storage_prefix="/fs/ess/PAS2836/ipa_gpt"
tokenizers_prefix="$storage_prefix/tokenizers"
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"                # temporary storage for job use
scratch_github_prefix="$scratch_prefix/github"              # temporary location of the github repo

model_type="normal"
for arg in "$@"; do
  case $arg in
    --model-type=*) model_type="${arg#*=}";;
    *)
      echo "unknown argument: $arg"
      exit 1
      ;;
  esac
done

# modifiable parameters

dataset="iggy12345/pair_russian_polish_ipa"
tokenizer_name="bpe-rus-pol-normal-number-preservation-08-20-2025"
feature="text"
if [[ "$model_type" == "ipa" ]]; then
  tokenizer_name="bpe-rus-ipa-normal-number-preservation-08-20-2025"
  feature="phonemes"
fi

# github repo

repo_name="ipa-gpt"                                     # the github repo name
repo_prefix="$scratch_github_prefix/$repo_name"         # the location of the downloaded repo
cd "$repo_prefix"

# Run code here
# transcription
echo "analyzing dataset..."
python bpe-analysis.py \
  "$tokenizers_prefix" "$tokenizer_name" \
  --dataset "$dataset" --feature "$feature" \
  --cache "$scratch_prefix/cache" \
  --cpus 96
