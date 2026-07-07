#!/bin/bash

#SBATCH --job-name=char-ngram-valla   # A descriptive name for your job
#SBATCH --output=char-ngram_%j.out    # File to capture standard output
#SBATCH --error=char-ngram_%j.err     # File to capture standard error
#SBATCH --nodes=1                     # Request a single node
#SBATCH --ntasks-per-node=1             # Run one task on the node
#SBATCH --cpus-per-task=10              # Number of CPUs per task (should match --num_workers)
#SBATCH --mem=64G                       # Memory per node
#SBATCH --time=04:00:00                 # Maximum wall-clock time (hh:mm:ss)
#SBATCH --partition=h100                # <--- CHANGE THIS to your cluster's partition name

# --- Environment Setup ---
echo "Setting up the environment..."
# Load any necessary modules (e.g., conda). This command might be different on your cluster.
# module load anaconda3 

# Activate your conda environment
source activate valla # <--- CHANGE THIS if your environment has a different name
echo "Conda environment activated."

# --- Job Execution ---
echo "Starting the Python script..."
# Run the CharNGram method.
# The example below runs the full ensemble for Authorship Attribution (AA).
# Add or remove flags as needed for your specific experiment.
python3 -m valla.methods.CharNGram \
    --project "pan20-charngram-aa" \
    --train_dataset "gerav_AV_train.csv" \
    --test_dataset "gerav_AV_test.csv" \
    --max_features 1000 \
    --min_df 0.01 \
    --sublinear_tf \
    --num_workers 10
    --av
# To run in Authorship Verification (AV) mode on a specific n-gram type, you could use:
# python3 -m valla.methods.CharNGram \
#     --project "pan20-charngram-av" \
#     --train_dataset "/path/to/your/pan20-authorship-verification-training-small.jsonl" \
#     --test_dataset "/path/to/your/pan20-authorship-verification-test.jsonl" \
#     --max_features 100000 \
#     --min_df 0.01 \
#     --sublinear_tf \
#     --num_workers 10 \
#     --av \
#     --type "char"

echo "Script finished."
