#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -J fetch_atomic
#SBATCH -t 08:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o logs/fetch_atomic_%j.out
#SBATCH -e logs/fetch_atomic_%j.err


module load python/3.13.5
export PYTHONUSERBASE="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH"

set -euo pipefail
mkdir -p logs

# Use your local installs only
export PYTHONUSERBASE="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"

# Disable pip network checks completely
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export PIP_EXISTS_ACTION=i
export PIP_DEFAULT_TIMEOUT=1

# Explorer proxy just for ssgetpy (data downloads, not pip)
export http_proxy="http://10.99.0.130:3128"
export https_proxy="http://10.99.0.130:3128"

cd "$HOME/projects/sparse-vs-dense-matmul"
python -c "import sys, ssgetpy; print('PY:', sys.version, sys.executable); print('PATH:', ssgetpy.__file__)"
python scripts/fetch_paper_matrices.py
