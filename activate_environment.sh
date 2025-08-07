#!/usr/bin/env bash

echo "🔥 Installing Miniconda…"
MINI="Miniconda3-py310_2025.06-0-Linux-x86_64.sh"   # pin if reproducibility matters
wget -q https://repo.anaconda.com/miniconda/"$MINI"
chmod +x "$MINI"
./"$MINI" -b -p "$HOME/miniconda3"
rm "$MINI"
echo "✅ Conda installed."

# Make conda available *in this shell*
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Accept Anaconda ToS once (non-interactive)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

cd "$HOME/plm-toxicity"

# Create environment in one shot
conda env create -y -f environment.yml

# Activate for current session
conda activate plm-toxicity
echo "🎉 Environment ready."