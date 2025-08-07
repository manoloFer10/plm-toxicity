set -euo pipefail                       # fail fast & catch unset vars

echo "🔥  Installing Miniconda …"

# ── 1. pick a valid installer (use 'latest' unless you need a frozen build)
MINI=Miniconda3-latest-Linux-x86_64.sh        # ← exists in the archive

wget -q https://repo.anaconda.com/miniconda/$MINI
chmod +x $MINI
./$MINI -b -p "$HOME/miniconda3"
rm $MINI
echo "✅  Conda installed."

# ── 2. make conda available in this shell
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ── 3. auto-accept Anaconda Terms of Service **once**
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# (or export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes to blanket-approve in CI
#  per Anaconda’s blog post):contentReference[oaicite:4]{index=4}

# ── 4. create the project env (run from repo root!)
conda env create -y -f environment.yml        # -y → no prompt
conda activate plm-toxicity
pip install requirements.txt

echo "🎉  Environment ready."