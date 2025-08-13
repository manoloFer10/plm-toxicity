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
if ! grep -q 'miniconda3/etc/profile.d/conda.sh' ~/.bashrc; then
  echo '. "$HOME/miniconda3/etc/profile.d/conda.sh"' >> ~/.bashrc
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ── 3. auto-accept Anaconda Terms of Service **once**
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# (or export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes to blanket-approve in CI
#  per Anaconda’s blog post):contentReference[oaicite:4]{index=4}

# ── 4. create the project env (run from repo root!)
cd plm-toxicity
conda env create -y -f environment.yml        # -y → no prompt
conda activate plmTox
pip install -r requirements.txt

echo "🎉  Environment ready, dowloading final pkgs..."

echo "Tmux..."
apt update && apt install -y tmux

echo "Pfam + HMMER..."

conda install -c conda-forge -c bioconda pfam_scan colabfold
mkdir -p ~/db/pfam && cd ~/db/pfam
# Download current Pfam release (hosted by InterPro)
wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz

# Unpack
gunzip Pfam-A.hmm.gz
gunzip Pfam-A.hmm.dat.gz

# Create HMMER indices (required for hmmscan)
hmmpress Pfam-A.hmm
# Produces: Pfam-A.hmm.h3{f,i,m,p}