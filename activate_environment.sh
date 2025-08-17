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
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set -u

# accept ToS (ok with -u off or on now)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda update -n base -c defaults -y conda

# create & activate env
cd plm-toxicity
conda env create -y -f environment.yml
set +u
conda activate plmTox
set -u

echo "Environment ready, dowloading final pkgs..."

echo "Tmux..."
apt update && apt install -y tmux

echo "Pfam..."

# prepare Pfam DB once
mkdir -p ~/db/pfam && cd ~/db/pfam
wget -q https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
wget -q https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz
gunzip -f Pfam-A.hmm.gz Pfam-A.hmm.dat.gz
hmmpress Pfam-A.hmm

#  make it discoverable 
echo 'export PFAM_DB_DIR="$HOME/db/pfam"' >> ~/.bashrc

echo "LocalColabFold..."

cd $HOME
git clone https://github.com/YoshitakaMo/localcolabfold
cd localcolabfold
bash install_colabbatch_linux.sh     # creates ./colabfold-conda with colabfold_batch

conda activate /root/localcolabfold/localcolabfold/colabfold-conda
# Use conda's libstdc++.so.6
echo 'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"'
echo 'export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"'

# tell the  scorer to use this binary
export TOXDL2_COLABFOLD_BIN="$HOME/localcolabfold/localcolabfold/colabfold-conda/bin/colabfold_batch"

conda deactivate
cd $HOME/plm-toxicity #finish on main dir




