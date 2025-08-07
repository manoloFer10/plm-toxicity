echo "🔥 Installing Miniconda…"
MINI="Miniconda3-latest-Linux-x86_64.sh"
wget --quiet https://repo.anaconda.com/miniconda/$MINI
chmod +x $MINI
./$MINI -b -p "$HOME/miniconda3"
rm $MINI

# Initialize conda in non-interactive shells
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda init bash
echo "✅ Conda installed."

echo "🔄 Updating conda…"
conda update -n base -y conda

echo "Setting up environment..."

conda env create -f environment.yml

echo "Done."