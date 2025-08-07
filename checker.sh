#!/usr/bin/env bash
#
# Continuously look for the two activation-tensor files produced by
#   torch.save(acts_tox, "preliminary_exps/acts/tox_acts.pt")
#   torch.save(acts_non_tox, "preliminary_exps/acts/non_tox_acts.pt")
# When both appear, add & push them (with Git LFS by default) and,
# if everything is clean, remove the pod.

set -euo pipefail                            # fail fast, unset vars → error

ACT_DIR="preliminary_exps/acts"
TOX_F="$ACT_DIR/tox_acts.pt"
NONTOX_F="$ACT_DIR/non_tox_acts.pt"

# Make sure Git LFS is ready (only needs to run once per repo)
if ! git lfs env &>/dev/null; then
    echo "❌  Git LFS not installed. Install it first:"
    echo "    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash"
    echo "    sudo apt-get install git-lfs && git lfs install"
    exit 1
fi

# Track *.pt files with LFS (safe to repeat)
git lfs track "*.pt" >/dev/null 2>&1 || true
git add .gitattributes || true

while true; do
    echo "🔍  Checking for activation tensors …"

    if [[ -f "$TOX_F" && -f "$NONTOX_F" ]]; then
        echo "✅  Found both tensors. Committing & pushing …"

        git add "$TOX_F" "$NONTOX_F"
        git commit -m "Add activation tensors (tox & non-tox)"
        git pull --rebase           # handle concurrent pushes politely
        git push

        echo "🔄  Verifying clean working tree …"
        if [[ -z "$(git status --porcelain)" ]] && [[ -z "$(git rev-list @{u}..HEAD)" ]]; then
            echo "✨  Push succeeded and repo is clean."

            # OPTIONAL: tear down your Runpod instance when done
            if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
                echo "⚠️  Removing pod $RUNPOD_POD_ID in 60 s …"
                sleep 60
                runpodctl remove pod "$RUNPOD_POD_ID"
            fi
            break
        else
            echo "⚠️  Something went wrong — not removing the pod."
            break
        fi
    else
        echo "⏳  Tensors not found yet. Sleeping 5 min …"
        sleep 300
    fi
done