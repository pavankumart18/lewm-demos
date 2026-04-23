#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# LeWM Experiments — Reproducible End-to-End Script
# ═══════════════════════════════════════════════════════════════
# Prerequisites: Ubuntu 24.04, NVIDIA GPU, ~60GB disk space
# Usage: bash run_all.sh
# ═══════════════════════════════════════════════════════════════

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CACHE_DIR="$HOME/.stable_worldmodel"
OUTPUT_DIR="$HOME/lewm_experiments"

echo "═══════════════════════════════════════════════════════════"
echo "  LeWM: World Model Experiments"
echo "  Running from: $SCRIPT_DIR"
echo "═══════════════════════════════════════════════════════════"

# ── Step 0: Environment Setup ──
echo -e "\n[0/7] Checking environment..."
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Setting up Python environment..."
    sudo apt install -y python3.10 python3.10-venv libgl1-mesa-dri libglx-mesa0 libosmesa6-dev patchelf zstd
    pip install uv 2>/dev/null || true
    cd "$SCRIPT_DIR"
    uv venv --python=3.10
    source .venv/bin/activate
    uv pip install stable-worldmodel[train,env]
    uv pip install "datasets==2.21.0" --force-reinstall
else
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

export MUJOCO_GL=osmesa
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

# ── Step 1: Download Push-T data ──
echo -e "\n[1/7] Downloading Push-T dataset + checkpoint..."
if [ ! -f "$CACHE_DIR/pusht_expert_train.h5" ]; then
    hf download --repo-type dataset quentinll/lewm-pusht --local-dir /tmp/lewm-pusht
    zstd -d /tmp/lewm-pusht/pusht_expert_train.h5.zst -o "$CACHE_DIR/pusht_expert_train.h5"
fi

if [ ! -d "$CACHE_DIR/pusht" ]; then
    hf download --repo-type model quentinll/lewm-pusht --local-dir "$CACHE_DIR/pusht"
fi

# ── Step 2: Download Two-Room data ──
echo -e "\n[2/7] Downloading Two-Room dataset + checkpoint..."
if [ ! -f "$CACHE_DIR/tworoom.h5" ]; then
    hf download --repo-type dataset quentinll/lewm-tworooms --local-dir /tmp/lewm-tworooms
    cd /tmp/lewm-tworooms && tar --zstd -xvf tworoom.tar.zst
    mv /tmp/lewm-tworooms/tworoom.h5 "$CACHE_DIR/"
fi

if [ ! -d "$CACHE_DIR/tworooms" ]; then
    hf download --repo-type model quentinll/lewm-tworooms --local-dir "$CACHE_DIR/tworooms"
fi

# ── Step 3: Build object checkpoints ──
echo -e "\n[3/7] Building model checkpoints..."
cd "$SCRIPT_DIR"

for ENV_NAME in pusht tworooms; do
    if [ ! -f "$CACHE_DIR/$ENV_NAME/lewm_object.ckpt" ]; then
        echo "  Building $ENV_NAME checkpoint..."
        python -c "
import sys; sys.path.insert(0, '.')
import torch, json, torch.nn as nn
from jepa import JEPA
from module import ARPredictor as Predictor, Embedder, MLP
from hydra.utils import instantiate
from omegaconf import OmegaConf

with open('$CACHE_DIR/$ENV_NAME/config.json') as f:
    cfg = json.load(f)
encoder = instantiate(OmegaConf.create(cfg['encoder']))
p = cfg['predictor']
predictor = Predictor(num_frames=p['num_frames'], input_dim=p['input_dim'],
    hidden_dim=p['hidden_dim'], output_dim=p['output_dim'],
    depth=p['depth'], heads=p['heads'], mlp_dim=p['mlp_dim'],
    dim_head=p['dim_head'], dropout=p['dropout'], emb_dropout=p['emb_dropout'])
action_encoder = Embedder(input_dim=cfg['action_encoder']['input_dim'], emb_dim=cfg['action_encoder']['emb_dim'])
proj = MLP(input_dim=cfg['projector']['input_dim'], output_dim=cfg['projector']['output_dim'], hidden_dim=cfg['projector']['hidden_dim'], norm_fn=nn.BatchNorm1d)
pred_proj = MLP(input_dim=cfg['pred_proj']['input_dim'], output_dim=cfg['pred_proj']['output_dim'], hidden_dim=cfg['pred_proj']['hidden_dim'], norm_fn=nn.BatchNorm1d)
model = JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder, projector=proj, pred_proj=pred_proj)
model.load_state_dict(torch.load('$CACHE_DIR/$ENV_NAME/weights.pt', map_location='cpu'))
model.eval()
torch.save(model, '$CACHE_DIR/$ENV_NAME/lewm_object.ckpt')
print('  ✓ $ENV_NAME checkpoint built')
"
    else
        echo "  ✓ $ENV_NAME checkpoint exists"
    fi
done

# ── Step 4: Run Experiment 1 ──
echo -e "\n[4/7] Running Experiment 1: Warehouse Routing..."
cd "$SCRIPT_DIR"
python exp1_routing.py

# ── Step 5: Run Experiment 2 ──
echo -e "\n[5/7] Running Experiment 2: Bin Packing..."
python exp2_packing.py

# ── Step 6: Run Experiment 3 ──
echo -e "\n[6/7] Running Experiment 3: Self-Driving..."
python exp3_driving.py

# ── Step 7: Summary ──
echo -e "\n[7/7] Collecting results..."
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Output files:"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  (no plots found)"
echo ""
echo "  Videos:"
ls "$CACHE_DIR"/pusht/rollout_0.mp4 2>/dev/null && echo "  ✓ Push-T rollout videos in $CACHE_DIR/pusht/"
ls "$CACHE_DIR"/tworooms/rollout_0.mp4 2>/dev/null && echo "  ✓ Two-Room rollout videos in $CACHE_DIR/tworooms/"
echo ""
echo "  To view results: xdg-open $OUTPUT_DIR/exp1_routing.png"
echo "═══════════════════════════════════════════════════════════"
