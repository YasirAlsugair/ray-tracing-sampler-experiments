#!/bin/bash
# One-paste setup for a RunPod A100 (PyTorch template).
# Clones the repo, checks CUDA, and launches the two CNN runs in the
# background with logs. Results land in repo/results/tables/*.npz.
set -e

cd /workspace
if [ ! -d ray-tracing-sampler-experiments ]; then
    git clone https://github.com/YasirAlsugair/ray-tracing-sampler-experiments.git
fi
cd ray-tracing-sampler-experiments
pip install --quiet numpy scipy torchvision

python - << 'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
print("device:", torch.cuda.get_device_name(0))
EOF

cd experiments

# Run 1: the exact CNN chain, the missing ground truth. Full-batch
# gradients with the Metropolis test on, 20,000 trajectories at the tuned
# settings (dt 1.5e-4, L = 30). Roughly overnight on an A100.
nohup python exp6_sample_metropolis.py run cnn 1.5e-4 30 20000 \
    > ../results/tables/exp6_rt_chain_cnn20k_runpod.log 2>&1 &
echo "launched: exact CNN chain (log: exp6_rt_chain_cnn20k_runpod.log)"

# Run 2: finish the CNN Eq. 33 chain's creep to a formal verdict.
# Resumes from the last committed leg (part 49) automatically.
nohup python - << 'EOF' > ../results/tables/exp6_cnn33_runpod.log 2>&1 &
import exp6_minibatch as mb
images, labels, n_train = mb.full_train_tensors()
mb.converge_gated(1e-4, 5, images, labels, n_train,
                  leg_steps=250_000, max_legs=40, arch="cnn")
EOF
echo "launched: CNN Eq. 33 continuation (log: exp6_cnn33_runpod.log)"

echo
echo "monitor with:"
echo "  tail -f /workspace/ray-tracing-sampler-experiments/results/tables/exp6_rt_chain_cnn20k_runpod.log"
echo "  tail -f /workspace/ray-tracing-sampler-experiments/results/tables/exp6_cnn33_runpod.log"
echo "when done, download results/tables/*.npz (runpodctl send, or scp)"
