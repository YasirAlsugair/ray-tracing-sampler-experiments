"""Pod driver: MAP-with-prior transit states as chain starts (Josh's idea).

1) Re-run the prior-draw MAP optimization, saving the state at epochs
   5/10/15/20/30/50 (the transit from shell-with-bad-fit toward the mode).
2) Launch a 2,000-trajectory exact chain from each candidate start at the
   tuned MLP settings (dt 3.5e-4, L=30).
3) Print a summary: start position, acceptance, and where each chain ended,
   using misfit = -ln_post - norm^2/2 (no forward passes needed).
"""
import sys, time
sys.path.insert(0, "/workspace/ray-tracing-sampler-experiments/experiments")
import numpy as np
import torch
import torch.nn.functional as F
import exp6_prior_start as ps
import exp6_sample_metropolis as sm

EPOCH_SAVES = {5, 10, 15, 20, 30, 50}
CHAIN_FROM = [10, 15, 20, 50]

transform = ps.transforms.Compose([ps.transforms.ToTensor(),
                                   ps.transforms.Normalize((0.1307,), (0.3081,))])
train = ps.datasets.MNIST(ps.ROOT / "data", train=True, download=True,
                          transform=transform)
images = torch.stack([x for x, _ in train]).to(ps.DEV)
labels = torch.tensor(train.targets).to(ps.DEV)

torch.manual_seed(2)
model = ps.MLP().to(ps.DEV)
with torch.no_grad():
    for p in model.parameters():
        p.copy_(torch.randn_like(p))

states = {}
optimizer = torch.optim.Adam(model.parameters(), lr=ps.LR)
generator = torch.Generator().manual_seed(0)
for epoch in range(1, 51):
    order = torch.randperm(ps.N, generator=generator)
    for i in range(0, ps.N, ps.BATCH):
        idx = order[i:i + ps.BATCH].to(ps.DEV)
        optimizer.zero_grad()
        prior = sum((p ** 2).sum() for p in model.parameters()) / (2 * ps.N)
        (F.cross_entropy(model(images[idx]), labels[idx]) + prior).backward()
        optimizer.step()
    if epoch in EPOCH_SAVES:
        states[epoch] = ps.flat(model).cpu().numpy()
        print(f"[map] epoch {epoch}: misfit {ps.misfit(model, images, labels):>10,.0f}  "
              f"norm^2 {ps.norm2(model):>10,.0f}", flush=True)

np.savez(ps.TAB / "exp6_map_transit_states.npz",
         **{f"epoch{k}": v for k, v in states.items()})

for k in CHAIN_FROM:
    theta0 = torch.tensor(states[k], dtype=torch.float32, device=sm.DEV)
    print(f"\n[chain from epoch {k}] launching 2,000 trajectories "
          f"(dt 3.5e-4, L=30)", flush=True)
    t0 = time.time()
    sm.run("mlp", 3.5e-4, 30, 2000, theta0=theta0, tag=f"mapstart_ep{k}")
    print(f"[chain from epoch {k}] done in {(time.time() - t0) / 60:.1f} min",
          flush=True)

print("\nsummary (misfit = -ln_post - norm^2/2):")
print(f"{'start':>12} {'misfit_0':>10} {'norm2_0':>9} {'misfit_end':>11} {'norm2_end':>10}")
for k in CHAIN_FROM:
    d = np.load(ps.TAB / f"exp6_rt_chain_mlp_mapstart_ep{k}.npz")
    s, lnp = d["samples"], d["ln_post"]
    n0, n1 = float((s[0] ** 2).sum()), float((s[-1] ** 2).sum())
    m0, m1 = -lnp[0] - n0 / 2, -lnp[-1] - n1 / 2
    print(f"{'epoch ' + str(k):>12} {m0:>10,.0f} {n0:>9,.0f} {m1:>11,.0f} {n1:>10,.0f}")
print("\nreference: typical set misfit ~350, norm^2 ~49,600; "
      "Adam start reached misfit 158, norm^2 45,134 after 20,000 trajectories")
print("MAPSTART-DRIVER-COMPLETE")
