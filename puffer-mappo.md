# PufferLib MAPPO/LCPPO Implementation Plan

This plan outlines how to train and fine‑tune MAPPO (and an LCPPO variant) on this Flatland environment using PufferLib, starting from the hyperparameter defaults used in Zhang et al. (LCPPO codebase `../lcppo`). It includes training/finetuning steps, CARBS tuning hooks, and device handling (CUDA/MPS/CPU).

## 1) Goals
- **Baseline MAPPO**: centralized critic, decentralized (shared) actor.
- **LCPPO variant**: masked local‑critic (Transformer attention) to emulate the Zhang et al. local‑critic approach.
- **Reuse TreeLSTM/TreeObs** from this repo (TreeCutils + TreeLSTM).
- **PufferLib** for data collection + batching.
- **CARBS** for hyperparameter search.

## 2) Device handling (CUDA / MPS / CPU)
Always pick the best available device but **fall back to CPU** when MPS/CUDA is not supported.

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

- **Env stays on CPU** (Flatland is CPU‑bound).
- **Model and training tensors** move to `device`.
- **Checkpoint loading** always uses `map_location=device`.

## 3) Environment wrapper (PufferLib)
Use the same TreeCutils features but output agent dictionaries compatible with MAPPO/LCPPO:

```python
obs = {
  agent_id: {
    "obs": obs_vec,                # flattened agent feature vector
    "action_mask": valid_actions,  # 0/1 action mask
    "agent_mask": agent_mask,      # mask of non‑neighbors (LCPPO)
    "last_agent_mask": prev_mask,  # stabilization (LCPPO)
  }
}
```

- **agent_mask** mirrors the neighbor logic in `../lcppo/marllib/envs/flatland_env/flatland_env.py`.
- **shared_reward**: start with global reward (same as Zhang et al.).

## 4) Model architecture
### 4.1 Shared encoder (TreeLSTM)
- Reuse `solution/nn/net_tree_torchrl.py` or the TreeLSTM encoder in `solution/nn/TreeLSTM.py`.
- Encode per‑agent observation to embedding.

```python
z_i = tree_lstm_encoder(obs_i)  # per agent embedding
```

### 4.2 Actor (shared)
- Single shared policy network across agents.
- Categorical action head with action masking.

```python
logits = actor(z_i)
logits = logits + log(action_mask + 1e-9)  # mask invalid actions
action = Categorical(logits=logits).sample()
```

### 4.3 Critic (MAPPO baseline)
- Centralized critic: takes **all agent embeddings** (concat or pooled) and outputs per‑agent values.

```python
z_all = concat(z_1, z_2, ... z_N)
values = critic(z_all)  # per‑agent or scalar
```

### 4.4 Critic (LCPPO variant)
- Transformer attention with **agent_mask** to restrict attention to neighbors.
- Output per‑agent values.

```python
values = masked_transformer(z_all, mask=agent_mask)
```

## 5) MAPPO training loop (PufferLib)
Use PPO‑style update with GAE and shared policy. Start from Zhang et al. hyperparams.

**Starting hyperparams (from `../lcppo`):**
- `lr: 5e-5`
- `clip_param: 0.1`
- `num_sgd_iter: 10`
- `vf_loss_coeff: 1.0`
- `entropy_coeff: 0.0`
- `lambda: 1.0`
- `target_kl: 0.01`
- `vf_clip_param: 10.0`

**Pseudocode:**
```python
collector = pufferlib.collector(envs, policy, rollout_len)

for batch in collector:
    # compute advantages via GAE
    advantages, returns = compute_gae(batch, gamma, lam)

    for _ in range(num_sgd_iter):
        loss_pi, loss_v = mappo_loss(batch, advantages, returns, clip)
        loss = loss_pi + vf_coef * loss_v - ent_coef * entropy
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

## 6) Fine‑tuning plan (Jiang‑style curriculum)
Mimic the multi‑phase curriculum from Jiang (Phase I/II/III):

1. **Phase I (50 agents)**
   - Emphasize environmental reward, deadlock penalty.
2. **Phase II (50 agents)**
   - Increase departure reward to encourage starts.
3. **Phase III (80/100/200 agents)**
   - Initialize from Phase II; fine‑tune on larger maps.

This aligns with the legacy `.pt` policies (`phase-III-50/80/100/200.pt`).

## 7) CARBS hyperparameter tuning
Wrap training as a CARBS trial, search around Zhang’s defaults.

**Search space (example):**
- `lr: [1e-5, 3e-4]`
- `clip_param: [0.05, 0.2]`
- `entropy_coeff: [0.0, 0.02]`
- `vf_loss_coeff: [0.5, 2.0]`
- `gae_lambda: [0.9, 1.0]`
- `rollout_len: [256, 2048]`

**CARBS hook skeleton:**
```python
def carbs_objective(cfg):
    metrics = train_mappo(cfg)
    return metrics["arrival_ratio"]  # or normalized reward
```

## 8) Evaluation plan
Evaluate against:
- **TorchRL PPO** checkpoints in `trained_model_checkpoints/`
- **Legacy Jiang `.pt` policies** in `solution/policy/`
- **MAPPO (PufferLib)** baseline
- **LCPPO (PufferLib)** variant

Metrics:
- arrival ratio
- deadlock ratio
- normalized reward
- steps to completion

Use both:
- `solution/debug-environments` for deterministic regression
- random seed set for generalization

## 9) Differences: MAPPO vs LCPPO (summary)
| Aspect | MAPPO | LCPPO |
|---|---|---|
| Critic input | All agents (global) | Masked local neighbors |
| Critic architecture | MLP / pooled | Transformer + mask |
| Scalability | Poor at large N | Improved via locality |
| Extra inputs | none | `agent_mask`, `last_agent_mask` |

## 10) Next code artifacts to add
- `puffer_mappo_train.py`
- `puffer_lcppo_train.py`
- `puffer_flatland_env.py` (PufferLib wrapper)
- `configs/mappo.yaml` (defaults from Zhang)
- `configs/lcppo.yaml` (same + masked critic settings)
