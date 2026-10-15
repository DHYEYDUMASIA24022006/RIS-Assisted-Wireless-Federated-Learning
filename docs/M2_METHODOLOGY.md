# M2 Methodology: Learning-Aware Joint Client Selection and RIS Design

## Status and scope

This document specifies the revised M2 method implemented in this branch. It is
a simulation methodology, not an RF deployment or a claim of globally optimal
joint learning and communication design. Numerical claims belong in a results
section only after the configured experiments have been rerun and their raw
outputs retained.

M2 asks: **in each synchronous federated-learning round, which clients should
participate and how should a shared RIS be configured, given a participant cap,
a round-latency budget, and per-client energy budgets?** The revised scheduler
also uses recent learning utility and successful-upload history when deciding
which feasible clients and channels to favor.

## Why M2 needs both learning and communication information

Federated Averaging (FedAvg) combines received client models in proportion to
the number of local training examples. A wireless scheduler that always picks
the strongest links may repeatedly omit clients with different data. Conversely,
choosing clients only for data value can waste a round on updates that miss the
deadline or energy limit. M2 therefore first enforces feasibility, then ranks
feasible participants using a learning-utility estimate and a participation-debt
term. The RIS is optimized for the current participant set using those same
client priorities.

## System model

There are `K` clients, a base station with `M` receive antennas, and an `N`-element
passive RIS. In round `t`, the binary variable `x_k(t)` indicates whether client
`k` is scheduled. The participant cap is

```text
x_k(t) ∈ {0,1},              sum_k x_k(t) ≤ K_max.
```

The RIS phase vector is `phi(t) = [exp(j theta_1), ..., exp(j theta_N)]`, with
unit magnitude at every element. The effective channel is

```text
h_k(phi) = h_d,k + H_BR @ diag(phi) @ h_RU,k.
```

For maximum-ratio combining, the simulator uses `gamma_k = P_k ||h_k||² / sigma²`
and Shannon rate `R_k = B log2(1 + gamma_k)`. These are idealized narrowband
models. They do not include finite-blocklength coding, protocol headers,
interference from other cells, channel-estimation overhead, RIS switching time,
or hardware phase errors. Those limitations must accompany any interpretation.

## Feasibility model

Each client has an estimated local-computation time and upload time:

```text
T_comp,k = n_k E C_k / f_k
T_upload,k = Q_k / R_k
T_k = T_comp,k + T_upload,k.
```

`n_k` is the client's training-set size, `E` is its local epoch count, `C_k` is
the modeled CPU cycles per example, `f_k` is CPU frequency, and `Q_k` is update
payload size in bits. The current simulator assumes synchronous parallel local
training, so round completion is modeled by the slowest selected client:

```text
T_round = max(T_k : x_k = 1).
```

Per-client energy is modeled as

```text
E_k = E_compute,k + P_k T_upload,k.
```

The revised M2 enforces `T_round ≤ T_max` and `E_k ≤ E_max` for every admitted
client. Its energy model uses the configured computation-energy-per-example
coefficient plus radio transmit energy. It does not claim a measured battery or
CPU-energy model. Values and units must be calibrated for the target device
scenario before publication.

The link is considered successful if it meets the configured SNR threshold and
the modeled latency and energy constraints. The SNR threshold represents a
coarse outage rule, not a calibrated packet-error curve. An infeasible candidate
is not scheduled. If no client is feasible, the round is recorded as idle; the
algorithm never adds an infeasible client merely to keep training moving.

## Learning utility and participation debt

The scheduler maintains one utility estimate `u_k(t)` per client. After each
successful selected client finishes local training, the server evaluates that
candidate client model on a fixed validation subset drawn from the training
dataset and excluded from all client partitions. If `L_val(w_t)` is the current
global-model validation loss and `L_val(w_k(t))` is the loss after the client's
local update, the observed utility is

```text
g_k(t) = L_val(w_t) - L_val(w_k(t)).
```

The utility estimate is updated by exponential smoothing:

```text
u_k(t) = beta u_k(t-1) + (1-beta) g_k(t),
```

where `beta` is `selection.utility_ema_momentum`. This is a **proxy** based on a
client's most recent observed update, not a guarantee that its next update will
help. Unselected clients retain stale estimates. The test set is never used for
selection or utility feedback.

The participation-debt signal is larger for clients with fewer successful
uploads. Utility and debt are converted to within-round rank scores in `[0,1]`
so that their raw units do not dominate one another. The configured priority is

```text
q_k(t) = lambda_U rank(u_k(t))
       + lambda_F rank(-successful_uploads_k(t)),
```

with `lambda_U + lambda_F = 1`. The defaults are 0.7 and 0.3. They are
experimental hyperparameters and require a sensitivity study. Initially,
utilities are equal; participation debt breaks ties toward less-served clients.

This validation-feedback design assumes the coordinator can inspect individual
client updates and evaluate them on a held-out validation set. It is not
compatible with secure aggregation that hides individual updates, and it is not
a differential-privacy guarantee. A deployment requiring those properties
needs a different utility signal.

## Joint objective and solution heuristic

The conceptual per-round problem is

```text
maximize over x, phi:  sum_k x_k p_k(phi) q_k(t)
subject to:
  sum_k x_k ≤ K_max,
  T_round(x, phi) ≤ T_max,
  E_k(phi) ≤ E_max for every selected client,
  |phi_n| = 1 for every RIS element.
```

`p_k(phi)` is the modeled probability of successful delivery. In the current
implementation the channel realization and threshold rules give a deterministic
per-round feasibility decision, so this is an expected-utility interpretation
of a heuristic, not a stochastic optimizer. The larger scientific objective is
to reduce wall-clock time to a target test accuracy; the per-round objective is
only a tractable proxy for that training-level objective.

The implementation uses feasibility-first alternating optimization:

1. Estimate client rates, completion times, and energies under an initial
   weighted sum-SNR RIS configuration.
2. Remove clients that violate a hard resource budget.
3. Greedily choose up to `K_max` feasible clients in descending priority order.
4. Re-optimize the RIS for the selected clients. Each RIS element is updated by
   coordinate descent to increase a priority-weighted sum of channel gains.
5. Recalculate feasibility and repeat until the participant set stops changing
   or the configured iteration limit is reached.
6. Recheck all selected clients under the final RIS configuration. Return an
   empty set if none remain feasible.
7. Train only successful selected clients, evaluate their individual candidate
   models on the validation subset, update utility and participation histories,
   and aggregate successful models with sample-weighted FedAvg.

This method is a greedy alternating heuristic. It does not optimize validation
loss directly during RIS phase updates, and it has no global-optimality or
convergence guarantee for the mixed-integer non-convex problem. The paper should
describe those facts explicitly.

## Baselines and ablations

All baselines share the same channel realization, client partition, model
initialization, participant cap, hard resource filtering, and seed:

| Condition | Client rule | RIS rule | Purpose |
|---|---|---|---|
| Random feasible | Uniform among feasible clients | None | Feasibility-aware reference |
| Channel-only feasible | Highest SNR among feasible clients | None | Isolate channel ranking |
| Random RIS + channel-only | Highest SNR under random phases | Random phases | Isolate phase optimization |
| Proposed, no RIS | Utility and participation debt | None | Isolate RIS contribution |
| Proposed M2 | Utility and participation debt | Priority-weighted AO | Full proposed heuristic |
| Proposed without utility | Participation debt only | Priority-weighted AO | Ablate validation utility |
| Proposed without debt | Validation utility only | Priority-weighted AO | Ablate participation fairness |

The present experiment code has the first three strategies and the proposed
strategy. The final three ablations are still required before the full protocol
described here is complete.

## Data and evaluation protocol

M2 uses the configured MNIST or CIFAR-10 dataset. CIFAR-10 is the primary
publication dataset; MNIST and the reduced `config_fast.yaml` run are smoke or
debug configurations, not final evidence. A fixed validation fraction is
removed from the training split before client partitioning. M2 supports IID or
Dirichlet non-IID client partitions; report the alpha value. The held-out test
set is used only for evaluation.

For each method and seed, preserve the exact configuration, per-round CSV,
summary JSON, and plots. Compare methods with paired client partitions and
channel random-number streams when possible. Use multiple independent seeds and
report mean and standard deviation or confidence intervals. Report accuracy
versus both communication rounds and modeled elapsed time, plus loss, time and
rounds to target, selected/successful client counts, outage, rate, maximum
per-round latency, mean per-client energy, and Jain fairness over cumulative
successful uploads. Also report the fraction of idle rounds and explicit
constraint-violation counts (which should be zero for the proposed constrained
method).

Budget sweeps should include ranges with feasible and infeasible operation.
Do not hide infeasible rounds or compare methods with different allowed client
counts. A short MNIST run with few rounds cannot support claims about CIFAR-10
performance, asymptotic convergence, statistical significance, or journal
novelty.

## What changed from the original M2 plan

The original M2 proposed joint client selection and RIS design under latency and
energy budgets, compared with random and channel-only selection. The revised
method adds or clarifies:

1. A learning-aware utility estimate from validation-loss change, with a
   separate validation set and no test-set feedback.
2. A successful-upload debt term to reduce repeated client exclusion.
3. Priority-weighted RIS phase optimization so the RIS objective reflects the
   scheduler's selected users rather than unweighted sum-SNR alone.
4. A strict feasibility-first rule; no forced client is inserted after an empty
   feasible set.
5. Synchronous round latency as the maximum completion time among participants.
6. Matched, feasibility-aware baselines and utility/fairness ablations.
7. Multi-seed process pooling for independent CPU experiments; CUDA execution is
   kept serial by default to avoid GPU memory contention.
8. Explicit separation of smoke-test outputs from final experimental evidence.

## Dependencies and novelty caveat

No additional third-party package is required for this M2 revision. Validation
splits use existing NumPy/PyTorch/TorchVision components; independent-seed
process pooling uses Python's standard library. MNIST and CIFAR-10 are already
supported, so no new dataset is required. CIFAR-10 should be used for the
primary final study, with MNIST reserved for smoke runs.

Joint RIS design and client selection in wireless FL have prior art. For
example, Liu et al. formulate joint device selection, over-the-air transceiver
design, and RIS configuration ([paper](https://arxiv.org/abs/2011.10282)); a
later IRS-FL work jointly studies scheduling, RIS phases, and resource
allocation across multiple access protocols ([paper](https://arxiv.org/abs/2412.00422)).
Therefore, the contribution should not be described as the first joint RIS/client
selection method. The revised learning-utility and participation-debt heuristic
is a candidate angle to investigate, not a verified novelty claim. A systematic
comparison against close work and complete ablation evidence are required before
making publication claims.
