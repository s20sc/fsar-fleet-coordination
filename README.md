# Federated Single-Agent Robotics: Multi-Robot Coordination Without Intra-Robot Multi-Agent Fragmentation

Simulation code for the paper:

> Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li. "Federated Single-Agent Robotics: Multi-Robot Coordination Without Intra-Robot Multi-Agent Fragmentation." 2026.
> arXiv: [2604.11028](https://arxiv.org/abs/2604.11028)

## Overview

Protocol-level simulator evaluating three fleet coordination architectures:

- **FSAR** (Federated Single-Agent Robotics) — proposed approach
- **CFC** (Centralized Fleet Controller) — baseline
- **DHMA** (Decomposition-Heavy Multi-Agent) — baseline

### Experiments

| Script | Description |
|--------|-------------|
| `fsar_simulator.py` | Main evaluation: 5 scenarios x 20 runs x 3 architectures + statistical tests |
| `ablation_full.py` | Component ablation study |
| `scaling_experiment.py` | Fleet scaling evaluation (4, 8, 16 robots) |
| `statistical_analysis.py` | Additional statistical analysis |

### Metrics

- Coordination success rate
- Governance locality
- Recovery containment
- Authority conflicts
- Policy violations
- Reassignment latency
- Human intervention rate
- Audit attributability

## Requirements

```
Python >= 3.11
```

No external dependencies required.

## Usage

Run main evaluation:

```bash
python fsar_simulator.py
```

Run ablation study:

```bash
python ablation_full.py
```

Run scaling experiment:

```bash
python scaling_experiment.py
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{qin2026fsar,
  title={Federated Single-Agent Robotics: Multi-Robot Coordination Without Intra-Robot Multi-Agent Fragmentation},
  author={Qin, Xue and Luan, Simin and See, John and Yang, Cong and Li, Zhijun},
  journal={arXiv preprint arXiv:2604.11028},
  year={2026}
}
```
