# Federated Single-Agent Robotics: Fleet Coordination Under Governed Autonomy

Simulation code for the paper:

> Xue Qin, Simin Luan, John See, Cong Yang, Zhijun Li. "Federated Single-Agent Robotics: Fleet Coordination Under Governed Autonomy." 2026.

> **Looking for the full production runtime?** See [aeros-runtime](https://github.com/s20sc/aeros-runtime) — the complete implementation with governance engine, evolution engine, fleet coordination, benchmarking, provider SDK, and marketplace.

**Project Page**: https://s20sc.github.io/aeros-project

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
  title={Federated Single-Agent Robotics: Fleet Coordination Under Governed Autonomy},
  author={Qin, Xue and Luan, Simin and See, John and Yang, Cong and Li, Zhijun},
  year={2026}
}
```
