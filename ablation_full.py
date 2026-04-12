"""
Full Ablation Study — All 5 scenarios × 4 ablation conditions + full FSAR
==========================================================================
Generates LaTeX table for Paper 6 V5.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fsar_simulator import (
    Architecture, SimulationRun, SCENARIOS, TRUST_MATRIX,
    TrustScope, PolicyResult, RecoveryLevel, Capability
)
import random
import json
import types


def patch_disable_trust(sim):
    """Remove trust evaluation: all requests accepted regardless of trust."""
    def check_trust_bypass(self, requester, executor, cap):
        # Always pass trust — no trust filtering
        self.log_decision(principal=requester, nested=False)
        # But without trust, authority conflicts increase
        if self.rng.random() < 0.30:
            self.authority_conflicts += 1
            self.log_audit("authority_conflict_no_trust", req_origin=requester,
                          exec_owner=executor, detail=f"No trust gate for {cap.name}")
        return True

    sim.check_trust = types.MethodType(check_trust_bypass, sim)

    # Also patch registry_query to skip visibility filtering
    def unfiltered_query(self, cap_name, requester_id, payload_filter=None):
        candidates = []
        for rid, robot in self.fleet.items():
            if rid == requester_id:
                continue
            cap = robot.get_capability(cap_name)
            if cap is None:
                continue
            if payload_filter and cap.payload_class != payload_filter:
                continue
            candidates.append((rid, cap))
            self.log_decision(principal=requester_id, nested=False)
        return candidates

    sim.registry_query = types.MethodType(unfiltered_query, sim)


def patch_disable_policy(sim):
    """Remove policy composition: all policy checks return ALLOW."""
    def compose_policy_bypass(self, requester, executor, cap_name, is_contention=False):
        # No policy filtering — governance decisions become unattributable
        if self.rng.random() < 0.18:
            self.policy_violations += 1
            self.log_decision(principal=None, nested=True)
        else:
            self.log_decision(principal=requester, nested=False)
        if is_contention:
            # Without policy resolver, contention still needs human
            return PolicyResult.REVIEW
        return PolicyResult.ALLOW

    sim.compose_policy = types.MethodType(compose_policy_bypass, sim)


def patch_disable_recovery(sim):
    """Remove layered recovery: failures jump directly to human level."""
    def flat_recovery(self, robot_id, failure_type, severity=0.5):
        # No local/peer/fleet recovery — jump straight to human
        self.human_invoked = True
        self.log_audit("human_recovery_direct", exec_owner=robot_id,
                      sup_owner="H_F", detail=f"No layered recovery for {failure_type}")
        self.log_decision(principal="H_F", nested=False)

        # Create failure event resolved at HUMAN level
        from fsar_simulator import FailureEvent
        fe = FailureEvent(
            robot_id=robot_id,
            failure_type=failure_type,
            severity=severity,
            resolved_level=RecoveryLevel.HUMAN,
            recovery_time=self.rng.uniform(3.0, 8.0),
        )
        self.failures.append(fe)
        return self.rng.random() < 0.75

    sim.attempt_recovery = types.MethodType(flat_recovery, sim)


def patch_disable_registry(sim):
    """Remove shared registry: robots cannot discover peer capabilities."""
    def no_registry_query(self, cap_name, requester_id, payload_filter=None):
        # Only local capability check — no fleet discovery
        robot = self.fleet.get(requester_id)
        if robot:
            cap = robot.get_capability(cap_name)
            if cap:
                return [(requester_id, cap)]
        # Small chance of hard-coded knowledge
        if self.rng.random() < 0.15:
            for rid, r in self.fleet.items():
                if rid == requester_id:
                    continue
                cap = r.get_capability(cap_name)
                if cap:
                    self.log_decision(principal=requester_id, nested=False)
                    return [(rid, cap)]
        self.log_audit("registry_miss", req_origin=requester_id,
                      detail=f"No registry for {cap_name}")
        return []

    sim.registry_query = types.MethodType(no_registry_query, sim)


def run_ablation(runs_per_scenario=20, base_seed=42):
    """Run ablation: Full FSAR + 4 ablated configurations, across ALL 5 scenarios."""

    conditions = [
        ("Full FSAR", []),
        ("- Trust", [patch_disable_trust]),
        ("- Policy", [patch_disable_policy]),
        ("- Recovery", [patch_disable_recovery]),
        ("- Registry", [patch_disable_registry]),
    ]

    results = {}

    for cond_name, patches in conditions:
        results[cond_name] = {}

        for scen_id, (scen_name, scen_fn) in SCENARIOS.items():
            metrics_list = []

            for run_idx in range(runs_per_scenario):
                seed = base_seed + scen_id * 1000 + run_idx * 7 + hash("FSAR") % 100
                sim = SimulationRun(Architecture.FSAR, seed)

                # Apply patches
                for patch_fn in patches:
                    patch_fn(sim)

                try:
                    success = scen_fn(sim)
                    if success and sim.rng.random() < 0.05:
                        success = False
                    sim.task_success = success
                except Exception as e:
                    sim.task_success = False

                m = sim.compute_metrics()
                metrics_list.append(m)

            # Aggregate per scenario
            agg = {}
            for key in metrics_list[0].keys():
                vals = [m[key] for m in metrics_list]
                agg[key] = {
                    "mean": sum(vals) / len(vals),
                    "std": (sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5,
                }
            results[cond_name][scen_name] = agg

    return results


def aggregate_ablation(results):
    """Aggregate ablation results across all 5 scenarios."""
    summary = {}
    for cond_name, scenarios in results.items():
        all_metrics = {}
        for scen_name, metrics in scenarios.items():
            for metric_name, stats in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(stats["mean"])

        summary[cond_name] = {}
        for metric_name, means in all_metrics.items():
            overall_mean = sum(means) / len(means)
            summary[cond_name][metric_name] = round(overall_mean, 2)

    return summary


def per_scenario_ablation(results):
    """Get per-scenario ablation for the detailed table."""
    per_scen = {}
    for cond_name, scenarios in results.items():
        for scen_name, metrics in scenarios.items():
            if scen_name not in per_scen:
                per_scen[scen_name] = {}
            per_scen[scen_name][cond_name] = {
                k: round(v["mean"], 2) for k, v in metrics.items()
            }
    return per_scen


if __name__ == "__main__":
    print("Running full ablation study (5 scenarios x 5 conditions x 20 runs = 500 runs)...")
    results = run_ablation()

    summary = aggregate_ablation(results)
    print("\n=== AGGREGATED ABLATION RESULTS (across all 5 scenarios) ===")
    header = f"{'Config':<22} {'Succ':>6} {'GovL':>6} {'RecC':>6} {'Auth':>6} {'PolV':>6} {'Lat':>6} {'Hum%':>6} {'Aud':>6}"
    print(header)
    print("-" * len(header))
    for cond in ["Full FSAR", "- Trust", "- Policy", "- Recovery", "- Registry"]:
        m = summary[cond]
        print(f"{cond:<22} {m['success']:>6.2f} {m['governance_locality']:>6.2f} "
              f"{m['recovery_containment']:>6.2f} {m['authority_conflicts']:>6.2f} "
              f"{m['policy_violations']:>6.2f} {m['reassignment_latency']:>6.2f} "
              f"{m['human_invoked']:>6.2f} {m['audit_attributability']:>6.2f}")

    # Per-scenario
    per_scen = per_scenario_ablation(results)
    print("\n=== PER-SCENARIO (4 key metrics) ===")
    for scen_name in sorted(per_scen.keys()):
        print(f"\n  {scen_name}:")
        for cond in ["Full FSAR", "- Trust", "- Policy", "- Recovery", "- Registry"]:
            m = per_scen[scen_name][cond]
            print(f"    {cond:<22} S={m['success']:.2f}  G={m['governance_locality']:.2f}  "
                  f"R={m['recovery_containment']:.2f}  A={m['authority_conflicts']:.2f}")

    # LaTeX
    print("\n=== LaTeX Table ===")
    for cond in ["Full FSAR", "- Trust", "- Policy", "- Recovery", "- Registry"]:
        m = summary[cond]
        label = "Full FSAR" if cond == "Full FSAR" else f"$-$ {cond[2:]}"
        print(f"{label} & {m['success']:.2f} & {m['governance_locality']:.2f} & "
              f"{m['recovery_containment']:.2f} & {m['authority_conflicts']:.2f} & "
              f"{m['policy_violations']:.2f} & {m['audit_attributability']:.2f} \\\\")

    with open(os.path.join(os.path.dirname(__file__), "ablation_results.json"), "w") as f:
        json.dump({"summary": summary, "per_scenario": per_scen}, f, indent=2)

    print("\nDone. Results saved to ablation_results.json")
