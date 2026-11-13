"""Calibrate on synthetic development summaries; no participant data or fitting.

Run from an installed environment: python examples/toy_tolerance_calibration.py
"""

import json

from wrapevofs import LockingCandidate, LockingConfig, calibrate_tolerance, lock_representative_run


def run_demo():
    banks = {}
    for bank_id in ["bank_a", "bank_b", "bank_c"]:
        banks[bank_id] = [
            LockingCandidate(0, [bank_id], 0.90, [0.88, 0.92]),
            LockingCandidate(1, ["shared"], 0.89, [0.87, 0.91]),
            LockingCandidate(2, ["shared"], 0.885, [0.865, 0.905]),
        ]
    fitted = calibrate_tolerance(banks, fold_ids=["development_split_0", "development_split_1"])
    print(json.dumps(fitted.metadata, indent=2))
    print(fitted.curve[["delta", "calibration_Jaccard", "calibration_mean_score", "score_feasible", "selected"]])

    # Freeze the fitted allowance before applying it to a subsequent bank.
    subsequent_bank = [
        LockingCandidate(0, ["new_a"], 0.91),
        LockingCandidate(1, ["new_a", "new_b"], 0.90),
    ]
    locked = lock_representative_run(subsequent_bank, LockingConfig(
        enabled=True, strategy="regret_constrained_medoid",
        tolerance_mode="absolute", regret_tolerance=fitted.selected_delta,
    ))
    print("Subsequent bank signature:", locked.selected_features)
    return fitted


if __name__ == "__main__":
    run_demo()
