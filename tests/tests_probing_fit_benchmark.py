"""Run the P2 fit benchmark separately from the ordinary test suite.

From the repository root, with the project environment activated:
PYTHONPATH=src python -m pytest -q -s tests/tests_probing_fit_benchmark.py

The tests_ prefix deliberately excludes this file from default pytest discovery.
Passing this file explicitly runs the benchmark without environment flags.
The -s option displays timings and environment details to copy into the PR.
Only synthetic data is generated; no models or datasets are downloaded.
"""

import os
import platform
import time

import numpy as np
import sklearn

from ws1_probing.probes import LinearProbe


def test_benchmark_50k_by_768() -> None:
    """Time the full default C search on 50k x 768 random training features.

    Returns
    -------
    None
        Prints elapsed fit time, dimensions and environment for the P2 PR.
        Data generation and prediction are excluded from the measured duration.
    """
    rng = np.random.default_rng(42)

    X_train = rng.standard_normal((50_000, 768), dtype=np.float32)
    y_train = rng.integers(0, 17, size=len(X_train))
    X_dev = rng.standard_normal((5_000, 768), dtype=np.float32)
    y_dev = rng.integers(0, 17, size=len(X_dev))

    probe = LinearProbe()

    start = time.perf_counter()
    probe.fit(X_train, y_train, X_dev, y_dev)
    elapsed = time.perf_counter() - start
    
    print(
        f"\nP2 benchmark: fit_seconds={elapsed:.3f}; train={X_train.shape}; dev={X_dev.shape}; "
        f"dtype={X_train.dtype}; classes=17; Cs={probe.Cs}; max_iter={probe.max_iter}; seed=42; "
        f"hparams={probe.hparams}; Python={platform.python_version()}; NumPy={np.__version__}; "
        f"scikit-learn={sklearn.__version__}; platform={platform.platform()}; CPUs={os.cpu_count()}"
    )
    assert probe.predict(X_dev).shape == y_dev.shape
