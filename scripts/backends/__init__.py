"""Example backend loaders for ``scripts/run_episode.py --backend-loader``.

Each file in this package is a small wrapper that constructs the official
upstream robot object and hands it to the matching harness backend
(``YAMRobotEnvBackend``, ``LeRobotBiDK1Backend``). The harness deliberately
does not pin the upstream robot SDKs as dependencies, so the import lives
here, not in ``vla_harness/``.

Copy a file in this directory, fill in the one stubbed function, and point at
your copy via ``--backend-loader``.
"""
