"""Repo-root scripts package.

Marks ``scripts/`` as importable so ``scripts.backends.<robot>`` can be passed
to ``run_episode.py --backend-loader`` regardless of how the launcher is
invoked (``python scripts/run_episode.py`` vs. ``python -m …``).
"""
