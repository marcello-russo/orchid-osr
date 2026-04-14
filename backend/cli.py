#!/usr/bin/env python3
"""
Entrypoint per esperimenti di classificazione orchidee.

Tutta la logica è in app.cli.experiments.
Questo file è solo un thin wrapper.

Usage:
    python run_experiments.py --status
    python run_experiments.py --all
    python run_experiments.py --train --exp 4
    python run_experiments.py --help
"""

from app.cli.experiments import main

if __name__ == "__main__":
    exit(main())
