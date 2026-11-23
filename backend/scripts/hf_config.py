"""Shared HF repo URLs for standalone scripts.

App code (under app/) uses config.py. Scripts (under scripts/)
cannot import app.config without sys.path hacks, so they use this.
"""

HF_DATASET_REPO = "marcellorusso/orchid-osr-dataset"
HF_MODELS_REPO = "marcellorusso/orchid-osr-models"
