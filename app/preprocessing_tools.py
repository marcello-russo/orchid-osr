# app/preprocessing_tools.py
"""
Compatibility alias for loading old pickled datasets.

This module redirects all imports to app.preprocessing for backward compatibility
with .pt files created before the refactoring.
"""

import sys

# Import the new preprocessing package
from app import preprocessing

# Alias the module so pickle can find it when loading old .pt files
sys.modules['app.preprocessing_tools'] = preprocessing

# Alias submodules
sys.modules['app.preprocessing_tools.rotation'] = preprocessing.rotation
sys.modules['app.preprocessing_tools.resizing'] = preprocessing.resizing
sys.modules['app.preprocessing_tools.normalization'] = preprocessing.normalization
sys.modules['app.preprocessing_tools.reproducibility'] = preprocessing.reproducibility
sys.modules['app.preprocessing_tools.dataset_tool'] = preprocessing.dataset_tool

# Re-export common items for direct import
from app.preprocessing.reproducibility import set_seed, seed_worker
from app.preprocessing.dataset_tool import (
    TensorDatasetWithName,
    ImageFolderWithName,
    SingleFolderDataset,
    getDatasetFromFile,
    augmentDataPath,
)
