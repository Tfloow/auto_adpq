# src/auto_adpq/__init__.py

"""Auto-AdpQ.

-----------------------
A brief description of what your package does.
"""

# Import key functions from internal modules to expose them at the top level
from .module import AdpQQuantizedWeights, Auto_AdpQ, AutoAdpQConfig

# Define the package version
__version__ = "0.1.6"

# List of names to expose when a user does `from auto_adpq import *`
__all__ = ["Auto_AdpQ", "AutoAdpQConfig", "AdpQQuantizedWeights", "__version__"]
