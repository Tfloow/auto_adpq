# src/auto_adpq/__init__.py

"""Auto-AdpQ.

-----------------------
A brief description of what your package does.
"""

# Import key functions from internal modules to expose them at the top level
from .class_format import AdpQQuantizedWeights, AutoAdpQConfig
from .module import Auto_AdpQ

# Define the package version
__version__ = "0.3.5"

# List of names to expose when a user does `from auto_adpq import *`
__all__ = ["Auto_AdpQ", "AutoAdpQConfig", "AdpQQuantizedWeights", "__version__"]
