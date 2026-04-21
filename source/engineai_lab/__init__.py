"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Ensure AMP data loaders register themselves on import.
from .utils import AMP_data_loader  # noqa: F401
