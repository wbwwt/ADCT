"""RT-DETR inference registry.

Modified by the ADCT authors from the official Apache-2.0 RT-DETR source.
Only model components needed for frozen inference are imported here. The
remaining upstream modules are intentionally not imported during startup.
"""

from . import nn
from . import zoo
