import sys
import os

jarpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'javalib/NumJy-0.1.0-SNAPSHOT.jar')
if not jarpath in sys.path:
    sys.path.append(jarpath)

from .core import *