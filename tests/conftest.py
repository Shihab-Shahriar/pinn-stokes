import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:  # grpy_tensors (RPY) is imported bare by src/mob_op_2b_combined.py
    sys.path.insert(1, SRC)
