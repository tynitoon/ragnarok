"""Paired comparison of two offline_v75 runs on the same rows: mean of A[arm_a] - B[arm_b] with a
cluster-bootstrap 95% interval over the groups.
Usage: C:/Python314/python.exe -m scripts.paired_v75 A_rows.npz arm_a B_rows.npz arm_b"""
import sys

import numpy as np

from scripts.offline_v75 import boot

A, a, B, b = np.load(sys.argv[1]), sys.argv[2], np.load(sys.argv[3]), sys.argv[4]
assert np.array_equal(A["grp"], B["grp"]), "different rows/groups"
d = A[a] - B[b]
print(f"{sys.argv[1]}:{a} - {sys.argv[3]}:{b} = {d.mean():+.3f} {boot(d, A['grp'])}")
