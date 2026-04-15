import os
import sys

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE_ROOT)
from config import RAW_DATA_ROOT

path = os.path.join(RAW_DATA_ROOT, "wmi-md", "NVT", "FSI", "298", "bin.77")
with open(path,"rb") as f:
    a = f.readline()
    print(a)
    a = f.readline()
    
    print(a)
    a = f.readline()
    
    print(a)
    a = f.readline()
    print(a)
    a = f.readline()
    print(a)
