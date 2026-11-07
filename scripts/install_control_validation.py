"""Build a clean lightweight controller without touching training/scorer environments."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv
p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[1]
python=a.prefix.absolute()/'bin/python'
if not python.exists():venv.EnvBuilder(with_pip=True).create(a.prefix)
subprocess.run([str(python),'-m','pip','install','--upgrade','pip','setuptools>=69','wheel'],check=True)
subprocess.run([str(python),'-m','pip','install','-e',str(root)],check=True)
subprocess.run([str(python),'-m','bench_image','protocols'],check=True)
