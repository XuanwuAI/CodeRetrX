from rich import print
from coderetrx.static.codebase import Codebase, File

code_file = File.jit_for_testing('tst.py', """
import os

def x():
    def y():
        return x
    return y

x()
"""
)

code_file.init_all()

for line in code_file.get_lines():
    print(line)
