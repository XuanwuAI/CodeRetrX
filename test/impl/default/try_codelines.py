from rich import print
from coderetrx.static.codebase import Codebase, File

code_file = File.jit_for_testing('tst.py', """
import os

def x():
    def y():
        return x
    print("hello world!")
    for i in range(1000):
        print(i)
    
    print("hello world! Again!!")
    g = 20
    do(sth)
    print("What on earth are we doing")
    y = 10
    return y

x()
"""
)

code_file.init_all()

for line in code_file.get_lines(max_chars=100):
    print(line)
