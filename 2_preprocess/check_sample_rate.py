import os
import numpy as np
import subprocess
from mpi4py import MPI


def check_sample_rate():
    cmd = "saclst delta f *.sac"
    output = os.popen(cmd).read().strip().splitlines()
    saclst = []
    for line in output:
        line = line.strip()
        delta = float(line.split()[1])
        if delta != 0.01 and delta != 0.02 and delta != 0.025:
            current_path = os.getcwd()
            sacname = current_path+"/"+line.split()[0]
            print(sacname, delta)
            saclst.append(sacname)

    # if len(saclst) != 0:
    #     os.putenv("SAC_DISPLAY_COPYRIGHT", '0')
    #     s = " "
    #     for i in range(len(saclst)):
    #         s += "readhdr %s \n" %(saclst[i])

    #     s += "ch delta 0.025 \n"
    #     s += "wh \n"
    #     s += "q \n"
    #     subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())    

            
main_path = os.getcwd()
sac_path = main_path+"/processedSeismograms"
dirs = os.listdir(sac_path)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

total_numbers = len(dirs)
numbers_per_task = total_numbers // size
remainder = total_numbers % size

if rank < remainder:
    start = rank * (numbers_per_task + 1)
    end = start + numbers_per_task + 1
else:
    start = rank * numbers_per_task + remainder
    end = start + numbers_per_task 


for dirname in dirs[start:end]:
    os.chdir("%s/%s" %(sac_path, dirname))
    check_sample_rate()

