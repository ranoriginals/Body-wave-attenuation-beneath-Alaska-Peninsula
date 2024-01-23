# Program to remove instrument response


import os
import sys
import glob
import subprocess
import numpy as np
from mpi4py import MPI


def remove_instrument_response(sacfile, PZfile_path):
    os.putenv("SAC_DISPLAY_COPYRIGHT", '0')
    cmd = "saclst delta f %s | awk -F ' ' '{print $2}'" %(sacfile)
    sampling_rate = 1/float(os.popen(cmd).read().strip())
    new_sacname = sacfile.replace("SAC","sac")

    s = ""
    net, sta, loc, chan = sacfile.split(".")[0:4]
    pz = glob.glob("%s/SAC_PZ_%s_%s" %(PZfile_path, net, sta))
    if len(pz) != 1:
        # sys.exit("PZ file error for %s" %(sacfile))
        print(sacfile)


    cmd2 = "cat %s | awk '/LATITUDE/ {print}'" %(pz[0])
    output2 = os.popen(cmd2).read().strip()
    stla = float(output2.split(":")[-1].strip())
    cmd3 = "cat %s | awk '/LONGITUDE/ {print}'" %(pz[0])
    output3 = os.popen(cmd3).read().strip()
    stlo = float(output3.split(":")[-1].strip())

    s += "r %s \n" %(sacfile)
    s += "rmean; rtr; taper \n"
    if sampling_rate == 40:
        s += "trans from polezero subtype %s to vel freq 0.01 0.05 18 19 \n" %(pz[0])
    elif sampling_rate == 50:
        s += "trans from polezero subtype %s to vel freq 0.01 0.05 22 24 \n" %(pz[0])
    elif sampling_rate == 100:
        s += "trans from polezero subtype %s to vel freq 0.01 0.05 45 48 \n" %(pz[0])
    # s += "mul 1.0e9 \n"
    s += "ch stla %s \n" %(stla)
    s += "ch stlo %s \n" %(stlo)
    s += "w %s \n" %(new_sacname)
    s += "q \n"
    subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())

    os.remove(sacfile)


main_path = os.getcwd()
sac_path = main_path+"/processedSeismograms"
dirs = os.listdir(sac_path)
PZfile_path = "/mnt/home/zhuoran/AACSE/SACPZ"

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
    print("processing on %s" %(dirname))
    os.chdir("%s/%s" %(sac_path, dirname))
    sac_list = glob.glob("*.SAC")
    for sacfile in sac_list:
        remove_instrument_response(sacfile, PZfile_path)