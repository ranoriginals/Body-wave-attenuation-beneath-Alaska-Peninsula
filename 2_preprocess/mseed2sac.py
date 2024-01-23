import os
import glob
import numpy as np
from obspy import read
from mpi4py import MPI


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def mseedtosac(miniseed, sac_path):
    st = read(miniseed)
    tr = st[0]
    net = tr.stats.network
    sta = tr.stats.station
    loc = tr.stats.location
    cha = tr.stats.channel
    sacfile = "%s/%s.%s.%s.%s.SAC" %(sac_path,net,sta,loc,cha)
    st.write(sacfile,format="SAC")


main_path = os.getcwd()
miniseed_path = main_path+"/waveforms"
dirs = os.listdir(miniseed_path)
os.system("ls %s > %s/Eventname" %(miniseed_path, main_path))

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
    os.chdir("%s/%s" %(miniseed_path,dirname))
    print("processing on %s" %(dirname))
    sac_path = main_path+"/processedSeismograms/"+dirname
    mkdir(sac_path)
    miniseed_list = glob.glob("*.mseed")
    for miniseed in miniseed_list:
        mseedtosac(miniseed, sac_path)