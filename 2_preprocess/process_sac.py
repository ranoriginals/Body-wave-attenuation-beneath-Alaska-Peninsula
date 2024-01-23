# Program to process sacfiles


import os
import glob
import datetime
import subprocess
import numpy as np
from mpi4py import MPI
from obspy import read_inventory


def check_single_channel():
    os.putenv("SAC_DISPLAY_COPYRIGHT", '0')
    cmd = "ls *Z.sac | awk -F '.' '{print $1,$2}'"
    output = os.popen(cmd).read().strip().split('\n')
    for line in output:
        line = line.strip()
        net = line.split()[0]
        sta = line.split()[1]
        lst = glob.glob("%s.%s.*.sac" %(net,sta))
        if len(lst) == 3:
            continue
        else:
            for sacfile in lst:
                chan = sacfile.split(".")[3]
                print(sacfile, chan)
                new_chan1 = chan[:-1]+"E"
                new_sacfile1 = net+"."+sta+"."+new_chan1+".sac"
                s = ""
                s += "read %s \n" %(sacfile)
                s += "ch KCMPNM %s \n" %(new_chan1)
                s += "w %s \n" %(new_sacfile1)
                s += "q \n"
                subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())

                new_chan2 = chan[:-1]+"N"
                new_sacfile2 = net+"."+sta+"."+new_chan2+".sac"
                s = ""
                s += "read %s \n" %(sacfile)
                s += "ch KCMPNM %s \n" %(new_chan2)
                s += "w %s \n" %(new_sacfile2)
                s += "q \n"
                subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())


def add_sac_header(sacfile,event_id,catalog,picks_file):
    os.putenv("SAC_DISPLAY_COPYRIGHT",'0')
    net = sacfile.split(".")[0]
    sta = sacfile.split(".")[1]

    # evlo, evla, evdp
    command1 = "cat %s | awk '/%s/ {print}'" %(catalog, event_id)
    event_info = os.popen(command1).read().strip()
    evlo = float(event_info.split()[1])
    evla = float(event_info.split()[2])
    evdp = float(event_info.split()[3])

    # P(T0) and S(T1) wave arrival
    o = 0
    t0 = 0
    t1 = 0
    command2 = "cat %s | awk '/%s/ {print}' | awk '/%s/ {print}'" \
               %(picks_file, event_id, sta)
    picks = os.popen(command2).read().strip().split("\n")
    if picks[0] != "":
        for pick in picks:
            pick = pick.strip()
            pick_sp = pick.split()
            phase = pick_sp[-1]
            origintime = pick_sp[1]
            o = datetime.datetime.strptime(origintime,"%Y-%m-%dT%H:%M:%S.%fZ")
            o_jday = o.strftime("%j")
            o_msec = int(o.microsecond/1000+0.5)
            if phase == "P":
                t0 = pick_sp[3]
                t0 = datetime.datetime.strptime(t0,"%Y-%m-%dT%H:%M:%S.%fZ")
                t0_jday = t0.strftime("%j")
                t0_msec = int(t0.microsecond/1000+0.5)
            elif phase == "S":
                t1 = pick_sp[3]
                t1 = datetime.datetime.strptime(t1,"%Y-%m-%dT%H:%M:%S.%fZ")
                t1_jday = t1.strftime("%j")
                t1_msec = int(t1.microsecond/1000+0.5)
    
    s = ""
    s += "readhdr %s.%s.*.sac \n" %(net,sta)
    s += "ch evla %s \n" %(evla)
    s += "ch evlo %s \n" %(evlo)
    s += "ch evdp %s \n" %(evdp)
    s += "ch lcalda True \n" # calculate distance and azimuth
    if type(o) is datetime.datetime:
        s += "ch o gmt {} {} {} {} {} {} \n".format(o.year,o_jday,o.hour,
                                             o.minute,o.second,o_msec)
    if type(t0) is datetime.datetime:
        s += "ch t0 gmt {} {} {} {} {} {} \n".format(t0.year,t0_jday,t0.hour,
                                              t0.minute,t0.second,t0_msec)
    if type(t1) is datetime.datetime:
        s += "ch t1 gmt {} {} {} {} {} {} \n".format(t1.year,t1_jday,t1.hour,
                                              t1.minute,t1.second,t1_msec)
    s += "wh \n"
    s += "q \n"
    subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())


def check_T0(sacfile):
    command = "saclst t0 f %s" %(sacfile)
    output = os.popen(command).read().strip()
    T0 = float(output.split()[1])
    if T0 == -12345:
        os.remove("./%s" %(sacfile))


def sactotxt(sacfile):
    os.putenv("SAC_DISPLAY_COPYRIGHT", '0')
    txt_filename = sacfile.replace("sac","txt")
    s = ""
    s += "r %s \n" %(sacfile)
    s += "ch allt (0 - &1,T0&) iztype IT0 \n" # set T0 as zero time
    s += "w alpha %s \n" %(txt_filename)
    s += "q \n"
    subprocess.Popen(['sac'], stdin=subprocess.PIPE).communicate(s.encode())


main_path = os.getcwd()
catalog = main_path+"/AACSE_catalog.dat"
picks_file = main_path+"/picks.dat"
seismograms = main_path+"/processedSeismograms"
dirs = os.listdir(seismograms)

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
    os.chdir("%s/%s" %(seismograms,dirname))

    check_single_channel()

    sac_list = glob.glob("*Z.sac")
    for sac in sac_list:
        add_sac_header(sac,dirname,catalog,picks_file)
    
    sacfls = glob.glob("*.sac")
    for sacfile in sacfls:
        check_T0(sacfile)
    
    sacfls = glob.glob("*.sac")    
    for sacfile in sacfls:
        sactotxt(sacfile)