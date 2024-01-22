import os
import obspy
from mpi4py import MPI
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader

## Need to make new directionary "waveforms" and "stations", then this program could run parallel

def download_data(orid):
    ## download data based on picks.dat
    picks_info = "./picks.dat"
    cmd = "grep %s %s" %(orid, picks_info)
    picks = os.popen(cmd).read().strip().split("\n")
    network = []
    station = []

    for i in range(len(picks)):
        pick = picks[i]
        station_info = pick.split()[2]
        net = station_info.split(".")[0]
        sta = station_info.split(".")[1]
        network.append(net)
        station.append(sta)

    # Remove repeated items
    network = list(set(network))
    station = list(set(station))
    network = ",".join([i for i in network])
    station = ",".join([i for i in station])

    origin_time = pick.split()[1]
    origin_time = obspy.UTCDateTime(origin_time)
    start_time = origin_time-60
    end_time = origin_time+480

    domain = RectangularDomain(minlatitude=51, maxlatitude=60, minlongitude=-163, maxlongitude=-147)
    restrictions = Restrictions(starttime=start_time,
                                endtime=end_time,
                                network=network,
                                station=station)
    mseed_storage = f"./waveforms/{orid}" + "/{network}.{station}.{channel}.{location}.{starttime}.{endtime}.mseed"
    stationxml_storage = "./stations"
    mdl = MassDownloader(providers=["IRIS"])
    mdl.download(domain, restrictions, mseed_storage=mseed_storage, stationxml_storage=stationxml_storage)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

picks_info = "./picks.dat"
catalog = "./deep_events.txt"
command = "cat %s | awk -F ' ' '{print $1}'" %(catalog)
oridlst = os.popen(command).read().strip().split("\n")

total_numbers = len(oridlst)
numbers_per_task = total_numbers // size
remainder = total_numbers % size

# Calculate the start and end indices for each task
if rank < remainder:
    start = rank * (numbers_per_task + 1)
    end = start + numbers_per_task + 1
else:
    start = rank * numbers_per_task + remainder
    end = start + numbers_per_task 

for orid in oridlst[start:end]:
    cmd = "grep %s %s" %(orid, picks_info)
    picks = os.popen(cmd).read().strip().split("\n")
    if len(picks) == 0:
        continue
    else:
        download_data(orid)