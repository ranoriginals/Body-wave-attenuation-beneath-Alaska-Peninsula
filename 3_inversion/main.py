import os
import tstar_load
import numpy as np
from mpi4py import MPI
import tstar_parameters as tp
import tstar_inversion_function as tf

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

param = tp.set_parameters()
oridlst = tp.working_dir()
site_dd = tstar_load.loadsite(param['add_site'])

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


# Each core processes its subset of the data
for orid in oridlst[start:end]:

    print(f"Rank {rank}: {orid}")
    
    tp.logfl.write('\nWorking on ORID # %s\n' % orid)
    
    resultfl = tp.resultdir+'/%s_pstar%03d.dat' % (orid,int(param["alpha"]*100))
    if os.path.isfile(resultfl):
        print('Skip %s, already exists' % orid)
        continue
    
    # Load event and station information
    ORIG = tstar_load.loadORIG(orid, param)
    fldir = tp.sacdir+"/%s" %(orid)
    stalst, ARRIV  = tstar_load.loaddata(param, orid, fldir)
    if len(stalst) == 0:
        print('Zero stations for # %s' % orid)
        continue
    os.chdir(tp.workdir)

    # Load constants with geometric spreading and free surface effects
    (PGS, SGS) = tstar_load.loadGS(orid, tp.gsdir)

    # Loop over records
    (staP1lst, staP2lst, staP3lst, staS1lst, staS2lst, staS3lst, saving, ARRIV) = \
        tf.Loop_Record(orid, stalst, param, ARRIV, PGS, SGS, ORIG)

    
    # Find the best fc if grid searching (grid search for best fc)
    if param['source_para'] == 1:
        if len(staP1lst) < 5:
            print('Not enough good P wave record for event %s.' % orid)
            continue
        
        # ## Grid search to find best fc
        # (ORIG, flag) = tf.bestfc(orid, saving, staP1lst, ORIG, 'P', 1, param, site_dd)
        # if flag == 0:
        #     continue
        
        ## Here a fixed corner frequency is applied
        (ORIG, flag) = tf.fixed_bestfc(orid, ORIG)
        if flag == 0:
            continue

    ## Get station list with fitting > param['misfitP']
    staP3lst = tf.get_good_stalst(orid, saving, staP2lst, ORIG, 'P', 2, param)

    if len(staP3lst) <= 5:
        continue

    ## Get constant correction for each event-station pair
    
    saving = tf.get_constant_correction(saving, staP3lst, ORIG, 'P', 3, param)

    ## Final inverion for tstar value
    (ORIG, saving, flag) = tf.inversion(orid, saving, staP3lst, ORIG, 'P', 4, param)

    if flag == 1:
        continue
    else:
        tf.output_results(orid, staP3lst, 'P', param, ORIG, saving, ARRIV)

    if len(staS2lst) <= 2:
        continue

    ## Get station list with fitting > param['misfitS']
    staS3lst = tf.get_good_stalst(orid, saving, staS2lst, ORIG, 'S', 2, param)

    if len(staS3lst) <= 3:
        continue
    
    ## Get constant correction for each event-station pair
    saving = tf.get_constant_correction(saving, staS3lst, ORIG, 'S', 3, param)

    ## Final inverion for tstar value
    (ORIG, saving, flag) = tf.inversion(orid, saving, staS3lst, ORIG, 'S', 4, param)

    if flag == 1:
        continue
    else:
        tf.output_results(orid, staS3lst, 'S', param, ORIG, saving, ARRIV)


tp.fittingfl.close()
tp.logfl.close()
tp.fclist.close()