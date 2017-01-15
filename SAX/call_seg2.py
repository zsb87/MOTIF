import matlab.engine
from argparse import ArgumentParser


eng = matlab.engine.start_matlab()
# i_subj = 1, run = 5, dist_thres = 1, n_motif =10 # ['Eric']=1 
ans = eng.FG_main_engy(4, 4, 1.0, 20.0)# ['Eric']=1 

print(ans)