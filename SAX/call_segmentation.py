import matlab.engine
from argparse import ArgumentParser


eng = matlab.engine.start_matlab()
# eng.FG_main_subj('')
# i_subj = 1, run = 5, dist_thres = 1, n_motif =10 # ['Eric']=1 
ans = eng.FG_main_engy(3, 1, 0.7, 27.0)# ['Eric']=1 

print(ans)