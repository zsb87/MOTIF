import time
import datetime
import numpy as np

ts = time.time()
current_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H-%M-%S')
print(str(current_time))

cm = np.array([0,0])
np.savetxt('F:/RF_185_motif_dist_cm'+str(current_time)+'.csv', cm, delimiter=",")
