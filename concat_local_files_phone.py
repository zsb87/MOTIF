import os
import numpy as np
import pandas as pd
import re
import sys
import csv

walk_dir = './right/Eric r structure/Gyroscope/9-16-2016/'
out_file = './right/data.csv'

result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(walk_dir) for f in filenames if os.path.splitext(f)[1] == '.csv']
df_list = [pd.read_csv(file) for file in result]
final_df = pd.concat(df_list)
final_df= final_df.sort_values('Time')
final_df.to_csv(out_file)



# print('walk_dir = ' + walk_dir)

# # If your current working directory may change during script execution, it's recommended to
# # immediately convert program arguments to an absolute path. Then the variable root below will
# # be an absolute path as well. Example:
# # walk_dir = os.path.abspath(walk_dir)
# print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))

# for root, subdirs, files in os.walk(walk_dir):
#     # print('--\nroot = ' + root)
#     list_file_path = os.path.join(root, 'data.csv')
#     # print('list_file_path = ' + list_file_path)

#     with open(list_file_path, 'w') as list_file:
        
#         # for subdir in subdirs:
#         #     print('\t- subdirectory ' + subdir)

#         for filename in files:
#             file_path = os.path.join(root, filename)

#             # print('\t- file %s (full path: %s)' % (filename, file_path))

#             with open(file_path, 'rb') as f:
#                 f_content = f.read()
#                 list_file.writerow(('The file %s contains:\n' % filename).encode('utf-8'))
#                 list_file.writerow(f_content)
#                 list_file.writerow(b'\n')

