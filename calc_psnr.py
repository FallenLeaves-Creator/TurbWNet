import os
import collections
import numpy as np

root="/media/mayue/Data_8T_D/Projects/ZhangXiao/X-Restormer/results/Turbulence_removal/visualization/Turb"
# root="/media/mayue/Data_8T_D/Projects/ZhangXiao/X-Restormer/results/Turbulence_removal_archived_20240522_091749/visualization/Turb"
filename=collections.OrderedDict()
for file in os.listdir(root):
    name_list=file.split('_')
    file_name=name_list[2]
    psnr=float(name_list[-1].split(".png")[0])
    if file_name in filename:
        if psnr>filename[file_name]:
            filename[file_name]=psnr
    else:
        filename[file_name]=psnr
best_psnr=sum(filename.values())/len(filename)
print(best_psnr)

