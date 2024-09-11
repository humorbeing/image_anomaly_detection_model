from upjab.tool.get_file_path_list import get_file_path_list

target_folder = 'dataset/working_folder/dataset3/abnormal7-9'
target_folder = 'dataset/working_folder/dataset5/round3_normal'
file_list = get_file_path_list(target_folder)
# file_list2 = glob.glob(target_folder + '/**/*.jpg', recursive=True)
# file_list.extend(file_list2)

import shutil
import os
save_folder_path = target_folder + '_onefolder'
os.makedirs(save_folder_path, exist_ok=True)
# save_folder_path
for f_ in file_list:
    # orginal_folder = f_.split('/')[:-1]
    # save_folder_path = os.path.join(save_folder, *orginal_folder)
    
    shutil.copy(f_, save_folder_path)  # dst can be folder