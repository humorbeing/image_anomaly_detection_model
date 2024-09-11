from upjab.tool.split_into_folders import split_into_folders


target_root = 'dataset/working_folder/dataset2/normal_split_remain_histogram'
target_root = 'dataset/working_folder/dataset5/remian_round2_histogram'
for i in range(10):
    target_folder = f'{target_root}/{i}'
    split_into_folders(
            target_folder=target_folder,
            file_extends=['jpg', 'JPG'],
            cutting=50,
            is_shuffle=True
        )

print('end')