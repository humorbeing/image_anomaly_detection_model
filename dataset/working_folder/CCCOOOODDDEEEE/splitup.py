from upjab.tool.shuffle_split_folder import shuffle_split_folder

target_folder = 'dataset/working_folder/dataset1/abnormal'
target_folder = 'dataset/working_folder/dataset1/normal'

# target_folder = '../../example_data/videos/fishes/not_crowd'
shuffle_split_folder(
    target_folder=target_folder,
    file_type_list=['jpg', 'JPG'],
    random_seed=5,
    split_ratio=0.3
)