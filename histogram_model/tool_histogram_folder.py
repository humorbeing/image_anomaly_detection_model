
import os
import glob
from tqdm import tqdm
import shutil

def histogram_folder(
    target_folder,
    detect,
    file_type_list=['JPG','jpg']
    ):

    save_path = target_folder + '_histogram'
    for i in range(10):
        new_path = f'{save_path}/{i}'
        os.makedirs(new_path, exist_ok=True)

    file_list = []
    for ty_ in file_type_list:
        temp_file_list = glob.glob(target_folder + f'/**/*.{ty_}', recursive=True)    
        file_list.extend(temp_file_list)
    

    for video_path in tqdm(file_list):
        score = detect(video_path)
        if score == 1:
            cata = '9'
        else:
            str_score = str(score)
            cata = str_score[2]
        shutil.copy(video_path, f'{save_path}/{cata}')
        


if __name__ == '__main__':
    from detect import detect
    target_folder = 'dataset/working_folder/dataset5/remian_round2'
    # detection_model_ckpt = 'experiment_setups/exp__v0001/ckpt/2024-09-08__01-58-52__101-round1-picklow__SGBDtpmTiZ_best_model_seed_5.pkl'
    # # det = Detector(detection_model_ckpt)
    # det = Detector(detection_model_ckpt, inference_with_max=True)
    histogram_folder(
        target_folder=target_folder,
        detect=detect,
        file_type_list=['jpg', 'JPG']
    )
    print('')