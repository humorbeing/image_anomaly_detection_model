original_path = 'dataset/working_folder/dataset4/original'
changed_path = 'dataset/working_folder/dataset4/changed'
from upjab.tool.what_missing_in_folder import what_missing
what_missing(
    original_path=original_path,
    changed_path=changed_path,
    file_type_list=['jpg', 'JPG']
)