from dataset_loader import Dataset_normal, Dataset_abnormal, Dataset_test
from torch.utils.data import DataLoader

def get_dataset_loader(batch_size):

    ntrain_dataset = Dataset_normal()

    ntrain_loader = DataLoader(
        ntrain_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    abtrain_dataset = Dataset_abnormal()
    abtrain_loader = DataLoader(
        abtrain_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    test_batch_size = batch_size
    test_dataset = Dataset_test()
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=False)
    
    return {
        "normal loader": ntrain_loader,
        "abnormal loader": abtrain_loader,
        "test loader": test_loader,
    }