import torch
import torch.nn.functional as F



class OptiHardTripletLoss(torch.nn.Module):
    def __init__(self, margin=2., mean_loss=True, eps=1e-8):
        super(OptiHardTripletLoss, self).__init__()
        self.margin = margin
        self.mean_loss = mean_loss
        self.eps = eps

    def forward(self, features, num_sample_persons, num_sample_imgs):
        loss_list = []

        D = features.mm(features.transpose(-2, -1))
        norms = D.diag().expand(features.size(0), features.size(0))
        D = norms + norms.transpose(-2, -1) - 2. * D
        D = D + self.eps
        D = torch.sqrt(D)
        for i in range(D.size(0)):
            person_id = int(i / num_sample_imgs)

            # same person
            temp_same_person_loss = torch.max(D[i][person_id * num_sample_imgs:(person_id + 1) * num_sample_imgs])
            # different person

            if person_id == 0:
                temp_diff_person_loss = torch.min(D[i][(person_id + 1) * num_sample_imgs:])
            elif person_id == num_sample_persons - 1:
                temp_diff_person_loss = torch.min(D[i][0:person_id * num_sample_imgs])
            else:
                temp_diff_person_loss = torch.min(
                    torch.cat((D[i][0:person_id * num_sample_imgs], D[i][(person_id + 1) * num_sample_imgs:])))
            loss = F.softplus(self.margin + temp_same_person_loss - temp_diff_person_loss)
            loss_list.append(loss)
        if self.mean_loss:
            return sum(loss_list) / float(features.size()[0])
        return sum(loss_list)
    

if __name__ == '__main__':    
    criterion = OptiHardTripletLoss(mean_loss=False, margin=2.0, eps=1e-08)
    batch_size = 2
    num_sample_persons = 2
    num_sample_imgs = 3
    num_f = num_sample_persons * num_sample_imgs
    img_torch = torch.rand(size=(batch_size, num_f, 3, 224, 224)) * 2 - 1
    def fake_model(img):
        fake_return = [torch.rand(size=(batch_size*num_f, 288)) * 2 - 1] * 5
        return fake_return

    model = fake_model
    outs = model(img_torch)
    
    loss_list = []
    for out in outs:
        loss_list.append(
            criterion(out, num_sample_persons=num_sample_persons, num_sample_imgs=num_sample_imgs)
        )
    loss = sum(loss_list)
    
    print('End')