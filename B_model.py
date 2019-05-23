#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:41:03 2019

@author: Mohammed Al-Rawi
"""

from misc_functions import binarize_tensor
    
    
def get_loss_of_B_classify(real_A, real_B): # item is a string 'A', 'A_neg' of the sample     
    optimizer_classify.zero_grad()
    B_ = binarize_tensor(G_AB(real_A).detach())                 
    output = B_classify(B_)      
    target = criterion_classify_labeling(B_, binarize_tensor(real_B)) 
    loss_B = criterion_classify(output, target)
    return loss_B

def train_B_classify(no_epochs):    
    print('Training B-Classifier')    
    B_classify.train()
    loss_of_model = []  
    
    # time.sleep(1);   # pbar = tqdm(total=no_epochs);     # time.sleep(1)    
    loss  = None
    for epoch_ in range(1, no_epochs):
        
        scheduler_B.step()
        total_loss = 0
        # pbar.update(1)
        for i, batch in enumerate(dataloader):
            real_B_pos = batch['B'].type(Tensor).to(device) # +ve Ground Truth
            real_B_neg = batch['B_neg'].type(Tensor).to(device) #  -ve Ground Truth
            loss_B_pos = get_loss_of_B_classify(batch['A'].type(Tensor).to(device), 
                                                real_B_pos) #  ''' +ve phase '''                     
            loss_B_neg = get_loss_of_B_classify(batch['A_neg'].type(Tensor).to(device), 
                                                real_B_neg) #  ''' -ve phase pass '''            
            
            loss =  torch.min(loss_B_pos, loss_B_neg) / (
                    torch.max(loss_B_pos, loss_B_neg)+ 1)           

            loss.backward()
            optimizer_classify.step()
            total_loss += loss.cpu().data.detach().numpy().tolist()
        
        print(', ', total_loss/len(dataloader.dataset), end='')    
        loss_of_model.append(total_loss/len(dataloader.dataset))            
            
    return loss_of_model

def test_B_classify(test_loss, test_dataloader):      
    loss = 0     
    B_classify.eval()
    with torch.no_grad():
       for batch_idx, batch in enumerate(test_dataloader):            
           real_B = batch['B'].type(Tensor) # since we are thresholding, there is no difference between B and B_neg
           real_A_pos = batch['A'].type(Tensor)           
           real_A_neg = batch['A_neg'].type(Tensor)  
           GAN_B_pos =  G_AB(real_A_pos).detach() 
           GAN_B_neg = G_AB(real_A_neg).detach()            
           out_B_pos =  B_classify(binarize_tensor(GAN_B_pos)).detach() # if we do thresholding in training, we should do it here
           out_B_neg =  B_classify(binarize_tensor(GAN_B_neg)).detach()
           if out_B_neg<out_B_pos:  
               B_good = GAN_B_neg
           else: B_good = GAN_B_pos
                   
           loss += test_loss(real_B,  B_good)
           x = G_AB( G_AB(GAN_B_pos) + G_AB(GAN_B_neg ) )
           
           img_sample = torch.cat(
                   (real_A_pos, 
                    real_B, 
                    GAN_B_pos,
                    GAN_B_neg,                   
                    G_AB(GAN_B_pos),
                    G_AB(GAN_B_neg),
                    x                    
                    ),  
                    0)
                                  
           save_image(img_sample, 'images/%s/%s.png' % 
                      (opt.dataset_name, batch_idx), nrow=7, normalize=True)        
                         
    model_id = 1
    torch.save(B_classify.state_dict(), 'saved_models/%s/model_classify_%d.pth' % (opt.dataset_name, model_id))            
    return loss/len(test_dataloader.dataset)


#lr_classify = 0.01
#B_classify = torchvis_models.resnet18(pretrained=True)
#B_classify.fc = torch.nn.Linear(2048, 1) #2048 for 256x256 image
#B_classify.aux_logits = False
#B_classify = B_classify.to(device)
#optimizer_classify = torch.optim.Adam(B_classify.parameters(), lr=lr_classify)
#criterion_classify_labeling = torch.nn.L1Loss() 
#criterion_classify = torch.nn.L1Loss()
#scheduler_B = torch.optim.lr_scheduler.MultiStepLR(optimizer_classify, 
#                                                   milestones=[50, 150, 300] , gamma= 0.1)     


# my_loss = train_B_classify(100)
test_performance = test_B_classify(criterion_classify, val_dataloader) 
print(test_performance.item())