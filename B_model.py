#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:41:03 2019

@author: malrawi
"""

def binarize_tensor(img):
    thresh_val = img.mean()    
    img = (img>thresh_val).float()*1       
    img = 2*img-1
    
    return img
    
    
def get_loss_of_B_classify(real_A, real_B): # item is a string 'A', 'A_neg' of the sample     
    optimizer_classify.zero_grad()
    B_ = G_AB(real_A).detach()                 
    output = B_classify(B_)  
    # target = criterion_classify_labeling(B_, real_B) # we can threshold B_pos and real_B here
    target = criterion_classify_labeling(binarize_tensor(B_), binarize_tensor(real_B)) # we can threshold B_pos and real_B here
    loss_B = criterion_classify(output, target)
    return loss_B

def train_B_classify(no_epochs):    
    print('Training B-Classifier')    
    B_classify.train()
    loss_of_model = []    
    # time.sleep(1);   # pbar = tqdm(total=no_epochs);     # time.sleep(1)    
    for epoch_ in range(1, no_epochs):
        total_loss = 0
        # pbar.update(1)   
        print('.', end='')
        for i, batch in enumerate(dataloader):
            real_B = batch['B'].type(Tensor).to(device) # +ve Ground Truth
            real_B_neg = batch['B_neg'].type(Tensor).to(device) #  -ve Ground Truth
            loss_B_pos = get_loss_of_B_classify(batch['A'].type(Tensor).to(device), real_B) #  ''' -ve phase '''                     
            loss_B_neg = get_loss_of_B_classify(batch['A_neg'].type(Tensor).to(device), 
                                                real_B_neg) #  ''' +ve phase pass '''            
            loss = (loss_B_pos+loss_B_neg)/2            
            loss.backward()
            optimizer_classify.step()
            total_loss += loss.cpu().data.detach().numpy().tolist()
            
        loss_of_model.append(total_loss/len(dataloader.dataset))
            
            # should we add loss_B to loss_Bneg and use one backward and step?
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
           GAN_B_neg = G_AB(real_A_pos).detach()            
           out_B_pos =  B_classify(GAN_B_pos).detach()
           out_B_neg =  B_classify(GAN_B_neg).detach()
           if out_B_neg<out_B_pos:  
               B_good = GAN_B_neg
           else: B_good = GAN_B_pos
           loss += test_loss(real_B,  B_good)
           
           img_sample = torch.cat(
                   (real_A_pos.data, 
                    real_B.data,                      
                    B_good.data, 
                    binarize_tensor(B_good),
                    
                    ),  
                    0)
                                  
           save_image(img_sample, 'images/%s/%s.png' % 
                      (opt.dataset_name, batch_idx), nrow=1, normalize=True)        
                         
    model_id = 1
    torch.save(B_classify.state_dict(), 'saved_models/%s/model_classify_%d.pth' % (opt.dataset_name, model_id))            
    return loss/len(test_dataloader.dataset)

my_loss = train_B_classify(100)
test_performance = test_B_classify(criterion_classify, val_dataloader) 
print(test_performance)