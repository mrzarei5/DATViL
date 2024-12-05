import os
import random
import argparse
import yaml
from tqdm import tqdm
import json

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets.utils import FewShotData, FewShotDataImagenet
from datasets.imagenet import ImageNet
import clip
from utils import *

import pickle
import csv

from networks import DATVIL


def serializeObject(object_,file_name):
    file_object = open(file_name,'wb')
    pickle.dump(object_, file_object,protocol = 2)
    file_object.close()
    return
def deserializeObject(file_name):
    file_object = open(file_name,'rb')
    object_ = pickle.load(file_object)
    file_object.close() 
    return object_


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of each dataset in yaml format')
    parser.add_argument('--shots', dest='shots', type=int, default=1, help='Number of shots')
    parser.add_argument('--root_path', dest='root_path', default='./data', help='Path to the root folder of all datasets')
    
    parser.add_argument('--model_name', dest='model_name', default='datvil', help='The name of the model to train. It can be either datvil or datvilc' )
    parser.add_argument('--per_sample_train', dest='per_sample_train', type=int , default=10, help='The number of per-sample training epochs in DATViL-C')
    parser.add_argument('--plus_residual', dest='plus_residual', type=int , default=1, help='Add residual-based adatpers')
    parser.add_argument('--plus_transform', dest='plus_transform', type=int , default=1, help='Add transormer-based adapters')

    args = parser.parse_args()

    return args


def train_datvil(cfg, test_features, test_labels, att_weights, clip_model, train_loader):
    alpha = cfg['alpha']
    network = DATVIL(att_weights, alpha, clip_model.dtype, cfg['plus_residual'], cfg['plus_transform']).cuda()
   
    optimizer = torch.optim.AdamW([{'params':network.text_attributes_weights, 'lr':cfg['lr_transformer']},
                                   {'params':network.text_attributes_residuals, 'lr':cfg['lr_residual']}], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader))

    for train_idx in range(1,cfg['train_epoch']+1):
        
        network.train()
        # Train
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

 
            updated_classifiers = network()
            
            updated_classifiers = updated_classifiers / updated_classifiers.norm(dim=-1, keepdim=True)

            logits = 100. * image_features @ updated_classifiers.t()
            
            loss = F.cross_entropy(logits, target)
    
            acc = cls_acc(logits, target)
            correct_samples += acc / 100 * len(logits)
            
            all_samples += len(logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

    network.eval()
    with torch.no_grad():

        updated_classifiers = network()
        updated_classifiers /= updated_classifiers.norm(dim=-1, keepdim=True)
        
        logits = 100. * test_features @ updated_classifiers.t()
        acc = cls_acc(logits, test_labels)

    print("**** DATViL test accuracy: {:.2f}. ****\n".format(acc))
    
    log_file = cfg['log_file'] 
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cfg['dataset'], str(cfg['shots']), str(alpha), str(cfg['train_epoch']), str(acc)])
                
    torch.save({
        'model_state_dict' : network.state_dict(),
        }, os.path.join(cfg['save_dir'],'network_'+ str(cfg['train_epoch']) + '.pth.tar'))
    return network


def main_datvil(cfg):
    
    save_dir = os.path.join('./checkpoints', cfg['model_name'], cfg['dataset'] , str(cfg['shots'])+ 'shot' + '_' + str(cfg['alpha']) + 'alpha')
    os.makedirs(save_dir, exist_ok=True)
    
    cfg['save_dir'] = save_dir
    
    print("\nRunning configs.")
    print(cfg, "\n")

    
    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model = clip_model.to(torch.float32)   

    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    
    
    print("Preparing dataset.")
    
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    attributes_weights = process_attributes_imagenet(imagenet.classnames, imagenet.attributes, clip_model)
    
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)
    
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader) 

    log_file = os.path.join(cfg['save_dir'],'log.csv')
    if not os.path.isfile(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header if needed
            writer.writerow(['dataset', 'shot', 'alpha', 'iter', 'acc'])
    
    cfg['log_file'] = log_file
    
    datvil_dir = os.path.join(cfg['save_dir'],'network_'+ str(cfg['train_epoch'])+'.pth.tar')
    
    if os.path.isfile(datvil_dir):
        network = DATVIL(attributes_weights, cfg['alpha'], clip_model.dtype, cfg['plus_residual'], cfg['plus_transform']).cuda()  
        network_saved = torch.load(datvil_dir)  
    
        network.load_state_dict(network_saved['model_state_dict']) 
        network.to('cuda')
    else: 
        network = train_datvil(cfg, test_features, test_labels, attributes_weights, clip_model, train_loader)
    
    
    candidate_classes = extract_candidates(cfg, network, test_features, imagenet.classnames)
    
   
    with open(os.path.join(cfg['save_dir'],cfg['dataset'] + '_' + str(cfg['shots']) + "_labels.json"), "w") as outfile:
        json.dump(candidate_classes, outfile, indent = 2)
    
    return


def train_datvilc(cfg, test_logits, test_labels, test_features, clip_model, datvil, few_shot_data_class, attributes_discriminative, topk=5):
    pred = test_logits.topk(topk, 1, True, True)[1]

    epoch_number = cfg['per_sample_train']
    beta = cfg['beta']
    alpha = cfg['alpha']
    
    for param in datvil.parameters():
        param.requires_grad = False
        
    classifiers_datvil = datvil()
    
    correct_sample_counter = 0
    counter_all = 0
    
    samples_num = len(pred)
    
    for i in range(samples_num):
       
        candidates = pred[i]
        samples, labels, labels_names = few_shot_data_class.getitem(candidates.to('cpu').numpy())
        
        
        labels_names_distinct = [item for index, item in enumerate(labels_names) if item not in labels_names[:index]]

        if len(labels_names_distinct) == 1:
            labels_names_distinct.append(labels_names_distinct[0])
        
        label_key1 = labels_names_distinct[0] + ':' + labels_names_distinct[1]
        label_key2 = labels_names_distinct[1] + ':' + labels_names_distinct[0]

        if label_key1 in attributes_discriminative.keys():
            attributes_dicriminative_weights = clip_attributes_discriminative(attributes_discriminative[label_key1], clip_model)
        else:
            try:
                attributes_dicriminative_weights = clip_attributes_discriminative([attributes_discriminative[label_key2][1],attributes_discriminative[label_key2][0]], clip_model)
            except:
                print('Context-aware descriptions for {}:{} not found. Please update the descriptions with the information of the new classes.'.format(label_key1[0], label_key1[1]))
                continue
     
        classifiers_datvil_selected = classifiers_datvil[candidates]
        
        classifiers_datvil_selected = classifiers_datvil_selected / classifiers_datvil_selected.norm(dim = -1, keepdim=True)
            

        datvilc = DATVIL(attributes_dicriminative_weights, alpha, clip_model.dtype, cfg['plus_residual'],cfg['plus_transform']).cuda()
        
        optimizer = torch.optim.AdamW([{'params':datvilc.text_attributes_weights, 'lr':cfg['lr_transformer']},
                                    {'params':datvilc.text_attributes_residuals, 'lr':cfg['lr_residual']}], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_number)
        
        images, target = samples.cuda(), labels.cuda()
            
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        for train_idx in range(epoch_number):
            # Train
            classifiers_datvilc = datvilc()
            classifiers_datvilc = classifiers_datvilc / classifiers_datvilc.norm(dim=-1, keepdim=True)
            
            final_weights = classifiers_datvil_selected + beta * classifiers_datvilc
            
            final_weights = final_weights / final_weights.norm(dim=-1, keepdim=True)
            
            
            logits = 100. * image_features @ final_weights.t()
            
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval
       
        with torch.no_grad():

            classifiers_datvilc = datvilc()
            classifiers_datvilc = classifiers_datvilc / classifiers_datvilc.norm(dim=-1, keepdim=True)
            
            final_weights = classifiers_datvil_selected + beta * classifiers_datvilc
            
            final_weights = final_weights / final_weights.norm(dim=-1, keepdim=True)
            
            
            logits = 100. * test_features[i:i+1] @ final_weights.t()
            
            pred_this = logits.topk(1, 1, True, True)[1].t()
            
            if candidates[pred_this[0].item()].item() == test_labels[i]:
                correct_sample_counter += 1
            counter_all += 1
        print('Sample number {:} / {:} done!'.format(i+1,samples_num))
    return correct_sample_counter/counter_all * 100



def main_datvilc(cfg):    
    save_dir = os.path.join('./checkpoints', cfg['model_name'], cfg['dataset'] , str(cfg['shots'])+ 'shot' + '_' + str(cfg['alpha']) + 'alpha_'+ str(cfg['beta']) + 'beta')
    
    os.makedirs(save_dir, exist_ok=True)
    
    cfg['save_dir'] = save_dir
    
    datvil_dir = os.path.join('./checkpoints', 'datvil', cfg['dataset'] , str(cfg['shots'])+ 'shot' + '_' + str(cfg['alpha']) + 'alpha')
   
    datvil_net_dir = os.path.join(datvil_dir,'network_'+ str(cfg['train_epoch'])+'.pth.tar')
        
    print("\nRunning configs.")
    print(cfg, "\n")

    
    # CLIP
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model = clip_model.to(torch.float32)   

    clip_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    
    
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    attributes_weights = process_attributes_imagenet(imagenet.classnames, imagenet.attributes, clip_model)
    
    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
  

    few_shot_data_class = FewShotDataImagenet(imagenet.train, imagenet.attributes, attributes_weights, imagenet.classnames, clip_model, input_size= 224, transform=train_tranform, is_train=True)
        
    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader) 

      
    assert (os.path.isfile(datvil_net_dir))
 
    datvil = DATVIL(attributes_weights, cfg['alpha'], clip_model.dtype, cfg['plus_residual'], cfg['plus_transform']).cuda()  
    network_saved = torch.load(datvil_net_dir)  

    datvil.load_state_dict(network_saved['model_state_dict']) 
    datvil.to('cuda')

    
    attribute_discriminative_path = os.path.join(cfg['root_path'],'attributes_discriminative_gpt4', cfg['dataset']+'.json')
    
    with open(attribute_discriminative_path, 'r') as f:
        attributes_discriminative = json.load(f)

    with torch.no_grad():  
        classifiers = datvil()
        classifiers /= classifiers.norm(dim=-1, keepdim=True)
        
        logits = 100. * test_features @ classifiers.t()
     
                
    acc_revised = train_datvilc(cfg, logits, test_labels, test_features, clip_model, datvil, few_shot_data_class, attributes_discriminative, topk=2)
    
    log_file = os.path.join(cfg['save_dir'],'log.csv')
    if not os.path.isfile(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['dataset', 'shot', 'alpha','beta', 'iter', 'acc'])
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cfg['dataset'], str(cfg['shots']), str(cfg['alpha']), str(cfg['beta']), str(cfg['per_sample_train']), str(acc_revised)])
    return 
    



def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    cfg['shots'] = args.shots
  
    cfg['root_path'] = args.root_path
    
    cfg['per_sample_train'] = args.per_sample_train
    
    cfg['plus_residual'] = args.plus_residual
    cfg['plus_transform'] = args.plus_transform
    
    cfg['model_name'] = args.model_name
    
    if cfg['model_name'] not in ['datvil', 'datvilc']:
        raise ValueError("Parameter model_name should be set to either datvil or datvilc.")

    if args.shots == 16:
        cfg['train_epoch'] = 150
    else:
        cfg['train_epoch'] = 100
    if cfg['dataset'] == 'imagenet' or cfg['dataset'] == 'food101':
        cfg['train_epoch'] = 10
    
    
    if cfg['model_name'] == 'datvil':
        main_datvil(cfg)
    elif cfg['model_name'] == 'datvilc':
       main_datvilc(cfg)
        

    

  
if __name__ == '__main__':
    main()