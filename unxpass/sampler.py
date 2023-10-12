import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
import tqdm
from sklearn.model_selection import train_test_split
import random
import pickle
import os
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import soccermap
import collections
#train : valid = success : fail이 동일하게 samlping
def SubsetRandomSampler_function(data):
    # train : valid = 0.75 : 0.25로 수행 -> train : valid : test = 6 : 2 : 2로 스플릿됨

    temp = data.labels.reset_index()
    train_temp, valid_temp = train_test_split(temp, test_size=0.25, stratify=temp['success'])
    
    #ascending을 해줘야 value[0] = 0의 개수, value[1] = 1의개수로 맞춰짐
    class_counts_train = train_temp['success'].value_counts(ascending=True).tolist()
    class_counts_valid = valid_temp['success'].value_counts(ascending=True).tolist()
    print(f"subset sampler value counts : train:{class_counts_train} & valid:{class_counts_valid}")
    #train_data => 0 : 1 = fail : success = 3061 : 23532
    #valid_data => 0 : 1 = fail : success = 1021 : 7844
    #인덱스를 파라미터로 주어서 랜덤샘플링 적용으로 데이터의 클래스가 균일할 때 사용

    train_sampler = SubsetRandomSampler(np.array(train_temp.index))
    val_sampler = SubsetRandomSampler(np.array(valid_temp.index))
    
    return train_sampler, val_sampler

#train에서 batch마다 동일한 label비율로 sampling
def WeightedRandomSampler_function(data):
    #성공한 패스 = 31376
    #실패한 패스 = 4082
    #성공 : 실패 = 1.13 : 8.68의 비율로 패스가 Sampling되어야함
    temp = data.labels.reset_index()
    train_temp, valid_temp = train_test_split(temp, test_size=0.25, stratify=temp['success'])
    class_counts = train_temp['success'].value_counts(ascending=True).tolist()
    num_samples = sum(class_counts)
    labels = train_temp['success'].tolist()
 
    class_weights = [num_samples / class_counts[i] for i in tqdm.tqdm(range(len(class_counts)),desc="class_weight")] 
    
    weights = [class_weights[labels[i]] for i in tqdm.tqdm(range(int(num_samples)),desc="weights")]
    
    #replacement = true이므로 복원추출을 수행(중복도 가능함)
    #이유는 모르겠는데, false로 비복원추출하면 추출이 하나의 라벨로만 추출되는 현상이 발생함
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    return train_temp, valid_temp, sampler

def My_Sampler(data):
    loading_image_data(len(data), kind='True')

    #패스가 성공 & 실패한 패스 데이터를 분리
    #loading이 너무 오래걸리므로 따로 저장해놓음
    true_samples = [data_tuple for data_tuple in tqdm.tqdm(data,desc="true sampling") if data_tuple[2].item() == 1.0]
    #false_samples = [data_tuple for data_tuple in tqdm.tqdm(data,desc="false sampling") if data_tuple[2].item() == 0.0]
    print(true_samples)
    print(true_samples[0])
    # random.shuffle(true_samples)
    # random.shuffle(false_samples)

    # #train이 true_smaples, false_samples 상위 75%를 가져가고, valid가 나머지 25%를 가져감
    # #즉, train : valid = 26593 : 8865 = 0.75 : 0.25의 비율로 분리
    # true_ratio = int(0.75 * len(true_samples))
    # false_ratio = int(0.75 * len(false_samples))
    
    # #전체 데이터셋 -> 성공한 패스 : 실패한 패스 = 31,364 : 4,081 = 8 : 1
    # #train 데이터셋-> 성공한 패스 : 실패한 패스 = 23,532 : 3,061 = 8 : 1
    # #valid 데이터셋-> 성공한 패스 : 실패한 패스 = 7,832 :  1,020 = 8 : 1
    # train_data = true_samples[:true_ratio] + false_samples[:false_ratio]
    # valid_data = true_samples[true_ratio:] + false_samples[false_ratio:]

    # random.shuffle(train_data)
    # random.shuffle(valid_data)
    
    # #train데이터에서 실패한 패스의 개수 = 3,061 & 성공한 패스의 개수 = 23,532
    # class_counts = [false_ratio, true_ratio]
    # num_samples = sum(class_counts)
    # class_weights = [num_samples / class_counts[i] for i in tqdm.tqdm(range(len(class_counts)),desc="class_weight")]
    
    # weights = [class_weights[int(data_tuple[2].item())] for data_tuple in tqdm.tqdm(train_data,desc="weights")]

    # train_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    # return train_data, valid_data, train_sampler
    
#king = True / False 데이터 불러오기
#데이터 형식은 = []리슽 ㅎ
def loading_image_data(num_sample, kind):
    channel_tensor = torch.zeros(11,68,104)
    mask_tensor = torch.zeros(1,68,104)
    label_tensor = torch.zeros(1)
    
    #사전에 데이터 형식을 구해놓고 데이터를 저장함(Ram부족)
    data = [[channel_tensor,mask_tensor,label_tensor] for _ in range(num_sample)]
    
    channel_list = ['Sparse_Location_attack','Sparse_Location_defend' ,
                'Dense_Distance_ball', 'Dense_Distance_goal' ,
                'Dense_Cosine__between_ball_goal', 'Dense_Sine_between_ball_goal' ,
                'Dense_Angle_goal', 'Dense_Speed_X', 'Dense_Speed_Y' ,
                'Sprase_Number_attack_players_between_ball_location', 'Sprase_Number_defend_players_between_ball_location']
    
    file_path_mask = f'../Pass Data/Train Data/{kind} Pass Data/Mask/Mask.pkl'
    file_path_label = f'../Pass Data/Train Data/{kind} Pass Data/Label/Label.pkl'
    
    with open(file_path_mask, 'rb') as file:
         Mask = pickle.load(file) 
        
    with open(file_path_label, 'rb') as file:
         Label = pickle.load(file)
         
    print(len(Mask))
    print(len(Label))
    
    for i in tqdm.tqdm(range(num_sample), desc='mask & label loading'):
        data[i][1] = Mask[i]
        data[i][2] = Label[i]
    
    for channel_number, key in tqdm.tqdm(enumerate(channel_list),desc='channel loading'):
        file_path_channel = f'../Pass Data/Train Data/{kind} Pass Data/Channel/' + key +'.pkl'
        with open(file_path_channel,'rb') as file:
            channel_temp = pickle.load(file) 
            
            for i in range(num_sample):
                data[i][0][channel_number] = channel_temp[i]
    
    ss
    
#kind = True / Fasle관련 image data를 저장할건지
def storing_image_data(data, kind):
    #각 channel의 데이터를 저장할 리스트
    Sparse_Location_attack = []
    Sparse_Location_defend = []
    Dense_Distance_ball = []
    Dense_Distance_goal = []
    Dense_Cosine__between_ball_goal = []
    Dense_Sine_between_ball_goal = []
    Dense_Angle_goal = []
    Dense_Speed_X = []
    Dense_Speed_Y = []
    Sprase_Number_attack_players_between_ball_location = []
    Sprase_Number_defend_players_between_ball_location = []
    channel_dict = {'Sparse_Location_attack' : Sparse_Location_attack, 
                    'Sparse_Location_defend' : Sparse_Location_defend, 
                    'Dense_Distance_ball' : Dense_Distance_ball, 
                    'Dense_Distance_goal' : Dense_Distance_goal,
                    'Dense_Cosine__between_ball_goal' : Dense_Cosine__between_ball_goal, 
                    'Dense_Sine_between_ball_goal' : Dense_Sine_between_ball_goal,
                    'Dense_Angle_goal' : Dense_Angle_goal, 
                    'Dense_Speed_X' : Dense_Speed_X, 
                    'Dense_Speed_Y' : Dense_Speed_Y,
                    'Sprase_Number_attack_players_between_ball_location' : Sprase_Number_attack_players_between_ball_location,
                    'Sprase_Number_defend_players_between_ball_location' : Sprase_Number_defend_players_between_ball_location
                    }
    channel_list = list(channel_dict)
    Mask = []
    Label = []
    
    #각 데이터를 부르면 channel, mask, label정보가 추출되는데, 각 변수에 따로 저장
    for i in tqdm.tqdm(range(len(data))):
        channel, mask, label = data[i]
        Mask.append(mask)
        Label.append(label)
        
        for key, value in enumerate(channel):
            channel_dict[channel_list[key]].append(value)

    #파일에 저장
    #True / False 두 곳에 저장
    file_path_mask = f'../Pass Data/Train Data/{kind} Pass Data/Mask/Mask.pkl'
    file_path_label = f'../Pass Data/Train Data/{kind} Pass Data/Label/Label.pkl'
    
    with open(file_path_mask,'wb') as file:
        pickle.dump(Mask, file)
        
    with open(file_path_label,'wb') as file:
        pickle.dump(Label, file)
         
    #True / False 두 곳에 저장
    for key in channel_list:
        file_path_channel = f'../Pass Data/Train Data/{kind} Pass Data/Channel/' + key +'.pkl'

        with open(file_path_channel,'wb') as file:
            pickle.dump(channel_dict[key], file)