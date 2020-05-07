from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.centerline_utils import get_nt_distance,get_oracle_from_candidate_centerlines,get_xy_from_nt_seq
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import math
import numpy as np
from random import shuffle
import os
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

''' Need better data representation as compared to just the trajectories.
Feed ground truth of major trajectories point.
Center lines.
'''
def collate_traj_lanecentre(list_data):
    train_agent=[]
    gt_agent=[]
    centerline=[]
    dict_collate={}
    dict_input=list_data[0]
    for key in dict_input.keys():
        v=[]
        # print("SOlving key", key)
        for data in list_data:
            # print(key,data[key].shape)
            v.append(data[key])
        if (key is 'centerline') or (key is 'city') or(key is 'seq_path'):
            dict_collate[key]=v
        elif key is 'seq_index':
            dict_collate[key]=torch.Tensor(v)
        else:
            dict_collate[key]=torch.stack(v,dim=0)
    return dict_collate
    # return {'train_agent': torch.stack(train_agent,dim=0),'gt_agent': torch.stack(gt_agent) , 'neighbour':neighbour} 

def collate_traj_social(list_data):
    train_agent=[]
    gt_agent=[]
    neighbour=[]
    neighbour_gt=[]
    av=[]
    av_gt=[]
    seq_path=[]
    idxs=[]
    city_names = []
    R = []
    T = []
    for data in list_data:
        train_agent.append(data['train_agent'])
        gt_agent.append(data['gt_agent'])
        neighbour.append(data['neighbour'])
        neighbour_gt.append(data['neighbour_gt'])
        av.append(data['av'])
        av_gt.append(data['av_gt'])
        seq_path.append(data['seq_path'])
        idxs.append(data['indexes'])
        city_names.append(data['city_name'])
        R.append(data['rotation'])
        T.append(data['translation'])
    return {'city_name': city_names,'indexes': idxs, 'seq_path': seq_path, 'train_agent': torch.stack(train_agent,dim=0),'gt_agent': torch.stack(gt_agent), 'neighbour':neighbour, 'neighbour_gt':neighbour_gt, 'av':torch.stack(av,dim=0), 'av_gt':torch.stack(av_gt), 'rotation': R, 'translation': T} 

def collate_traj_social_test(list_data):
    seq_index=[]
    train_agent=[]
    neighbour=[]
    for data in list_data:
        train_agent.append(data['train_agent'])
        neighbour.append(data['neighbour'])
        seq_index.append(data['seq_index'])
    return {'seq_index': torch.stack(seq_index,dim=0), 'train_agent': torch.stack(train_agent,dim=0) , 'neighbour':neighbour} 
    

class Argoverse_Data(Dataset):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data',train_seq_size=20,cuda=False,test=False):
        super(Argoverse_Data,self).__init__()
        self.root_dir=root_dir
        self.afl = ArgoverseForecastingLoader(self.root_dir)
        self.seq_paths=glob.glob(f"{self.root_dir}/*.csv")
        self.train_seq_size=train_seq_size
        self.use_cuda=cuda
        self.mode_test=test

    def __len__(self):
        return len(self.seq_paths)

    def old_transform(self,trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        translation=trajectory[0]
        trajectory=trajectory-trajectory[0]
        theta=rotation_angle(trajectory[19,0],trajectory[19,1])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s], [s, c]])
        trajectory=torch.tensor(trajectory)
        trajectory=trajectory.permute(1,0)
        trajectory=np.matmul(R,trajectory)
        trajectory=torch.tensor(trajectory)
        trajectory=trajectory.permute(1,0)
        if self.mode_test:
            return trajectory[0:self.train_seq_size].float(),R,translation
        else:
            return trajectory[0:self.train_seq_size].float(),trajectory[self.train_seq_size:].float()

    def transform(self,trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        
        if self.mode_test:
            translation=-trajectory[0]
            train_trajectory=trajectory+translation
            theta=rotation_angle(train_trajectory[19,0],train_trajectory[19,1])
            c, s = np.cos(theta), np.sin(theta)
            R = torch.Tensor([[c,-s], [s, c]]).float()
            train_trajectory=torch.tensor(train_trajectory).float()
            train_trajectory=torch.matmul(R,train_trajectory.permute(1,0)).permute(1,0)

            return train_trajectory,R,torch.Tensor(translation).float()
        else:
            old_trajectory=trajectory
            translation=-trajectory[0]
            transformed_trajectory=trajectory+translation
            theta=rotation_angle(transformed_trajectory[19,0],transformed_trajectory[19,1])
            c, s = np.cos(theta), np.sin(theta)
            R = torch.Tensor([[c,-s], [s, c]]).float()
            transformed_trajectory=torch.tensor(transformed_trajectory).float()
            transformed_trajectory=torch.matmul(R,transformed_trajectory.permute(1,0)).permute(1,0)
            train_trajectory=transformed_trajectory[:self.train_seq_size]
            gt_transformed_trajectory=transformed_trajectory[self.train_seq_size:]
            actual_gt_trajectory=torch.Tensor(trajectory[self.train_seq_size:]).float()
            return train_trajectory,gt_transformed_trajectory,actual_gt_trajectory,R,torch.Tensor(translation).float()


    def inverse_transform_one(self,trajectory,R,t):
        out=torch.matmul(R,trajectory.permute(1,0)).permute(1,0)
        return out+ t.reshape(1,2)


    def inverse_transform(self,trajectory,traj_dict):
        R=traj_dict['rotation']
        t=traj_dict['translation']
        if self.use_cuda:
            R=R.cuda()
            t=t.cuda()
        out=torch.matmul(R.permute(0,2,1),trajectory.permute(0,2,1)).permute(0,2,1)
        out= out - t.reshape(t.shape[0],1,2)
        return out


    def __getitem__(self,index):
        '''
        Obtain neighbour trajectories as well.
        Obtain map parameters at the trajectories
        Do it in the coordinates of the centerlines as well
        '''

        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        
#         if self.test:
#             agent_train_traj,R,translation=self.transform(agent_traj)
#             seq_index=int(os.path.basename(self.seq_paths[index]).split('.')[0])
#             return {'seq_index': seq_index,'train_agent':agent_train_traj,'rotation':R,'translation':translation,'city':current_loader.city}
#         else:
        agent_train_traj,agent_gt_traj,agent_unnorm_gt_traj,R,translation=self.transform(agent_traj)
        return {'seq_path':self.seq_paths[index], 'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj, 'gt_unnorm_agent':agent_unnorm_gt_traj, 'rotation':R,'translation':translation,'city':current_loader.city}


class Argoverse_Social_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data/forecasting_sample/data', train_seq_size=20, agent_rel=True, cuda=False, test=False):
        super(Argoverse_Social_Data,self).__init__(root_dir,train_seq_size,cuda,test)
        self.agent_rel=agent_rel

    def transform_social(self, agent_trajectory, neighbour_trajectories, av_trajectory):
        def rotation_angle(x,y):
            angle=np.arctan(abs(y/x))
            direction= -1* np.sign(x*y)
            return direction*angle
        
        trajectory_mean = agent_trajectory[0]
        trajectory_rotation=rotation_angle(agent_trajectory[19,0],agent_trajectory[19,1])
        c, s = np.cos(trajectory_rotation), np.sin(trajectory_rotation)
        R = np.array([[c,-s], [s, c]])

        agent_trajectory = agent_trajectory-trajectory_mean
        agent_trajectory = np.swapaxes(agent_trajectory, 1, 0)
        agent_trajectory = np.matmul(R,agent_trajectory)
        agent_trajectory = np.swapaxes(agent_trajectory, 1, 0)
        agent_trajectory = agent_trajectory.astype(float)
        agent_trajectory = torch.tensor(agent_trajectory).float()
        
        normalized_neighbour_trajectories=[]
        
        for neighbour_trajectory in neighbour_trajectories:
            trajectory=neighbour_trajectory
            trajectory=trajectory-trajectory_mean
            trajectory=np.swapaxes(trajectory, 1, 0)
            trajectory=np.matmul(R,trajectory)
            trajectory=np.swapaxes(trajectory, 1, 0)
            trajectory = torch.tensor(trajectory).float()
            
            if self.agent_rel:
                trajectory = trajectory - agent_trajectory
            normalized_neighbour_trajectories.append(trajectory)
         
        av_trajectory = av_trajectory-trajectory_mean
        av_trajectory = np.swapaxes(av_trajectory, 1, 0)
        av_trajectory = np.matmul(R,av_trajectory)
        av_trajectory = np.swapaxes(av_trajectory, 1, 0)
        av_trajectory = av_trajectory.astype(float)
        av_trajectory = torch.tensor(av_trajectory).float()
            
        if self.agent_rel:
            normalized_av_trajectory = av_trajectory - agent_trajectory
            
        if len(normalized_neighbour_trajectories)!= 0:
            normalized_neighbour_trajectories= torch.stack(normalized_neighbour_trajectories,dim=0)
        else:
            normalized_neighbour_trajectories=torch.zeros(1,50,2)
            
        if (normalized_av_trajectory.shape[0])== 0:
            normalized_av_trajectory = torch.zeros(1,50,2)    
        
        return agent_trajectory[0:self.train_seq_size], agent_trajectory[self.train_seq_size:], normalized_neighbour_trajectories[:,0:self.train_seq_size], normalized_neighbour_trajectories[:,self.train_seq_size:], normalized_av_trajectory[0:self.train_seq_size], normalized_av_trajectory[self.train_seq_size:], R, trajectory_mean

    def __getitem__(self,index):
        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj = current_loader.agent_traj
        neighbours_traj = current_loader.neighbour_traj()
        city = current_loader.city
        av_traj = current_loader.av_traj
        
        agent_train_traj,agent_gt_traj,neighbours_traj,neighbours_gt_traj, av_traj_, av_gt_traj, R, t =self.transform_social(agent_traj,neighbours_traj, av_traj)

        return {'city_name':city , 'indexes':index, 'seq_path':self.seq_paths[index],'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj, 'neighbour':neighbours_traj,'neighbour_gt':neighbours_gt_traj, 'av':av_traj_, 'av_gt':av_gt_traj, 'rotation': R, 'translation': t}

class Argoverse_LaneCentre_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data//data',avm=None,social=False,train_seq_size=20,cuda=False,test=False,oracle=True):
        super(Argoverse_LaneCentre_Data,self).__init__(root_dir,train_seq_size,cuda,test)
        if avm is None:
            self.avm=ArgoverseMap()
        else:
            self.avm=avm
        self.stationary_threshold=2.0
        self.oracle=oracle
        print("Done loading map")
    # def __len__(self):
    #     # return 10000
    #     return len(self.seq_paths)
    def inverse_transform(self,trajectory,traj_dict):
        centerline=traj_dict['centerline']
        if self.use_cuda:
            trajectory=trajectory.cpu()
        out=get_xy_from_nt_seq(nt_seq=trajectory,centerlines=centerline)
        out=torch.Tensor(out).float().clone().detach()
        if self.use_cuda:
            out=out.cuda()
        return out
        # pass

    def __getitem__(self,index):
        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        candidate_centerlines = self.avm.get_candidate_centerlines_for_traj(agent_traj, current_loader.city,viz=False)
        # if self.oracle:
        current_centerline=get_oracle_from_candidate_centerlines(candidate_centerlines,agent_traj)
        # else:
            # current_centerline=candidate_centerlines
        if self.mode_test:
            seq_index=int(os.path.basename(self.seq_paths[index]).split('.')[0])
            
            agent_train_traj=agent_traj[:self.train_seq_size,:]
            agent_train_traj=get_nt_distance(agent_train_traj,current_centerline)
            agent_train_traj=torch.Tensor(agent_train_traj).float()
            # gt_agent=self.get_coordinate_from_centerline(oracle_centerline,agent_train_traj)
            return {'seq_index': seq_index,'train_agent':agent_train_traj,'centerline':current_centerline,'city':current_loader.city}

        else:
            agent_train_traj=agent_traj[:self.train_seq_size,:]
            agent_train_traj=get_nt_distance(agent_train_traj,current_centerline)
            agent_train_traj=torch.Tensor(agent_train_traj).float()

            agent_gt_traj=agent_traj[self.train_seq_size:,]
            agent_gt_traj=get_nt_distance(agent_gt_traj,current_centerline)
            agent_gt_traj=torch.Tensor(agent_gt_traj).float()

            agent_unnorm_gt_traj=torch.Tensor(agent_traj[self.train_seq_size:,]).float()

            return {'seq_path':self.seq_paths[index],'train_agent':agent_train_traj, 'gt_agent':agent_gt_traj,'gt_unnorm_agent':agent_unnorm_gt_traj,'centerline':current_centerline,'city':current_loader.city}


class Argoverse_MultiLaneCentre_Data(Argoverse_Data):
    def __init__(self,root_dir='argoverse-data//data',avm=None,social=False,train_seq_size=20,cuda=False,test=False,oracle=False):
        super(Argoverse_LaneCentre_Data,self).__init__(root_dir,train_seq_size,cuda,test)
        if avm is None:
            self.avm=ArgoverseMap()
        else:
            self.avm=avm
        self.stationary_threshold=2.0
        self.oracle=oracle
        print("Done loading map")
    
    def __len__(self):
        # return 10000
        return len(self.seq_paths)
    def inverse_transform(self,trajectory,traj_dict):
        centerline=traj_dict['centerline']
        if self.use_cuda:
            trajectory=trajectory.cpu()
        out=get_xy_from_nt_seq(nt_seq=trajectory,centerlines=centerline)
        out=torch.Tensor(out).float()
        if self.use_cuda:
            out=out.cuda()
        return out

    def __getitem__(self,index):
        current_loader = self.afl.get(self.seq_paths[index])
        agent_traj=current_loader.agent_traj
        candidate_centerlines = self.avm.get_candidate_centerlines_for_traj(agent_traj, current_loader.city,viz=False)
        if self.oracle:
            candidate_centerlines=[get_oracle_from_candidate_centerlines(candidate_centerlines,agent_traj)]
        if self.mode_test:
            seq_index=int(os.path.basename(self.seq_paths[index]).split('.')[0])
            
            agent_train_traj=agent_traj[:self.train_seq_size,:]
            all_centerline_traj=[]
            for centerline in candidate_centerlines:
                all_centerline_traj.append(torch.Tensor(get_nt_distance(agent_train_traj,current_centerline)).float())
            
            return {'seq_index': seq_index,'train_agent':all_centerline_traj,'centerline':candidate_centerlines,'city':current_loader.city}

        else:
            agent_train_traj=agent_traj[:self.train_seq_size,:]
            agent_gt_traj=agent_traj[self.train_seq_size:,]
            all_centerline_train_traj=[]
            all_centerline_gt_traj=[]
            for centerline in candidate_centerlines:
                all_centerline_train_traj.append(torch.Tensor(get_nt_distance(agent_train_traj,current_centerline)).float())
                all_centerline_gt_traj.append(torch.Tensor(get_nt_distance(agent_gt_traj,current_centerline)).float())
            
            agent_unnorm_gt_traj=torch.Tensor(agent_traj[self.train_seq_size:,]).float()

            return {'train_agent':all_centerline_train_traj, 'gt_agent':all_centerline_gt_traj,'gt_unnorm_agent':agent_unnorm_gt_traj,'centerline':current_centerline,'city':current_loader.city}
