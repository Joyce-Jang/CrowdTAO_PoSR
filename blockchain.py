"""
 - Blockchain for Federated Learning -
   Blockchain script 
"""

import hashlib
import json
import time
from flask import Flask,jsonify,request
from uuid import uuid4
from urllib.parse import urlparse
import requests
import random
from threading import Thread, Event
import pickle
import codecs
import data.federated_data_extractor as dataext
import numpy as np
from federatedlearner import *
import sys
import logging

# # 配置日志文件
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(message)s",
#     handlers=[
#         logging.FileHandler("logs/blockchain.log"),
#         logging.StreamHandler(sys.stdout)  # 同时输出到控制台
#     ]
# )
#
# # 重定向 print
# print = logging.info


def compute_global_model(base,updates,lrate,positive=None):

    '''
    Function to compute the global model based on the client 
    updates received per round
    '''

    upd = dict()
    for x in ['w1','w2','wo','b1','b2','bo']:
        upd[x] = np.array(base[x], copy=True)

    # for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
    #         upd[x]=upd[x]+base[x]
    # print("base",base)
    print('positive 1: ', positive)

    if positive is not None:
        number_of_clients = sum(positive)
    else:
        number_of_clients = len(updates)

    i=0
    # for client in updates.keys():
    for idx, (client, update) in enumerate(updates.items()):

        for x in ['w1','w2','wo','b1','b2','bo']:
            model = update.update
            if positive is not None:
                if number_of_clients!=0:
                    upd[x] += int(positive[idx])*(lrate/number_of_clients)*(model[x])

                else:
                    upd[x]=upd[x]
                    print("upd[x]=upd[x]")
            else:
                upd[x] += (lrate / number_of_clients) * (model[x])
                print("positive is None")
        print("client: ", client, "positive[i]: ", positive[idx])
        # i=i+1

    print('positive 2: ', positive)


    upd["size"] = 0
    reset()
    dataset = dataext.load_data("data/mnist.d")
    worker = NNWorker(None,
        None,
        dataset['test_images'],
        dataset['test_labels'],
        0,
        "validation")
    worker.build(upd)
    accuracy = worker.evaluate()
    worker.close()
    return accuracy,upd

def find_len(text,strk):

    ''' 
    Function to find the specified string in the text and return its starting position 
    as well as length/last_index
    '''
    return text.find(strk),len(strk)

class Update:
    def __init__(self,client,baseindex,update,datasize,computing_time,timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters
        '''
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time

    @staticmethod
    def from_string(metadata):

        ''' 
        Function to get the update string values
        '''
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'baseindex':")
        i3,l3 = find_len(metadata,"'update': ")
        i4,l4 = find_len(metadata,"'client':")
        i5,l5 = find_len(metadata,"'datasize':")
        i6,l6 = find_len(metadata,"'computing_time':")
        baseindex = int(metadata[i2+l2:i3].replace(",",'').replace(" ",""))
        update = dict(pickle.loads(codecs.decode(metadata[i3+l3:i4-1].encode(), "base64")))
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        client = metadata[i4+l4:i5].replace(",",'').replace(" ","")
        datasize = int(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        computing_time = float(metadata[i6+l6:].replace(",",'').replace(" ",""))
        return Update(client,baseindex,update,datasize,computing_time,timestamp)


    def __str__(self):

        ''' 
        Function to return the update string values in the required format
        '''
        return "'timestamp': {timestamp},\
            'baseindex': {baseindex},\
            'update': {update},\
            'client': {client},\
            'datasize': {datasize},\
            'computing_time': {computing_time}".format(
                timestamp = self.timestamp,
                baseindex = self.baseindex,
                update = codecs.encode(pickle.dumps(sorted(self.update.items())), "base64").decode(),
                client = self.client,
                datasize = self.datasize,
                computing_time = self.computing_time
            )


class Block:
    def __init__(self,miner,index,basemodel,accuracy,updates,timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters per created block
        '''
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(metadata):

        ''' 
        Function to get the update string values per block
        '''
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'basemodel': ")
        i3,l3 = find_len(metadata,"'index':")
        i4,l4 = find_len(metadata,"'miner':")
        i5,l5 = find_len(metadata,"'accuracy':")
        i6,l6 = find_len(metadata,"'updates':")
        i9,l9 = find_len(metadata,"'updates_size':")
        index = int(metadata[i3+l3:i4].replace(",",'').replace(" ",""))
        miner = metadata[i4+l4:i].replace(",",'').replace(" ","")
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        basemodel = dict(pickle.loads(codecs.decode(metadata[i2+l2:i5-1].encode(), "base64")))
        accuracy = float(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        su = metadata[i6+l6:i9]
        su = su[:su.rfind("]")+1]
        updates = dict()
        for x in json.loads(su):
            isep,lsep = find_len(x,"@|!|@")
            updates[x[:isep]] = Update.from_string(x[isep+lsep:])
        updates_size = int(metadata[i9+l9:].replace(",",'').replace(" ",""))
        return Block(miner,index,basemodel,accuracy,updates,timestamp)

    def __str__(self):

        ''' 
        Function to return the update string values in the required format per block
        '''
        return "'index': {index},\
            'miner': {miner},\
            'timestamp': {timestamp},\
            'basemodel': {basemodel},\
            'accuracy': {accuracy},\
            'updates': {updates},\
            'updates_size': {updates_size}".format(
                index = self.index,
                miner = self.miner,
                basemodel = codecs.encode(pickle.dumps(sorted(self.basemodel.items())), "base64").decode(),
                accuracy = self.accuracy,
                timestamp = self.timestamp,
                updates = str([str(x[0])+"@|!|@"+str(x[1]) for x in sorted(self.updates.items())]),
                updates_size = str(len(self.updates))
            )



class Blockchain(object):
    def __init__(self,miner_id,base_model=None,gen=False,update_limit=7,epoch=10,agent=20, member=5,malicious=1,time_limit=180000):
        super(Blockchain,self).__init__()
        self.miner_id = miner_id  #IP address
        self.curblock = None
        self.hashchain = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit
        self.agent_num=agent
        self.reputation=np.random.uniform(10, 100, self.agent_num)
        self.coin=np.zeros(self.agent_num)
        self.member_num=member
        self.deta_rep=np.zeros(self.member_num)
        self.malicious_num=malicious
        self.committee_indices=None
        self.worker_indices=None
        self.is_malicious=np.zeros(self.agent_num)
        self.malic_done=np.zeros(self.agent_num)
        self.malic_done_changes=[]
        self.rep_changes=[]
        self.com_ind_changes=[]
        self.coin_changes=[]
        self.epoch=0
        self.total_reward_R=0.1 #1
        self.total_reward_C1=1
        self.total_reward_C2 =20*7
        self.threshold=0.1   #0.15
        self.rho=10 #恶意行为下降速度
        self.mali_prob=1
        self.z=0.9#时间衰减
        self.total_epoch=epoch
        self.gamma=50#信誉值系数
        self.finish=False
        self.finished_nodes=[]
        
        if gen:
            genesis,hgenesis = self.make_block(base_model=base_model,previous_hash=1)
            self.store_block(genesis,hgenesis)
            self.rep_changes.append(self.reputation.copy())
            # self.com_ind_changes.append(self.committee_indices)
            self.coin_changes.append(self.coin.copy())
            if self.malicious_num!=0:
                mali_indices = random.sample(range(self.agent_num), self.malicious_num)
                # mali_indices=[0]
                self.is_malicious[mali_indices]=self.mali_prob
                with open("changes/is_malicious.pkl", "wb") as f:
                    pickle.dump(self.is_malicious, f)
            if self.member_num!=0:
                self.allocation()

        self.nodes = set()

    def allocation(self):
        rep = self.reputation
        N = len(rep)
        M = self.member_num
        deta_rep = np.zeros(N)
        np.random.seed(self.epoch+1)
        for i in range(N):
            deta_rep[i] = np.random.uniform(0, rep[i])
        probabilities = deta_rep / np.sum(deta_rep)
        committee_indices = np.random.choice(range(N), size=M, replace=False, p=probabilities)
        non_committee_indices = [i for i in range(N) if i not in committee_indices]
        self.committee_indices = sorted(committee_indices)
        self.worker_indices = sorted(non_committee_indices)
        self.deta_rep = deta_rep[committee_indices]
        print("self.committee_indices:",  self.committee_indices)
        print("self.worker_indices:",self.worker_indices)

    def register_node(self,address):
        if address[:4] != "http":
            address = "http://"+address
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        print("Registered node",address)

    def make_block(self,previous_hash=None,base_model=None,positive=None):
        accuracy = 0
        basemodel = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain)>0:
            update_limit = self.last_block['update_limit']
            time_limit = self.last_block['time_limit']
        if previous_hash==None:
            previous_hash = self.hash(str(sorted(self.last_block.items())))
        if base_model!=None:
            accuracy = base_model['accuracy']
            basemodel = base_model['model']
        elif len(self.current_updates)>0:
            base = self.curblock.basemodel
            accuracy,basemodel = compute_global_model(base,self.current_updates,1,positive)
        index = len(self.hashchain)+1
        block = Block(
            miner = self.miner_id,
            index = index,
            basemodel = basemodel,
            accuracy = accuracy,
            updates = self.current_updates
            )
        hashblock = {
            'index':index,
            'hash': self.hash(str(block)),
            'proof': random.randint(0,100000000),
            'previous_hash': previous_hash,
            'miner': self.miner_id,
            'accuracy': str(accuracy),
            'timestamp': time.time(),
            'time_limit': time_limit,
            'update_limit': update_limit,
            'model_hash': self.hash(codecs.encode(pickle.dumps(sorted(block.basemodel.items())), "base64").decode())
            }
        return block,hashblock

    def store_block(self,block,hashblock):
        if self.curblock:
            with open("blocks/federated_model"+str(self.curblock.index)+".block","wb") as f:
                pickle.dump(self.curblock,f)
        self.curblock = block
        self.hashchain.append(hashblock)
        self.current_updates = dict()
        return hashblock

    def store_information(self):
        with open("changes/reputation.pkl", "wb") as f:
            pickle.dump(self.rep_changes, f)
        with open("changes/coin.pkl", "wb") as f:
            pickle.dump(self.coin_changes, f)
        with open("changes/committee.pkl", "wb") as f:
            pickle.dump(self.com_ind_changes, f)
        if self.malicious_num != 0:
            with open("changes/malic_changes.pkl", "wb") as f:
                pickle.dump(self.malic_done_changes, f)



    def new_update(self,client,baseindex,update,datasize,computing_time):
        self.current_updates[client] = Update(
            client = client,
            baseindex = baseindex,
            update = update,
            datasize = datasize,
            computing_time = computing_time
            )
        return self.last_block['index']+1

    @staticmethod
    def hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def last_block(self):
        return self.hashchain[-1]

    @staticmethod
    def compute_loss_and_accuarcy(test_images, test_labels,model):
        worker = NNWorker(None,
                          None,
                          test_images,
                          test_labels,
                          0,
                          "validation")
        worker.build(model)

        accuracy = worker.evaluate()
        # loss=worker.compute_loss()
        loss=-1
        worker.close()
        return accuracy, loss

    def update_committee_reward(self,median,accur,length):
        all_consist=[]
        deta_rep=np.zeros(self.agent_num)
        for i in range(self.update_limit):
            consist = []
            for j in range(length):
                if (median[i] >= self.threshold and accur[j][i] >= self.threshold) or (median[i] < self.threshold and accur[j][i] < self.threshold):
                    consist.append(True)
                else:
                    consist.append(False)
            for k in range(length):
                    if consist[k]:  # 评分一致
                        reward_ratio = self.deta_rep[k] / np.sum(self.deta_rep[consist])
                        # self.reputation[self.committee_indices[k]] += reward_ratio * self.total_reward_R
                        deta_rep[self.committee_indices[k]]+=reward_ratio * self.total_reward_R
                        self.coin[self.committee_indices[k]] += reward_ratio * self.total_reward_C1
                    else:  # 评分不一致
                        # self.reputation[self.committee_indices[k]] -= self.deta_rep[k]
                        deta_rep[self.committee_indices[k]] -= self.deta_rep[k]
                        # self.reputation[self.committee_indices[k]] = max(self.reputation[self.committee_indices[k]], 0)  # 确保不为负
            self.reputation=self.reputation+deta_rep
            self.reputation[self.reputation<0]=0 # 确保不为负
            all_consist.append(consist)
        return all_consist

    def update_worker_reward(self,median):
        alpha=0.5
        positive=np.zeros(self.update_limit)
        # median=median
        reward_ratio=np.zeros_like(median)
        negative=np.zeros(self.update_limit)

        for i in range(self.update_limit):
            positive[i]=(median[i] >= self.threshold)
            negative[i]=(median[i] < self.threshold)

        positive=[True if x == 1 else False for x in list(positive)]
        negative=[True if x == 1 else False for x in list(negative)]

        print("positive",positive)
        print("negative",negative)


        for i in range(self.update_limit):
            if positive[i]==True:
                reward_ratio[i] = (median[i]-self.threshold) / np.sum(median[positive]-self.threshold)
                self.coin[self.worker_indices[i]] += reward_ratio[i] * self.total_reward_C2
                # print("self.worker_indices[i]",self.worker_indices[i],"positive",positive[i])
            else:
                reward_ratio[i] = (median[i]-self.threshold) / np.sum(median[negative]-self.threshold)
                # print("self.worker_indices[i]", self.worker_indices[i], "positive-Ne", positive[i])



        for i in range(self.update_limit):
            worker_id=self.worker_indices[i]
            x_i =abs(median[i] - self.threshold) * 10
            y_i = int(median[i]< self.threshold) * (self.threshold- median[i])*10
            u_i = 0  # 不确定值固定为0
            b_i_t = (1 - u_i) * x_i/ (x_i + np.exp(self.rho * y_i))
            d_i_t = (1 - u_i) * np.exp(self.rho * y_i) / (x_i + np.exp(self.rho * y_i))
            e_i_t = (b_i_t + alpha * u_i)*self.gamma
            self.reputation[worker_id] = self.rep_changes[-1][worker_id]  * self.z + e_i_t

            print("worker_id: ", worker_id, "x_i: ", x_i, "y_i: ", y_i)
            # if self.is_malicious[worker_id]!=0:
            #     print("worker_id: ", worker_id, "x_i: ",x_i,"y_i: ", y_i)
            #     print("worker_id: ", worker_id, "median[i] ", median[i], "self.threshold ",self.threshold)
            #     print("int(median[i]< self.threshold)",int(median[i]< self.threshold))
            #     print("(self.threshold- median[i])*10",(self.threshold- median[i])*10)


        return positive

    def proof_of_work(self,stop_event):
        print("Enter Proof_of_work")
        print("self.update_limit",self.update_limit)

        dev_idx = self.committee_indices

        dev_sets = []
        for i in dev_idx:
            # with open('data/federated_data_{}.d'.format(i), 'rb') as f:
            #     print("f", 'data/federated_data_{}.d'.format(i))
            #     dev_sets.append(pickle.load(f))
            dev_sets.append(dataext.load_data('data/federated_data_{}.d'.format(i)))


        base_accur = np.zeros((len(dev_idx), self.update_limit))
        # base_losses = np.zeros((len(dev_idx), self.update_limit))
        accur=np.zeros((len(dev_idx),self.update_limit))
        # losses=np.zeros((len(dev_idx),self.update_limit))
        # deta_accur = np.zeros((len(dev_idx), self.update_limit))
        # deta_losses = np.zeros((len(dev_idx), self.update_limit))


        # print('self.current_updates', self.current_updates.items())

        # upd = dict()
        # for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
        #     upd[x] = np.array(self.curblock.basemodel[x], copy=True)

        self.current_updates= dict(sorted(self.current_updates.items(), key=lambda x: int(x[0])))




        for k in range(len(dev_sets)):

            dev_images = np.concatenate((dev_sets[k]['train_images'], dev_sets[k]['test_images']))
            dev_labels = np.concatenate((dev_sets[k]['train_labels'], dev_sets[k]['test_labels']))
            j=0

            for client, values in self.current_updates.items():

                if self.is_malicious[dev_idx[k]]==0 or self.malic_done[int(client)]==0:

                    base_accuracy, base_loss = self.compute_loss_and_accuarcy(dev_images, dev_labels,self.curblock.basemodel)
                    base_accur[k][j] = base_accuracy
                    # base_losses[k][j] = base_loss

                    reset()
                    # for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:
                    #     upd[x] = self.curblock.basemodel[x] + values.update[x]
                    accuracy, loss=self.compute_loss_and_accuarcy(dev_images, dev_labels,values.update)
                    accur[k][j]=accuracy
                    # losses[k][j]=loss
                    # print("client: ",client,"accuracy:", accuracy)
                else:
                    accur[k][j]=-1
                j=j+1

        coordinates = np.argwhere(accur == -1)
        if coordinates.size > 0:
            for coord in coordinates:
                max=np.max(accur[coord[0]])
                filter_idx=(accur[coord[0]]!=-1)
                accur[coord[0],:]=np.ones(len(accur[coord[0]]))*np.min(accur[coord[0]][filter_idx])
                accur[coord[0]][coord[1]] = max
                #

        deta_accur=accur-base_accur
        committee_median = np.median(deta_accur, axis=0)

        self.update_committee_reward(committee_median, deta_accur,len(dev_sets))

        positive=self.update_worker_reward(committee_median)

        self.rep_changes.append(self.reputation.copy())
        self.com_ind_changes.append(self.committee_indices)
        self.coin_changes.append(self.coin.copy())

        malic_idx=np.nonzero(self.is_malicious)

        # print("deta_accur",deta_accur)
        print("self.current_updates",self.current_updates)
        print("self.committee_indices", self.committee_indices)
        print("self.worker_indices", self.worker_indices)
        print("self.is_malicious", self.is_malicious)
        print("malic_idx",  malic_idx)
        print("self.malic_done: ", self.malic_done)
        # print("accur", accur)
        # print("base_accur", base_accur)


        print("reputation", self.reputation)
        # print("rep_changes", self.rep_changes)
        print("Coin", self.coin)
        # print("coin_changes", self.coin_changes)
        print("committee_median ", self.committee_indices)
        # print("com_ind_changes ", self.com_ind_changes)
        print("positive 3",positive)


        block, hblock = self.make_block(positive=positive)
        self.store_block(block, hblock)


        print("epoch ", self.epoch, "threshold ", self.threshold)

        if self.malicious_num!=0:
            self.malic_done_changes.append(self.malic_done)


        self.malic_done=self.malic_done*0
        self.threshold=-0.1



        if  self.total_epoch==self.epoch:
            self.store_information()



        self.finish=True

        if self.member_num != 0:
            self.allocation()




        print("Done")
        # block,hblock = self.make_block()
        stopped = False
        # while self.valid_proof(str(sorted(hblock.items()))) is False:
        #     if stop_event.is_set():
        #         stopped = True
        #         break
        #     hblock['proof'] += 1
        #     if hblock['proof']%1000==0:
        #         print("mining",hblock['proof'])
        # if stopped==False:
        #     self.store_block(block,hblock)
        # if stopped:
        #     print("Stopped")
        # else:
        #     print("Done")

        return hblock,stopped

    @staticmethod
    def valid_proof(block_data):
        guess_hash = hashlib.sha256(block_data.encode()).hexdigest()
        k = "00000"
        return guess_hash[:len(k)] == k


    def valid_chain(self,hchain):
        last_block = hchain[0]
        curren_index = 1
        while curren_index<len(hchain):
            hblock = hchain[curren_index]
            if hblock['previous_hash'] != self.hash(str(sorted(last_block.items()))):
                print("prev_hash diverso",curren_index)
                return False
            if not self.valid_proof(str(sorted(hblock.items()))):
                print("invalid proof",curren_index)
                return False
            last_block = hblock
            curren_index += 1
        return True

    def resolve_conflicts(self,stop_event):
        neighbours = self.nodes
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)
        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length>max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node
        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            resp = requests.post('http://{node}/block'.format(node=bnode),
                json={'hblock': hblock})
            self.current_updates = dict()
            if resp.status_code == 200:
                if resp.json()['valid']:
                    self.curblock = Block.from_string(resp.json()['block'])
            return True
        return False
