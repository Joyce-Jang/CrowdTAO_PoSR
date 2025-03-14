"""
  - Blockchain for Federated Learning - 
    Client script 
"""
import tensorflow as tf
import pickle
from federatedlearner import *
from blockchain import *
from uuid import uuid4
import requests
import data.federated_data_extractor as dataext
import time
import numpy as np


class Client:
    def __init__(self,miner,dataset):
        # self.id = str(uuid4()).replace('-','')
        self.id =dataset.split('_')[2].split('.')[0]
        self.miner = miner
        self.dataset = self.load_dataset(dataset)

    def get_last_block(self):
            return self.get_chain()[-1]

    def get_chain(self):
        response = requests.get('http://{node}/chain'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()['chain']

    def get_full_block(self,hblock):
        response = requests.post('http://{node}/block'.format(node=self.miner),
            json={'hblock': hblock})
        if response.json()['valid']:
            return Block.from_string(response.json()['block'])
        print("Invalid block!")
        return None

    def get_model(self,hblock):
        response = requests.post('http://{node}/model'.format(node=self.miner),
            json={'hblock': hblock})
        if response.json()['valid']:
            return dict(pickle.loads(codecs.decode(response.json()['model'].encode(), "base64")))
        print("Invalid model!")
        return None

    def get_miner_status(self):
        response = requests.get('http://{node}/status'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()

    def load_dataset(self,name):

        ''' 
        Function to load federated data for client side training
        '''
        if name==None:
            return None
        return dataext.load_data(name)

    def update_model(self,model,steps):

        ''' 
        Function to compute the client model update based on 
        client side training
        '''
        # reset()
        t = time.time()
        worker = NNWorker(self.dataset['train_images'],
            self.dataset['train_labels'],
            self.dataset['test_images'],
            self.dataset['test_labels'],
            len(self.dataset['train_images']),
            self.id,
            steps)
        worker.build(model)
        worker.train()
        update = worker.get_model()
        accuracy = worker.evaluate()
        worker.close()
        return update,accuracy,time.time()-t

    def send_update(self,update,cmp_time,baseindex):

        ''' 
        Function to post client update details to blockchain
        '''
        # wait=True
        # while wait:
        response=requests.post('http://{node}/transactions/new'.format(node=self.miner),
                json = {
                    'client': self.id,
                    'baseindex': baseindex,
                    'update': codecs.encode(pickle.dumps(sorted(update.items())), "base64").decode(),
                    'datasize': len(self.dataset['train_images']),
                    'computing_time': cmp_time
                })
            # if response.status_code == 201:
            #      wait=False
            # else:
            #      time.sleep(10)
        # print(response)
       
def work(miner,elimit,data,limits):
    '''
    Function to check the status of mining and wait accordingly to output
    the final accuracy values
    '''
    last_model = -1
        
    for i in range(elimit):

        # send_epoch(i,miner)
        # worker_index = get_dataset_from_miner(miner)
        # print("Epoch "+str(i)+" work_index", worker_index)
        #
        # data_idx=worker_index[data]
        # dataset = 'data/federated_data_{}.d'.format(data_idx)
        # client = Client(args.miner, dataset)
        # print('dataset: ', dataset)
        #
        # dataext.get_dataset_details(client.dataset)
        # device_id = client.id[:2]
        # print(device_id, "device_id")
        #
        # is_malicious=get_malicious_from_miner(miner)
        # print("is_malicious",is_malicious)
        # malic_done=[0] * len(is_malicious)
        #
        # wait = True
        # while wait:
        #     status = client.get_miner_status()
        #     if status['status']!="receiving" or last_model==status['last_model_index']:
        #         time.sleep(10)
        #         print("waiting")
        #     else:
        #         wait = False

        wait = True
        while wait:
            status = get_miner_status(miner)
            if status['status'] != "receiving" or last_model == status['last_model_index']:
                time.sleep(10)
                print("waiting")
            else:
                wait = False

        send_epoch(i,miner)
        worker_index = get_dataset_from_miner(miner)
        print("Epoch "+str(i)+" work_index", worker_index)

        data_idx=worker_index[data]
        dataset = 'data/federated_data_{}.d'.format(data_idx)
        client = Client(args.miner, dataset)
        print('dataset: ', dataset)

        # send_finish(client.id, miner) #发送已经准备好了

        dataext.get_dataset_details(client.dataset)
        device_id = client.id[:2]
        print(device_id, "device_id")

        is_malicious=get_malicious_from_miner(miner)
        print("is_malicious",is_malicious)
        malic_done=[0] * len(is_malicious)


        hblock = client.get_last_block()
        baseindex = hblock['index']
        print("Accuracy global model",hblock['accuracy'])

        last_model = baseindex
        model = client.get_model(hblock)
        update,accuracy,cmp_time = client.update_model(model,10)

        with open("clients/device"+str(device_id)+"_model_v"+str(i)+".block","wb") as f:
            pickle.dump(update,f)		#j = j+1
        print("Epoch "+str(i)+"  Accuracy local update---------"+str(device_id)+"--------------:",accuracy)

        if is_malicious[data_idx] > np.random.rand():
            print("Enter malicious behavior ")
            for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:                # update[x] = np.random.normal(0, 1, update[x].shape).astype(np.float32)  # value * -Intensity
                update[x]=update[x]*(-2.1)
            malic_done[data_idx] = 1
            send_malic(malic_done, miner)

        # else:
        #     print("No malicious behavior")
        #     for x in ['w1', 'w2', 'wo', 'b1', 'b2', 'bo']:                # update[x] = np.random.normal(0, 1, update[x].shape).astype(np.float32)  # value * -Intensity
        #         update[x]=update[x]
        #     malic_done[data_idx] = 0
            # print("No change")

        # start_time=time.time()
        accuracy1,_=compute_loss_and_accuarcy( client.dataset['test_images'],client.dataset['test_labels'],update)
        # end_time=time.time()

        # print("test time", end_time-start_time)

        print("Epoch " + str(i) + "  Actual Accuracy local update---------" + str(device_id) + "--------------:", accuracy1)
        # time.sleep(np.random.uniform(0.1,0.5))
        client.send_update(update,cmp_time,baseindex)

        time.sleep(30)


        #
        # no_finish = True
        # while no_finish:
        #     status = get_miner_finish(miner)  #(data, miner)
        #     if status['status']!=True:
        #         time.sleep(10)
        #         print('waiting')
        #     else:
        #         no_finish= False
        #         send_finish(client.id, miner)




def get_miner_status(miner):
    response = requests.get('http://{node}/status'.format(node=miner))
    if response.status_code == 200:
        return response.json()

def get_miner_finish(miner):
    response = requests.get('http://{node}/finish'.format(node=miner))
    if response.status_code == 200:
        return response.json()

def send_finish(id,miner):
    requests.post('http://{node}/finish/send'.format(node=miner),
                  json={
                      'id':int(id),
                  })
    # return response.json()


def send_epoch(i,miner):
    requests.post('http://{node}/epoch/receive'.format(node=miner),
                  json={
                      'epoch': i + 1,
                  })

def get_dataset_from_miner(miner):
    response = requests.get('http://{node}/allocation'.format(node=miner))
    if response.status_code == 200:
        return response.json()

def get_malicious_from_miner(miner):
    response = requests.get('http://{node}/malicious/is'.format(node=miner))
    if response.status_code == 200:
        return response.json()

def send_malic(malic_done,miner):
    requests.post('http://{node}/malic/receive'.format(node=miner),
                  json={
                      'malic_done': malic_done,
                  })
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
        return accuracy, loss


if __name__ == '__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--miner', default='127.0.0.1:5001', help='Address of miner')
    parser.add_argument('-d', '--dataset', default=0,type=int, help='number')
    parser.add_argument('-e', '--epochs', default=5,type=int, help='Number of epochs')
    args = parser.parse_args()

    # worker_index=get_dataset_from_miner(args.miner)
    # print("work_index", worker_index)
    # dataset='data/federated_data_{}.d'.format(args.dataset)
    # client = Client(args.miner,dataset)
    # print("dataset",dataset)
    # print("--------------")
    # print(client.id," Dataset info:")
    # print("client.dataset",type(client.dataset))
    # # Data_size, Number_of_classes = dataext.get_dataset_details(client.dataset)
    # dataext.get_dataset_details(client.dataset)
    # print("--------------")
    # device_id = client.id[:2]
    # print(device_id,"device_id")
    print("--------------")
    #攻击强度
    limits=15
    work(args.miner, args.epochs, args.dataset,limits)

