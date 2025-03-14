"""
 - Blockchain for Federated Learning -
           Mining script 
"""

import hashlib
import json
import time
from flask import Flask,jsonify,request
from uuid import uuid4
import requests
import random
import pickle
from blockchain import *
from threading import Thread, Event
from federatedlearner import *
import numpy as np
import codecs
import os
import glob
import sys
import logging

# 配置日志文件
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(message)s",
#     handlers=[
#         logging.FileHandler("logs/miner.log"),
#         logging.StreamHandler(sys.stdout)  # 同时输出到控制台
#     ]
# )
#
# # 重定向 print
# print = logging.info


def make_base():

    ''' 
    Function to do the base level training on the first set of client data 
    for the genesis block
    '''
    reset()
    dataset = None
    with open("data/federated_data_0.d",'rb') as f:
        dataset = pickle.load(f)
    worker = NNWorker(dataset["train_images"],
        dataset["train_labels"],
        dataset["test_images"],
        dataset["test_labels"],
        0,
        "base0")
    worker.build_base()
    model = dict()
    model['model'] = worker.get_model()
    model['accuracy'] = worker.evaluate()
    print("model['accuracy']",model['accuracy'])
    worker.close()
    return model


class PoWThread(Thread):
    def __init__(self, stop_event,blockchain,node_identifier):
        self.stop_event = stop_event
        Thread.__init__(self)
        self.blockchain = blockchain
        self.node_identifier = node_identifier
        self.response = None

    def run(self):
        block,stopped = self.blockchain.proof_of_work(self.stop_event)
        self.response = {
            'message':"End mining",
            'stopped': stopped,
            'block': str(block)
        }
        on_end_mining(stopped)


STOP_EVENT = Event()

app = Flask(__name__)
status = {
    's':"receiving",
    'id':str(uuid4()).replace('-',''),
    'blockchain': None,
    'address' : ""
    }

def mine():
    STOP_EVENT.clear()
    thread = PoWThread(STOP_EVENT,status["blockchain"],status["id"])
    status['s'] = "mining"
    thread.start()

def on_end_mining(stopped):
    if status['s'] == "receiving":
        return
    if stopped:
        status["blockchain"].resolve_conflicts(STOP_EVENT)
    status['s'] = "receiving"
    for node in status["blockchain"].nodes:
        requests.get('http://{node}/stopmining'.format(node=node))

@app.route('/transactions/new',methods=['POST'])
def new_transaction():
    # print("Enter new_transaction",status['s'])
    if status['s'] != "receiving":
        return 'Miner not receiving', 400
    values = request.get_json()

    required = ['client','baseindex','update','datasize','computing_time']
    if not all(k in values for k in required):
        print('Missing values')
        return 'Missing values', 400
    if values['client'] in status['blockchain'].current_updates:
        print('Model already stored')
        return 'Model already stored', 400
    index = status['blockchain'].new_update(values['client'],
        values['baseindex'],
        dict(pickle.loads(codecs.decode(values['update'].encode(), "base64"))),
        values['datasize'],
        values['computing_time'])
    for node in status["blockchain"].nodes:
        requests.post('http://{node}/transactions/new'.format(node=node),
            json=request.get_json())
    if (status['s']=='receiving' and (
        len(status["blockchain"].current_updates)>=status['blockchain'].last_block['update_limit'])):
        # or time.time()-status['blockchain'].last_block['timestamp']>status['blockchain'].last_block['time_limit'])):
        mine()  #从这里到proof-work
    response = {'message': "Update will be added to block {index}".format(index=index)}
    return jsonify(response),201

@app.route('/status',methods=['GET'])
def get_status():
    response = {
        'status': status['s'],
        'last_model_index': status['blockchain'].last_block['index']
        }
    return jsonify(response),200
#
@app.route('/finish',methods=['GET'])
def get_finish():
    response = {
        'status': status["blockchain"].finish
        }
    # print('response["status"]',response['status'])
    # print('len: ', response['len'],response['value'] )
    # if  response['len']>=status['blockchain'].last_block['update_limit']:
    #     status["blockchain"].finished_nodes=[]
    return jsonify(response),200

@app.route('/finish/send',methods=['POST'])
def send_finish():

    # for node in status["blockchain"].nodes:
    #     requests.post('http://{node}//finish/send'.format(node=node),
    #                   json=request.get_json())
    value = request.get_json()
    status["blockchain"].finished_nodes.append(value['id'])



    print(status['blockchain'].last_block['update_limit'],len(status["blockchain"].finished_nodes),sorted(status["blockchain"].finished_nodes))

    if (len(status["blockchain"].finished_nodes)>= status['blockchain'].last_block['update_limit']):
        status["blockchain"].finish = False
        status["blockchain"].finished_nodes = []

    #     # status["blockchain"].finish=False
    #     status["blockchain"].finished_nodes=set()
    #     response={'status': 'success'}
    #     return jsonify(response), 201
    # else:
    #     response = {'status': 'waiting more'}
    #     response = {'status': 'waiting more'}

    return "success", 201




@app.route('/chain',methods=['GET'])
def full_chain():
    response = {
        'chain': status['blockchain'].hashchain,
        'length':len(status['blockchain'].hashchain)
    }
    return jsonify(response),200

@app.route('/nodes/register',methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Enter valid nodes in the list ", 400
    for node in nodes:
        if node!=status['address'] and not node in status['blockchain'].nodes:
            status['blockchain'].register_node(node)
            for miner in status['blockchain'].nodes:
                if miner!=node:
                    print("node",node,"miner",miner)
                    requests.post('http://{miner}/nodes/register'.format(miner=miner),
                        json={'nodes': [node]})
    response = {
        'message':"New nodes have been added",
        'total_nodes':list(status['blockchain'].nodes)
    }
    return jsonify(response),201

@app.route('/block',methods=['POST'])
def get_block():
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block","rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
            json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block","wb") as f:
                    pickle.dump(block,f)
    valid = False
    if Blockchain.hash(str(block))==hblock['hash']:
        valid = True
    response = {
        'block': str(block),
        'valid': valid
    }
    return jsonify(response),200

@app.route('/model',methods=['POST'])
def get_model():
    values = request.get_json()
    hblock = values['hblock']
    block = None
    if status['blockchain'].curblock.index == hblock['index']:
        block = status['blockchain'].curblock
    elif os.path.isfile("./blocks/federated_model"+str(hblock['index'])+".block"):
        with open("./blocks/federated_model"+str(hblock['index'])+".block","rb") as f:
            block = pickle.load(f)
    else:
        resp = requests.post('http://{node}/block'.format(node=hblock['miner']),
            json={'hblock': hblock})
        if resp.status_code == 200:
            raw_block = resp.json()['block']
            if raw_block:
                block = Block.from_string(raw_block)
                with open("./blocks/federated_model"+str(hblock['index'])+".block","wb") as f:
                    pickle.dump(block,f)
    valid = False
    model = block.basemodel
    if Blockchain.hash(codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode())==hblock['model_hash']:
        valid = True
    response = {
        'model': codecs.encode(pickle.dumps(sorted(model.items())), "base64").decode(),
        'valid': valid
    }
    return jsonify(response),200

@app.route('/nodes/resolve',methods=["GET"])
def consensus():
    replaced = status['blockchain'].resolve_conflicts(STOP_EVENT)
    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': status['blockchain'].hashchain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': status['blockchain'].hashchain
        }
    return jsonify(response), 200


@app.route('/stopmining',methods=['GET'])
def stop_mining():
    status['blockchain'].resolve_conflicts(STOP_EVENT)
    response = {
        'mex':"stopped!"
    }
    return jsonify(response),200

@app.route('/epoch/receive',methods=['POST'])
def receive_epoch():
    value = request.get_json()
    status['blockchain'].epoch=value['epoch']
    print(status['blockchain'].epoch)
    response = {'message': "Epoch is updated"}
    return jsonify(response), 201

@app.route('/malic/receive',methods=['POST'])
def receive_malic():
    value = request.get_json()
    status['blockchain'].malic_done+=np.array(value['malic_done'])
    response = {'message': "Malic_done is updated"}
    return jsonify(response), 201

@app.route('/allocation',methods=['GET'])
def allocation():
    # rep=status['blockchain'].reputation
    # N=len(rep)
    # M=status['blockchain'].member_num
    # deta_rep=np.zeros(N)
    # np.random.seed(status['blockchain'].epoch)
    # for i in range(N):
    #     deta_rep[i] = np.random.uniform(0, rep[i])
    # probabilities =  deta_rep / np.sum(deta_rep)
    # committee_indices = np.random.choice(range(N), size=M, replace=False, p=probabilities)
    # non_committee_indices = [i for i in range(N) if i not in committee_indices]
    # status['blockchain'].committee_indices= sorted(committee_indices)
    # status['blockchain'].worker_indices= sorted(non_committee_indices)
    # status['blockchain'].deta_rep=deta_rep[committee_indices]

    non_committee_indices=status['blockchain'].worker_indices

    return jsonify(non_committee_indices),200

@app.route('/malicious/is',methods=['GET'])
def is_malicious():
    response = list(status['blockchain'].is_malicious)
    return jsonify(response), 200


def delete_prev_blocks():
    files = glob.glob('blocks/*.block')
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    files = glob.glob('clients/*.block')
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5006, type=int, help='port to listen on')  #default=5000
    parser.add_argument('-i', '--host', default='127.0.0.32', help='IP address of this miner')  #default='127.0.0.3'
    parser.add_argument('-g', '--genesis', default=0, type=int, help='instantiate genesis block')
    parser.add_argument('-l', '--ulimit', default=15, type=int, help='number of updates stored in one block')
    parser.add_argument('-ma', '--maddress', help='other miner IP:port')
    parser.add_argument('-e', '--epochs', default=50, type=int, help='Number of epochs')
    args = parser.parse_args()
    address = "{host}:{port}".format(host=args.host,port=args.port)
    status['address'] = address
    if args.genesis==0 and args.maddress==None:
        raise ValueError("Must set genesis=1 or specify maddress")
    delete_prev_blocks()
    agent_num=20
    member_num=5
    malicious_num=11
    if args.genesis==1:
        model = make_base()
        print("base model accuracy:",model['accuracy'])
        status['blockchain'] = Blockchain(address,model,True,args.ulimit,args.epochs,agent_num,member_num,malicious_num)
    else:
        status['blockchain'] = Blockchain(address)
        status['blockchain'].register_node(args.maddress)
        requests.post('http://{node}/nodes/register'.format(node=args.maddress),
            json={'nodes': [address]})
        status['blockchain'].resolve_conflicts(STOP_EVENT)
    app.run(host=args.host,port=args.port,debug=True, use_reloader=False)
