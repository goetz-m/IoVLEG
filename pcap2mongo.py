import pymongo
import pyshark
import os
import hashlib
import numpy as np
import pandas as pd
import socket
import random
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer





#label_dic = {'NetBIOS':0, 'UDP':1}
#label_dic = {'DrDoS_MSSQL':0, 'DrDoS_NetBIOS':1, 'DrDoS_SSDP':2, 'DrDoS_UDP':3, 'Syn':4, 'UDP-lag':5}
victim_ips = ['192.168.50.4','192.168.50.6','192.168.50.7','192.168.50.8','192.168.50.9']

protocols = ['IP','Ethernet','Padding','ARP','DATA','DNS','ICMP','HTTP','TCP','LLC','VXLAN','Raw']
powers_of_two = np.array([2**i for i in range(len(protocols))])
vector_proto = CountVectorizer()
vector_proto.fit_transform(protocols).todense()

def _shuffle(arr):
    random.shuffle(arr)
    random.shuffle(arr)
    random.shuffle(arr)
    return arr


def _insert_by_id(target_col, pairs, original_db):
    for pair in pairs:
        name, bid = pair
        bson_targets = original_db.get_collection(name).find({'_id': bid}, no_cursor_timeout=True)
        tmp = [x for x in bson_targets]
        assert len(tmp) == 1
        sample = tmp[0]
        sample.pop('_id')
        target_col.insert_one(sample)


def parse_packet(pkt):
    features_list = []
    tmp_id = [0,0,0,0,0]


    try:
        features_list.append(float(pkt.sniff_timestamp))  # timestampchild.find('Tag').text
        features_list.append(int(pkt.ip.len))  # packet length
        features_list.append(int(hashlib.sha256(str(pkt.highest_layer).encode('utf-8')).hexdigest(),
                                    16) % 10 ** 8)  # highest layer in the packet
        features_list.append(int(int(pkt.ip.flags, 16)))  # IP flags
        tmp_id[0] = str(pkt.ip.src)  # int(ipaddress.IPv4Address(pkt.ip.src))
        tmp_id[2] = str(pkt.ip.dst)  # int(ipaddress.IPv4Address(pkt.ip.dst))

        protocols = vector_proto.transform([pkt.frame_info.protocols]).toarray().tolist()[0]
        protocols = [1 if i >= 1 else 0 for i in
                     protocols]  # we do not want the protocols counted more than once (sometimes they are listed twice in pkt.frame_info.protocols)
        protocols_value = int(np.dot(np.array(protocols), powers_of_two))
        features_list.append(protocols_value)

        features_list.append(int(pkt.ip.dsfield,16))
        #features_list.append(int(pkt.ip.ttl))

        protocol = int(pkt.ip.proto)
        tmp_id[4] = protocol
        if pkt.transport_layer != None:
            if protocol == socket.IPPROTO_TCP:
                tmp_id[1] = int(pkt.tcp.srcport)
                tmp_id[3] = int(pkt.tcp.dstport)
                features_list.append(int(pkt.tcp.len))  # TCP length
                features_list.append(int(pkt.tcp.ack))  # TCP ack
                features_list.append(int(pkt.tcp.flags, 16))  # TCP flags
                features_list.append(int(pkt.tcp.window_size_value))  # TCP window size
                features_list = features_list + [0, 0]  # UDP + ICMP positions
            elif protocol == socket.IPPROTO_UDP:
                features_list = features_list + [0, 0, 0, 0]  # TCP positions
                tmp_id[1] = int(pkt.udp.srcport)
                features_list.append(int(pkt.udp.length))  # UDP length
                tmp_id[3] = int(pkt.udp.dstport)
                features_list = features_list + [0]  # ICMP position
        elif protocol == socket.IPPROTO_ICMP:
            features_list = features_list + [0, 0, 0, 0, 0]  # TCP and UDP positions
            features_list.append(int(pkt.icmp.type))  # ICMP type
        else:
            features_list = features_list + [0, 0, 0, 0, 0, 0]  # padding for layer3-only packets
            tmp_id[4] = 0

        if tmp_id[0] < tmp_id[2]:
            id = f'{tmp_id[0]}-{tmp_id[1]}-{tmp_id[2]}-{tmp_id[3]}-{tmp_id[4]}'
        else:
            id = f'{tmp_id[2]}-{tmp_id[3]}-{tmp_id[0]}-{tmp_id[1]}-{tmp_id[4]}'

        return id,features_list

    except AttributeError as e:
        # ignore packets that aren't TCP/UDP or IPv4
        return e,e

def normalize_padding(flow,max_pkts_limit=50):
    start_time = flow['begin_time']
    features_list = flow['features_list']
    features_list[:,0] = features_list[:,0] - start_time
    pkt_num = features_list.shape[0]
    if pkt_num > max_pkts_limit:
        features_list = features_list[:max_pkts_limit,...]
        pkt_num = max_pkts_limit
    norm = normalize(features_list,norm='max',axis=0)
    norm = np.pad(norm,((0,max_pkts_limit-pkt_num),(0,0)),'constant',constant_values=(0, 0))
    flow['features_list'] = norm.tolist()
    #flow['features_list2'] = csv_data.get_chunk(1).values[0].tolist()
    


if __name__=='__main__':
    file_path = ''
    label_dic = {'Benign':0,'MSSQL':1, 'NetBIOS':2, 'LDAP':3, 'SYN':4,'UDP':5}

    time_thresh1 = 120
    time_thresh2 = 5
    max_pkts_limit = 50
    session = pymongo.MongoClient()
    database = session.get_database('Feature')
    
    #for file in label_dic.keys():
    for file in os.listdir(file_path):
        print('Wroking on {0}'.format(file))
        flow_maps = dict()
        label = file.split('.')[0]

        
        label_num = label_dic[label]

        #csv_file = os.path.join(csv_path,label+'.csv')
        #csv_data = pd.read_csv(csv_file,iterator = True)
        col = database.get_collection(label)
        cap = pyshark.FileCapture(os.path.join(file_path,file))
        #cap = pyshark.FileCapture(file)
        
        cnt = 0
        for i,pkt in enumerate(cap):

            try:
                pkt.ip
            except Exception as e:
                continue

            if label!='Benign' and pkt.ip.src not in victim_ips and pkt.ip.dst not in victim_ips:
                continue
            
            id,features = parse_packet(pkt)
            if isinstance(id,Exception):
                continue
            
            pkt_time = float(pkt.sniff_timestamp)
            if id in flow_maps:
                cur_biflow = flow_maps[id]
                last_seen_time = cur_biflow['last_seen_time']
                begin_time = cur_biflow['begin_time']
                if pkt_time - begin_time > time_thresh1 \
                    or pkt_time - last_seen_time > time_thresh2:
                    normalize_padding(cur_biflow,max_pkts_limit)
                    col.insert_one(cur_biflow)
                    cnt += 1
                    
                    flow_maps[id] = {
                        'biflow_id': id,
                        'features_list': np.array([features]),
                        'begin_time': pkt_time,
                        'last_seen_time': pkt_time,
                        'label': label_num
                    }
                else:
                    cur_biflow['features_list'] = np.vstack((cur_biflow['features_list'],features))
                    cur_biflow['last_seen_time'] = pkt_time
            else:
                flow_maps[id] = {
                        'biflow_id': id,
                        'features_list': np.array([features]),
                        'begin_time': pkt_time,
                        'last_seen_time': pkt_time,
                        'label': label_num
                    }
        for ele in list(flow_maps.values()):
            normalize_padding(ele,max_pkts_limit)
            col.insert_one(ele)
            cnt += 1
        print(f'Biflow cnt: {cnt}, Done')
        cap.close()
    session.close()

    raw_db = 'Feature'
    new_db = 'Mixed_613'
    session = pymongo.MongoClient()
    raw_db = session.get_database(raw_db)
    new_db = session.get_database(new_db)

    train_ids, valid_ids, test_ids = [],[],[] 
    oversample_dict = {}
    undersample_dict = {}
    names = raw_db.list_collection_names()
    for name in names:
        
        col = raw_db.get_collection(name)
        cur_ids,train_tmp,test_tmp,valid_tmp = [],[],[],[]
        for bs in col.find(no_cursor_timeout=True):
            cur_ids.append((name,bs['_id']))

        flow_num = len(cur_ids)
        len1 = int(flow_num*0.6)
        len2 = int(flow_num*0.7)

        train_tmp = cur_ids[:len1]
        valid_tmp = cur_ids[len1:len2]
        test_tmp =  cur_ids[len2:]

        if label in oversample_dict:
            sample_rate = oversample_dict[label]
            assert sample_rate > 1
            cnt = 1
            while cnt < sample_rate:
              cnt += 1
              print(cnt)
              train_ids.extend(train_tmp)
              valid_ids.extend(valid_tmp)
              test_ids.extend(test_tmp)
        elif label in undersample_dict:
            sample_rate = undersample_dict[label]
            assert sample_rate < 1
            len1_ = int(len1*sample_rate)
            len2_ = int(len2*sample_rate)
            len3_ = int(flow_num*sample_rate)
            train_ids = train_tmp[:len1_]
            valid_ids = valid_tmp[len1_:len2_]
            test_ids = test_tmp[len2_:len3_]
        else:
            train_ids.extend(train_tmp)
            valid_ids.extend(valid_tmp)
            test_ids.extend(test_tmp)
    

    print(' ===== train ===== ')
    train_ids = _shuffle(train_ids)
    train_ids = _shuffle(train_ids)
    train_col = new_db.get_collection('train')
    _insert_by_id(train_col, train_ids, raw_db) 

    print(' ===== validation ===== ')
    valid_ids = _shuffle(valid_ids)
    valid_ids = _shuffle(valid_ids)
    valid_col = new_db.get_collection('valid')
    _insert_by_id(valid_col, valid_ids, raw_db)  
        
    print(' ===== test ===== ')
    test_ids = _shuffle(test_ids)
    test_ids = _shuffle(test_ids)
    test_col = new_db.get_collection('test')
    _insert_by_id(test_col, test_ids, raw_db)
    session.close()
    








        
