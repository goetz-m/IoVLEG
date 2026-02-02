
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn import tree
import pandas as pd
import numpy as np
import pickle
import os

def load_feature():
    path = ''
    dic = {'DrDoS_MSSQL':0, 'DrDoS_NetBIOS':1, 'DrDoS_SSDP':2, 'DrDoS_UDP':3, 'Syn':4, 'UDP-lag':5,
            'DrDoS_DNS':6,'DrDoS_LDAP':7,'DrDoS_NTP':8,'DrDoS_SNMP':9,'TFTP':10}
    
    drop_list = ['Unnamed: 0','Flow ID',' Source IP',' Source Port',' Destination IP',
                    ' Destination Port',' Timestamp','SimillarHTTP']
    conc = pd.DataFrame()
    for file in os.listdir(path):
        print('--------process----------{0}'.format(file))
        label = file.split('.')[0]
        data = pd.read_csv(os.path.join(path,file),nrows=600000)
        
        data.drop(columns=drop_list,axis=1,inplace=True)
        data.replace([np.inf,-np.inf],np.nan,inplace=True)
        data.dropna(inplace=True)
        data.replace(dic,inplace=True)

    
        data = data.loc[data[' Label']==dic[label]]
        print(data.shape)
        if data.shape[0]<500000:
            pass
        else:
            data = data.iloc[:500000,:]

        #columns = data.columns[:-1]
        #scaler = MinMaxScaler()
        #data[columns] = scaler.fit_transform(data[columns])
        conc = pd.concat([conc,data],ignore_index=True)
    columns = conc.columns[:-1]
    for column in columns:
        if conc[column].dtypes==np.int64:
            conc[column] = conc[column].astype(np.float64)
        elif conc[column].dtypes==np.float64:
            pass
        else:
            print("Something goes wrong with the data type")
    conc[' Label'] = conc[' Label'].astype(np.int32)
    return conc


def feature_rank(data):

    columns = data.columns[:-1]

    data = shuffle(data)
    data = shuffle(data)
    
    #mutual_info = mutual_info_classif(data[columns],data[' Label'])
    data_train,data_test = train_test_split(data,test_size=0.3,random_state=123)
    x_train = data_train[columns]
    y_train = data_train[' Label']

    x_test = data_test[columns]
    y_test = data_test[' Label']

    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
    clf.fit(x_train,y_train)
    result = permutation_importance(clf,x_train,y_train,n_repeats=30,random_state=0)
    #result = clf.feature_importances_
    y_pred = clf.predict(x_test)
    
    show_result(y_pred,y_test)

    zipped = zip(columns,result.importances_mean)
    
    return zipped

def feature_select(score,data):
    features = [x[0] for x in score][:15]
    columns = data.columns[:-1]
    drop_list = list(set(columns) - set(features))
    data.drop(columns=drop_list,axis=1,inplace=True)

    data = shuffle(data)
    data = shuffle(data)
    
    data_train,data_test = train_test_split(data,test_size=0.3,random_state=123)

    columns = data.columns[:-1]
    x_train = data_train[columns]
    y_train = data_train[' Label']
    x_test = data_test[columns]
    y_test = data_test[' Label']

    clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    show_result(y_pred,y_test)

def show_result(y_pred,y_test):
    digits = 6
    label_names = ['DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'UDP-lag',
            'DrDoS_DNS','DrDoS_LDAP','DrDoS_NTP','DrDoS_SNMP','TFTP']
    cl_re = classification_report(y_test, y_pred, digits=digits,
                                  labels=[i for i in range(len(label_names))],
                                  target_names=label_names, output_dict=True)
    print(round(cl_re['accuracy'], digits))
    print(round(cl_re['macro avg']['precision'], digits))
    print(round(cl_re['macro avg']['recall'], digits))
    print(round(cl_re['macro avg']['f1-score'], digits))

def show_importance(zipped):
    fin = sorted(zipped,key=lambda x:x[1] )

    fea = [x[0] for x in fin]
    imp = [x[1] for x in fin]
    fig,ax = plt.subplots(figsize=(10,8))
    ax.barh(np.arange(len(fea)),imp,color='xkcd:grey')
    ax.set_yticks(np.arange(len(fea)))
    ax.set_yticklabels(['F'+str(x) for x in range(len(fea))])
    ax.set_xlabel('Importance score')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    for x,y in enumerate(imp):
        plt.text(y,x,'%s'%round(y,2),va='center')
    plt.savefig('./figure/feature_importance.png')

def feature_correlation(data,drop=False):
    preserve = [' Flow IAT Min', ' Total Fwd Packets', ' act_data_pkt_fwd', 'Subflow Fwd Packets', ' min_seg_size_forward', ' Fwd Header Length.1',
                ' Fwd IAT Min', ' Protocol', ' Fwd Header Length', ' Inbound']
    data = data.loc[:,(data!=0).any(axis=0)]
    columns = data.columns[:-1]
    
    if drop:
        drop_list = list(set(columns)-set(preserve))
        data.drop(columns=drop_list,axis=1,inplace=True)
        columns = data.columns[:-1]
    corr = data[columns].corr(method='pearson')
    fig,ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True,vmax=1,vmin=-1,square=True,cmap='Greys',fmt='.2f',xticklabels=['F'+str(x) for x in range(len(preserve))],
                yticklabels=['F'+str(x) for x in range(len(preserve))])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Feature')
    plt.title('Correlation')
    plt.tight_layout()
    plt.savefig('./figure/corr.png')
    #corr.to_csv('./corr3.csv')

if __name__=='__main__':
    
    feature = load_feature() 
    col1 = feature.columns[:-1]
    feature = feature.loc[:,(feature!=0).any(axis=0)]
    col2 = feature.columns[:-1]
    print(list(set(col1)-set(col2)))
