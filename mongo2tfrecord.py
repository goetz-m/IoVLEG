import tensorflow as tf
import pymongo 
import os
from itertools import chain




def serialize_example(features,  label):
    assert isinstance(features, list)
    #assert isinstance(features2, list)
    assert isinstance(label, int)
    feature = {
        'features_list': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        #'features_list2': tf.train.Feature(float_list=tf.train.FloatList(value=features2)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def str_mongo_to_sparse_tfrecord(save_dir, col_name, db_name='Mixed_613', bs=20000):
    save_dir = os.path.join(save_dir, col_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    client = pymongo.MongoClient()
    db = client.get_database(db_name)
    col = db.get_collection(col_name)

    pa_cn = 0
    save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
    tfrecord_writer = tf.io.TFRecordWriter(save_path)

    sample_n = 0
    for bson_ds in col.find(no_cursor_timeout=True):

        label = bson_ds['label']
        features_list = bson_ds['features_list']
        features_list = list(chain(*features_list))
        #features_list2 = bson_ds['features_list2']
        tfrecord_writer.write(serialize_example(features_list, label))
        sample_n += 1

        if bs > 0 and sample_n % bs == 0:
            tfrecord_writer.close()
            pa_cn += 1
            save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
            tfrecord_writer = tf.io.TFRecordWriter(save_path)

    tfrecord_writer.close()
    client.close()


str_mongo_to_sparse_tfrecord(save_dir='./tf',
              col_name='train', db_name='Mixed_613', bs=20000)

str_mongo_to_sparse_tfrecord(save_dir='./tf',
              col_name='test', db_name='Mixed_613', bs=20000)

str_mongo_to_sparse_tfrecord(save_dir='./tf',
              col_name='valid', db_name='Mixed_613', bs=20000)