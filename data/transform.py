import numpy as np
import tensorflow as tf

class digit_image:
    def __init__(self,feature,label):
        self.width=28
        self.height=28
        self.feature=np.array(feature, dtype=np.float32)
        self.label=label

class tfrecord_cntl:
    def __init__(self,list_of_image, tfrecord_name):
        self.list_of_image=list_of_image
        self.tfrecord_name=tfrecord_name

    def image_to_tfrecorde(self, image):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'feature':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image.feature.tostring()])),
                    'label':tf.train.Feature(float_list= tf.train.FloatList(value=[image.label])),
                    'height':tf.train.Feature(int64_list= tf.train.Int64List(value=[image.height])),
                    'width':tf.train.Feature(int64_list= tf.train.Int64List(value=[image.width]))
                    }
                )
            )

        return example

    def proceding(self):
        writer = tf.python_io.TFRecordWriter(self.tfrecord_name)
        i=1
        for image in self.list_of_image:
            tf_example=self.image_to_tfrecorde(image)
            writer.write(tf_example.SerializeToString())
            if i % 100 == 0:
                print 'tfrecord:', i
            i+=1

        writer.close()


class image_digit_cntl:
    def __init__(self, input_path):
        self.input_path=input_path

    def proceding(self):
        fp=open(self.input_path,'r')
        i=1
        image_instances=[]
        for line in fp.readlines():
            if 'label' in line:
                continue
        
            feature_label=map(eval, line.split(','))
            label=feature_label[0]
            feature=feature_label[1:]
            
            image_instance=digit_image(feature, label)
            image_instances.append(image_instance)

            if i % 100 == 0:
                print 'image:', i
            i+=1

        return image_instances

if __name__=='__main__':
    file_path='validation.csv'
    tfrecord_path='validation.tfrecord'
    #file_path='validation.csv'
    #tfrecord_path='validation.tfrecord'
    image_list=image_digit_cntl(file_path).proceding()
    tfrecord_cntl(image_list, tfrecord_path).proceding()
    exit();
    i=0
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecord"):
        i+=1
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        image = example.features.feature['feature'].bytes_list.value
        label = example.features.feature['label'].float_list.value
        print i, image, label
