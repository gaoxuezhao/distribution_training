# encoding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from model import *
from stream import *
from config import *
import tensorflow as tf
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

FLAGS = None

class Trainer():
    def __init__(self, stream, model):
        self.stream=stream
        self.model=model
    
    def train(self):
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")

        # Create a cluster from the parameter server and worker hosts.
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index,
                                 start=True)
        print("server.target")
        print(server.target)
        print(server.target)
        print(server.target)
        print("server.target")
        # # 参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程。
        # server.join()会一直停在这条语句上
        if FLAGS.job_name == "ps":
            server.join()
        elif FLAGS.job_name == "worker":

            # 通过tf.train.replica_device_setter函数来指定执行每一个运算的设备
            # tf.train.replica_device_setter函数会自动将所有的参数分配到参数服务器上，而
            # 计算分配到当前的计算服务器上

            # tf.train.replica_device_setter()会根据job名，将with内的Variable op放到ps tasks，
            # 将其他计算op放到worker tasks。默认分配策略是轮询。
            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):

                    feature, label = self.stream.get_data_batch()
                    
                    Yp = self.model.build_graph(feature)
                    loss=-tf.reduce_sum(label*tf.log(Yp))
                    global_step = tf.train.get_or_create_global_step()
                    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
                    train_op = optimizer.minimize(loss, global_step=global_step)
                    
                    correct_prediction = tf.equal(tf.argmax(Yp,1), tf.argmax(label,1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    
                    max_acc=0
                    
                    #np.set_printoptions(threshold='nan') 
                    np.set_printoptions(threshold=sys.maxsize) 
                    # tensorboard
                    tf.summary.scalar('cost', loss)
                    tf.summary.scalar("accuracy", accuracy)
                    summary_op = tf.summary.merge_all()

                    # The StopAtStepHook handles stopping after running given steps.
                    stop_hook=tf.train.StopAtStepHook(last_step=7000000)
                    
                    # 默认在所有的节点上执行，所以会在非chief节点上执行save
                    #saver=tf.train.Saver(max_to_keep=100)
                    #save_hook=tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.checkpoint_dir,save_steps = 500, saver=saver)
                    
                    #hooks=[stop_hook, save_hook]
                    hooks=[stop_hook]

                    # # 通过设置log_device_placement选项来记录operations 和 Tensor 被指派到哪个设备上运行
                    ## 为了避免手动指定的设备不存在这种情况, 你可以在创建的 session 里把参数 allow_soft_placement
                    ## 设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
                    ## device_filters:硬件过滤器，如果被设置的话，会话会忽略掉所有不匹配过滤器的硬件。
                    # config = tf.ConfigProto(
                    #     allow_soft_placement=True,
                    #     log_device_placement=False,
                    #     device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                    # )

                    # 通过设置log_device_placement选项来记录operations 和 Tensor 被指派到哪个设备上运行
                    config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                    )
                    
                    # for train_init_op tf.data.Dataset if without this sentence, will crash with error no init train_init_op
                    scaffold = tf.train.Scaffold(
                        
                        #local_init_op=tf.group(tf.local_variables_initializer(),
                        #                    train_init_op), 
                        saver=tf.train.Saver(max_to_keep=100)
                    )

                    # The MonitoredTrainingSession takes care of session initialization,
                    # restoring from a checkpoint, saving to a checkpoint, and closing when done
                    # or an error occurs.
                    # master="grpc://" + worker_hosts[FLAGS.task_index]
                    # if_chief: 制定task_index为0的任务为主任务，用于负责变量初始化、做checkpoint、保存summary和复原
                    # 定义计算服务器需要运行的操作。在所有的计算服务器中有一个是主计算服务器。
                    # 它除了负责计算反向传播的结果，它还负责输出日志和保存模型
                    step=0
                    with tf.train.MonitoredTrainingSession(master=server.target,
                                                           config=config,
                                                           is_chief=(FLAGS.task_index == 0),
                                                           hooks=hooks,
                                                           #hooks=[],
                                                           save_checkpoint_secs=60,
                                                           checkpoint_dir=FLAGS.checkpoint_dir, 
                                                           scaffold=scaffold
                                                           ) as mon_sess:
                        #tf.global_variables_initializer().run()
                        #tf.local_variables_initializer().run()
                        i=0
                        try:
                            while not mon_sess.should_stop():
                                # Run a training step asynchronously.
                                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                                # perform *synchronous* training.
                                # mon_sess.run handles AbortedError in case of preempted PS.
                                step, _, loss_r, accuracy_r = mon_sess.run([global_step, train_op, loss, accuracy])
                                
                                #if accuracy_r > max_acc:
                                #    max_acc = accuracy_r
                                #    #saver.save(sess,'ckpt/mnist.ckpt', global_step=i)
                                #    print("step %d, loss: %f, accur:%f" %(step, loss_r, accuracy_r))
                                if step % 1 == 0:
                                    print("jobname %s, task_index %d, global_step %d, local_step: %d, loss: %f, accur:%f" %(FLAGS.job_name, FLAGS.task_index, step, i, loss_r, accuracy_r))
                                i+=1
                        except tf.errors.OutOfRangeError:
                            print('done! now lets kill all the threads')
                        finally:
                            print('all threads are asked to stop!')
                        print('all threads are stopped!')
                        print("%d"%(i-1))


def main(_):
    data_path='../data/train.tfrecord'
    config = Config(data_path)
    stream = Stream(config)
    model = Model()
    trainer = Trainer(stream, model)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="path to a directory where to restore variables."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate"
    )

    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
