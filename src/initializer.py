import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json

FLAGS = tf.flags.FLAGS

with open('hyper_params.json', 'r') as f:
    params_dict = json.load(f)

# params_dict = json.loads(params_dict)
wgan_flags = params_dict[0]
ann_flags = params_dict[1]
data_flags = params_dict[2]

tf.flags.DEFINE_bool('k_fold_pcf2', False, 'Use Kfold for pcf2')
tf.flags.DEFINE_bool('k_fold', False, 'Use Kfold for pcf-SPR')
tf.flags.DEFINE_bool('train_wgan', False, 'train wgan, defaul:True')
tf.flags.DEFINE_bool('generate', False, 'generate data, defaul:True')
tf.flags.DEFINE_bool('train_ann', False, 'train ann, defaul:True')
tf.flags.DEFINE_bool('test_ann', False, 'test ann, defaul:True')

tf.flags.DEFINE_integer('save_step',wgan_flags['save_step'], 'Wgan save step')
tf.flags.DEFINE_integer('wgan_batch_size', wgan_flags['BATCH_SIZE'], 'batch size, default: 12')
tf.flags.DEFINE_integer('noise_dim', wgan_flags['noise_dim'], 'z_dimension, default:7')
tf.flags.DEFINE_integer('num_critic_input', wgan_flags['num_critic_input'], 'number of inputs to the critic, default:7')
tf.flags.DEFINE_integer('n_critic', wgan_flags['n_critic'], 'Critic iterations, default:5')
tf.flags.DEFINE_integer('grad_penalty_weight', wgan_flags['grad_penalty_weight'], 'Gradient penalty weight, default:10')
tf.flags.DEFINE_integer('num_examples_to_generate', wgan_flags['num_examples_to_generate'],
                        'number of examples to generate, default:8')
tf.flags.DEFINE_integer('wgan_epochs', wgan_flags['epochs'], 'WGAN epochs,default:2000')
tf.flags.DEFINE_integer('gen_num_layers', wgan_flags['gen_num_layers'], 'Generator number of hidden layers,default:5')
tf.flags.DEFINE_integer('d_num_layers', wgan_flags['cr_num_layers'], 'Critic number of hidden layers,default:5')
tf.flags.DEFINE_bool('gen_batch_norm', wgan_flags['gen_batch_norm'], 'Generator batch normalization')
tf.flags.DEFINE_string('load_wga_model_dir', wgan_flags['load_wga_model_dir'], 'Saved wgan model directory')
tf.flags.DEFINE_integer('num_layers', ann_flags['num_layers'], 'number of hidden layers of the ANN model, default:6')
tf.flags.DEFINE_integer('num_inputs', ann_flags['num_inputs'], 'number of inputs to the ANN , default:6')
tf.flags.DEFINE_integer('num_outputs', ann_flags['num_outputs'], 'number of outputs of the ANN ,default:1')
tf.flags.DEFINE_integer('num_neurons', ann_flags['num_neurons'], 'number of neurons in the ANN, default:50')
tf.flags.DEFINE_float('lr', ann_flags['learning_rate'], 'ANN learning rate, default:1e-3')
tf.flags.DEFINE_integer('epochs', ann_flags['epochs'], 'ANN epochs, default:2000')
tf.flags.DEFINE_integer('ANN_batch_size', ann_flags['batch_size'], 'ANN batch size,default:8')
tf.flags.DEFINE_bool('batch_norm', ann_flags['batch_norm'], 'ANN batch norm,default:True')
tf.flags.DEFINE_string('save_dir', ann_flags['save_dir'], 'ANN save directory, default:model1000.h5')
tf.flags.DEFINE_string('chkdir', ann_flags['chkdir'], 'ANN checkpoint directory, default:weights1000.hdf5')
tf.flags.DEFINE_string('model_to_test', ann_flags['model_to_test'], 'Test model directory')

tf.flags.DEFINE_integer('augment_size', ann_flags['augment_size'], 'data augmentation size, default:1000')
tf.flags.DEFINE_integer('gen_iterations', wgan_flags['gen_iterations'], 'iteration of data generation, default:1000')

tf.flags.DEFINE_integer('no_spr_samples_per_analyte',data_flags['no_spr_samples_per_analyte'], 'Number of samples per analyte')
tf.flags.DEFINE_integer('no_spr_samples',data_flags['no_spr_samples_per_analyte'], 'Number of samples per analyte')
tf.flags.DEFINE_integer('no_spr_analytes',data_flags['no_spr_analytes'], 'Number of Analytes')
tf.flags.DEFINE_integer('no_spr_configs',data_flags['no_spr_configs'], 'Number of SPR geometric configurations')

tf.flags.DEFINE_string('data_set', data_flags['data_set'], 'The dataset used to train the system: PCF or SPR')
tf.flags.DEFINE_string('gen_data_dir', data_flags['gen_data_dir'],
                       'generated data directory,default:gen_data/gen_data.txt')
tf.flags.DEFINE_string('data', data_flags['data'], 'SPR-PCF data directory')
tf.flags.DEFINE_string('shuffled_data', data_flags['shuffled_data'], 'SPRC-PCF suffled data directory')
tf.flags.DEFINE_string('pcf_data', data_flags['pcf_data'], 'PCF data directory')
tf.flags.DEFINE_string('shuffled_pcf_data', data_flags['shuffled_pcf_data'], 'PCF shuffled data directory')


def init():
    return FLAGS
