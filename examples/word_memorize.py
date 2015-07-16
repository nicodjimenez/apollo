import sys
import numpy as np
import random
import matplotlib
import os
import simplejson as json
import re
import pickle
import logging
import glob

import apollo
from apollo import layers

# maximum number of chars in a sentence
MAX_LEN = 100
random.seed(0)

def get_hyper():
    hyper = {}
    hyper['vocab_size'] = 256
    hyper['batch_size'] = 32
    hyper['init_range'] = 0.1
    hyper['zero_symbol'] = hyper['vocab_size'] - 2
    hyper['filler_symbol'] = hyper['vocab_size'] - 1
    hyper['test_interval'] = 10
    hyper['test_iter'] = 20
    hyper['base_lr'] = 20
    hyper['weight_decay'] = 0
    hyper['momentum'] = 0
    hyper['clip_gradients'] = 0.24
    hyper['display_interval'] = 10
    hyper['max_iter'] = 10000
    hyper['snapshot_prefix'] = '/tmp/char'
    hyper['snapshot_interval'] = 1000
    hyper['random_seed'] = 22
    hyper['gamma'] = 0.8
    hyper['stepsize'] = 2500
    hyper['mem_cells'] = 1000
    hyper['graph_interval'] = 1000
    hyper['graph_prefix'] = ''
    hyper['i_temperature'] = 1.5
    return hyper

hyper = get_hyper()

def get_batch_iter(sentence_list):
    while True: 
        batch = [random.choice(sentence_list).lower().strip() for _ in xrange(hyper['batch_size'])]
        yield batch 
        
def pad_batch(sentence_batch):
    # add 1 so that all sentences will have a stop symbol at the end
    max_len = max(len(x) for x in sentence_batch) + 1 
    result = []
    for sentence in sentence_batch:
        # add stop symbol at end of every sentence
        chars = [min(ord(c), 255) for c in sentence] + [hyper['zero_symbol']]
        result.append(chars + [hyper['filler_symbol']] * (max_len - len(chars)))

    return result

def forward(net, sentence_batch_iter):
    batch = next(sentence_batch_iter)
    sentence_batch = np.array(pad_batch(batch))
    length = min(sentence_batch.shape[1], MAX_LEN)
    assert length > 0

    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_seed',
        data=np.zeros((hyper['batch_size'], hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='label',
        data=np.zeros((hyper['batch_size'] * length, 1, 1, 1))))
    hidden_concat_bottoms = []
    for step in range(length):
        net.forward_layer(layers.DummyData(name=('word%d' % step),
            shape=[hyper['batch_size'], 1, 1, 1]))
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
            word = np.zeros(sentence_batch[:, 0].shape)
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
            word = sentence_batch[:, step - 1]

        net.tops['word%d' % step].data[:,0,0,0] = word
        net.forward_layer(layers.Wordvec(name=('wordvec%d' % step),
            bottoms=['word%d' % step],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        net.forward_layer(layers.Concat(name='lstm_concat%d' % step,
            bottoms=[prev_hidden, 'wordvec%d' % step]))
        net.forward_layer(layers.Lstm(name='lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout%d' % step,
            bottoms=['lstm%d_hidden' % step], dropout_ratio=0.16))
        hidden_concat_bottoms.append('dropout%d' % step)

    net.forward_layer(layers.Concat(name='hidden_concat', concat_dim=0, bottoms=hidden_concat_bottoms))
    net.tops['label'].data[:,0,0,0] = sentence_batch[:, :length].T.flatten()
    net.forward_layer(layers.InnerProduct(name='ip', bottoms=['hidden_concat'],
        num_output=hyper['vocab_size'], weight_filler=filler))
    loss = net.forward_layer(layers.SoftmaxWithLoss(name='softmax_loss',
        ignore_label=hyper['filler_symbol'], bottoms=['ip', 'label']))
    return loss

def eval_performance(net):
    eval_net = apollo.Net()
    eval_forward(eval_net)
    eval_net.copy_params_from(net)
    output_words = eval_forward(eval_net)
    print ''.join([chr(x) for x in output_words])

def softmax_choice(data):
    return np.random.choice(range(len(data.flatten())), p=data.flatten())

def argmax_choice(data):
    return np.argmax(data.flatten())

def eval_forward(net):
    output_words = []
    filler = layers.Filler(type='uniform', max=hyper['init_range'],
        min=(-hyper['init_range']))
    net.forward_layer(layers.NumpyData(name='lstm_hidden_prev',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))
    net.forward_layer(layers.NumpyData(name='lstm_mem_prev',
        data=np.zeros((1, hyper['mem_cells'], 1, 1))))

    length = MAX_LEN
    for step in range(length):
        net.forward_layer(layers.NumpyData(name=('word'),
            data=np.zeros((1, 1, 1, 1))))
        prev_hidden = 'lstm_hidden_prev'
        prev_mem = 'lstm_mem_prev'
        word = np.zeros((1, 1, 1, 1))
        if step == 0:
            output = 0
        else:
            output = softmax_choice(net.tops['softmax'].data)
            # we've encountered stop symbol and can exit
            if output == hyper["zero_symbol"]:
                net.reset_forward() 
                return output_words
            else:
                output_words.append(output)

        net.tops['word'].data[0,0,0,0] = output
        net.forward_layer(layers.Wordvec(name=('wordvec'),
            bottoms=['word'],
            dimension=hyper['mem_cells'], vocab_size=hyper['vocab_size'],
            param_names=['wordvec_param'], weight_filler=filler))
        net.forward_layer(layers.Concat(name='lstm_concat',
            bottoms=[prev_hidden, 'wordvec']))
        net.forward_layer(layers.Lstm(name='lstm',
            bottoms=['lstm_concat', prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm_hidden_next', 'lstm_mem_next'],
            num_cells=hyper['mem_cells'], weight_filler=filler))
        net.forward_layer(layers.Dropout(name='dropout',
            bottoms=['lstm_hidden_next'], dropout_ratio=0.16))

        net.forward_layer(layers.InnerProduct(name='ip', bottoms=['dropout'],
            num_output=hyper['vocab_size'], weight_filler=filler))
        net.tops['ip'].data[:] *= hyper['i_temperature']
        net.forward_layer(layers.Softmax(name='softmax',
            ignore_label=hyper['filler_symbol'], bottoms=['ip']))
        net.tops['lstm_hidden_prev'].data_tensor.copy_from(net.tops['lstm_hidden_next'].data_tensor)
        net.tops['lstm_mem_prev'].data_tensor.copy_from(net.tops['lstm_mem_next'].data_tensor)
        net.reset_forward()
    return output_words

if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
        with open(file_name, 'r') as f:
            sentence_list = f.read().splitlines()
            
        # this seems necessary, as widely varying lengths of input do not fare well
        sentence_list = [x.lower().strip() for x in sentence_list if len(x.strip()) == 10]
    else:
        #raise IOError("Valid dictionary file must be provided!")
        sentence_list = ["chris", "james", "sophia", "david", "jonathan", "richard", "mitchell", "mariah"]

    apollo.Caffe.set_random_seed(hyper['random_seed'])
    #apollo.Caffe.set_mode_gpu()
    apollo.Caffe.set_mode_cpu()
    apollo.Caffe.set_device(0)
    apollo.Caffe.set_logging_verbosity(3)

    assert len(sentence_list) > 0
    net = apollo.Net()
    sentence_batch_iter = get_batch_iter(sentence_list)

    forward(net, sentence_batch_iter)
    net.reset_forward()
    train_loss_hist = []

    for i in range(hyper['max_iter']):
        train_loss_hist.append(forward(net, sentence_batch_iter))
        net.backward()
        lr = (hyper['base_lr'] * (hyper['gamma'])**(i // hyper['stepsize']))
        net.update(lr=lr, momentum=hyper['momentum'],
            clip_gradients=hyper['clip_gradients'], weight_decay=hyper['weight_decay'])
        if i % hyper['display_interval'] == 0:
            logging.info('Iteration %d: %s' % (i, np.mean(train_loss_hist[-hyper['display_interval']:])))
        if i % hyper['test_interval'] == 0:
            eval_performance(net)
        if i % hyper['snapshot_interval'] == 0 and i > 0:
            filename = '%s_%d.h5' % (hyper['snapshot_prefix'], i)
            logging.info('Saving net to: %s' % filename)
            net.save(filename)
        if i % hyper['graph_interval'] == 0 and i > 0:
            sub = 100
            plt.plot(np.convolve(train_loss_hist, np.ones(sub)/sub)[sub:-sub])
            filename = '%strain_loss.jpg' % hyper['graph_prefix']
            logging.info('Saving figure to: %s' % filename)
            plt.savefig(filename)


