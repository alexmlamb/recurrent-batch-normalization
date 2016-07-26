import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl

from theano.misc import pkl_utils
load = pkl_utils.load


def get_infos(data, grads, show_valid=False, show_grad=False,
              error_name='error_rate', cost_name='cross_entropy'):
    iteration = []
    train = []
    test = []
    valid = []
    train_ce = []
    test_ce = []
    valid_ce = []

    grads_norm = []
    # for g in xrange(len(grads)):
    #     grads_norm.append([])

    nb_epoch = 0

    for i, k in enumerate(data.keys()):
        if data[k]:
            if 'test_'+error_name in data[k]:
                nb_epoch += 1
                iteration.append(i)
                if show_valid:
                    valid.append(data[k][ 'valid_' + error_name])
                    valid_ce.append(data[k]['valid_' + cost_name])
                test.append(data[k]['test_' + error_name])
                train.append(data[k]['train_' + error_name])
                train_ce.append(data[k]['train_' + cost_name])
                test_ce.append(data[k]['test_' + cost_name])
            if 'test_inference_'+ error_name in data[k]:
                nb_epoch += 1
                iteration.append(i)
                if show_valid:
                    valid.append(data[k]['valid_training_' + error_name])
                    valid_ce.append(data[k]['valid_training_' + cost_name])
                test.append(data[k]['test_training_' + error_name])
                train.append(data[k]['train_training_' + error_name])
                train_ce.append(data[k]['train_training_' + cost_name])
                test_ce.append(data[k]['test_training_' + cost_name])
            elif 'valid_training_'+ error_name in data[k]:
                nb_epoch += 1
                iteration.append(i)
                if show_valid:
                    valid.append(data[k]['valid_training_' + error_name])
                    valid_ce.append(data[k]['valid_training_' + cost_name])
                #test.append(data[k]['test_training_' + error_name])
                train.append(data[k]['train_training_' + error_name])
                train_ce.append(data[k]['train_training_' + cost_name])
                #test_ce.append(data[k]['test_training_' + cost_name])
            # if show_grad:
            #     for i, g in enumerate(grads):
            #         grads_norm[i].append(data[k]['iteration_grad_norm:h_'+str(g)])

    return (iteration, train, test, valid,
            train_ce, test_ce, valid_ce, grads_norm)






if __name__ == "__main__":

    if (len(sys.argv) < 2):
        print("%s: file" % sys.argv[0])
        exit(1)

    show_train=True
    show_test=True
    show_valid=True
    show_grad=False
    grads = [290, 250, 200, 150, 100, 50, 10]

    error_name='error_rate'
    #error_name='bpc'
    cost_name='cross_entropy'


    #error_name='MSE'
    #cost_name='MSE'


    datas = []
    names = []
    for i in xrange(1, len(sys.argv)):
        with open(sys.argv[i]) as fd:
            try:
                fd.seek(0)
                data = load(fd)
            except:
                fd.seek(0)
                data = pkl.load(fd)
            print data[0].keys()
            datas.append(data)
            names.append(os.path.basename(os.path.dirname(sys.argv[i])))

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    plt.title("Adding (Training Set)")
    plt.ylabel("MSE")
    plt.xlabel("Training Iteration")
    f2 = plt.figure(2)
    ax2 = f2.add_subplot(111)
    plt.title("Adding (Test Set)")
    plt.ylabel("MSE")
    plt.xlabel("Training Iteration")


    if show_valid:
        f3 = plt.figure(3)
        ax3 = f3.add_subplot(111)
        plt.title("MNIST (Validation Set)")
        plt.ylabel("Accuracy")
        plt.xlabel("Training Iteration")




    print  names
    #colors = [ 'b', 'r', 'm', 'c']
    for i, data in enumerate(datas):
        (iteration,
         train, test, valid,
         train_ce, test_ce, valid_ce,
         grads_norm) = get_infos(data, grads, show_valid, show_grad,
                                 error_name=error_name, cost_name=cost_name)

        # if i == 2:
        #     iteration = np.array(iteration) * 2
        # if i == 3:
        #     iteration = np.array(iteration) * 4
        if show_train:
            ax1.plot(iteration, train, label=names[i], linewidth=3,)
            #color=colors[i])
            #ax1.plot(iteration, train_ce, label=names[i])
        if show_test:
            import pdb; pdb.set_trace()
            ax2.plot(iteration, test, label=names[i], linewidth=3,)
            #color=colors[i])
            #ax2.plot(iteration, test_ce, label=names[i])
        if show_valid:
            ax3.plot(iteration, valid, label=names[i], linewidth=3,)
            #color=colors[i])
            #ax3.plot(iteration, valid_ce, label=names[i])


    ax1.legend(shadow=True, loc=1)
    #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax2.legend(shadow=True, loc=1)
    #bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if show_valid:
        ax3.legend(shadow=True, loc=1)
                  # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()




