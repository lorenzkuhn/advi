import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2

test_accuracies = [[], [], []]
run_time_test_accuracies = [[], [], []]
wall_times = [[], [], []]
test_nlls = [[], [], []]
batch_sizes = [1,5, 10, 20, 50, 100, 200, 250, 500, 1000]

batch_size_for_run_time_comparison = 50
last_vi_step = 49
last_sgmcmc_step = 199

for batch_size in batch_sizes:
    
    # Specifies path to VI results
    path_0 = 'runs/vi/mnist/mfvi_initsigma_0.01__opt_adam__lr_sch_i_0.0001___epochs_{}_wd_5.0_batchsize_{}_temp_1.0__seed_9/'.format(last_vi_step, batch_size)   
    # Specifies path to SGLD results
    path_1 = 'runs/sgmcmc/mnist/sgld_mom_0.0_preconditioner_None__lr_sch_constant_i_3e-08_f_None_c_50_bi_1___epochs_{}_wd_5.0_batchsize_{}_temp_1.0__seed_14/'.format(last_sgmcmc_step, batch_size)
    # Specifies path to SGHMC results
    path_2 = 'runs/sgmcmc/mnist/sgld_mom_0.9_preconditioner_None__lr_sch_constant_i_3e-08_f_None_c_50_bi_1___epochs_{}_wd_5.0_batchsize_{}_temp_1.0__seed_14/'.format(last_sgmcmc_step, batch_size)
        
    for i, path in enumerate([path_0, path_1, path_2]):
        events_file = None
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, 'events.out*'):
                events_file = file

        path += events_file       
        summary_iterator = tf.compat.v1.train.summary_iterator(path)

        for e in list(summary_iterator):
            if (i == 0 and e.step == last_vi_step) or (i >= 1 and e.step == last_sgmcmc_step):
                for v in e.summary.value:
                    if v.tag == 'test/nll':
                        print(np.frombuffer(v.tensor.tensor_content, dtype="float32"))
                        test_nlls[i].append(np.frombuffer(v.tensor.tensor_content, dtype="float32"))

                
                    if v.tag == 'test/accuracy':
                        print(np.frombuffer(v.tensor.tensor_content, dtype="float32"))
                        test_accuracies[i].append(np.frombuffer(v.tensor.tensor_content, dtype="float32"))

            if batch_size == batch_size_for_run_time_comparison:
                wall_times[i].append(e.wall_time)
                for v in e.summary.value:
                    if i == 1:
                        print(v)
                    if v.tag == 'test/accuracy':
                        print(np.frombuffer(v.tensor.tensor_content, dtype="float32"))
                        run_time_test_accuracies[i].append(np.frombuffer(v.tensor.tensor_content, dtype="float32"))


sns.set_style('whitegrid')

wall_times = [np.array(x) for x in wall_times]

wall_times[0] -= wall_times[0][0]
wall_times[1] -= wall_times[1][0]
wall_times[2] -= wall_times[2][0]

sns.lineplot(x=wall_times[0][:5], y=np.concatenate(run_time_test_accuracies[0])[:5], label='VI')
sns.lineplot(x=wall_times[1][:len(run_time_test_accuracies[1])], y=np.concatenate(run_time_test_accuracies[1]), label='SGLD')
sns.lineplot(x=wall_times[2][:len(run_time_test_accuracies[2])], y=np.concatenate(run_time_test_accuracies[2]), label='SGHMC')
plt.title('Run time comparison of ADVI and stochastic MCMC methods')
plt.xlabel('Run time [s]')
plt.ylabel('Test accuracy')
plt.savefig('runtime_comparison.pdf')                 

plt.figure()
sns.lineplot(x=batch_sizes, y=np.concatenate(test_accuracies[0]))
sns.lineplot(x=batch_sizes, y=np.concatenate(test_accuracies[1]))
sns.lineplot(x=batch_sizes, y=np.concatenate(test_accuracies[2]))
plt.title('Subsampling behaviour of ADVI and stochastic MCMC methods')
plt.xlabel('Batch size')
plt.ylabel('Test accuracy')
plt.legend(labels=["ADVI","SGLD", "SGHMC"])
plt.savefig('subsampling_accuracies_mnist.pdf')

plt.figure()
sns.lineplot(x=batch_sizes, y=np.concatenate(test_nlls[0]))
sns.lineplot(x=batch_sizes, y=np.concatenate(test_nlls[1]))
sns.lineplot(x=batch_sizes, y=np.concatenate(test_nlls[2]))
plt.title('Subsampling behaviour of ADVI and stochastic MCMC methods')
plt.xlabel('Batch size')
plt.ylabel('Test NLL')
plt.legend(labels=["ADVI","SGLD", "SGHMC"])
plt.savefig('subsampling_nlls_mnist.pdf')
