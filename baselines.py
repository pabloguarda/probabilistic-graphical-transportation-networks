from dbn import *
from rbm import *
# import dbn
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from keras import backend as K

import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from sklearn.metrics import confusion_matrix


def denseQ(Q: np.matrix, remove_zeros: bool = True):

    q = [] # np.zeros([len(np.nonzero(Q)[0]),1])

    if remove_zeros:
        for i, j in zip(*Q.nonzero()):
            q.append(Q[(i,j)])

    else:
        for i, j in np.ndenumerate(Q):
            q.append(Q[i])

    q = np.array(q)[:,np.newaxis]

    assert q.shape[1] == 1, "od vector is not a column vector"

    return q

def originalQ(new_q: np.matrix, original_Q:np.matrix,removed_zeros = True):
    # q = []  # np.zeros([len(np.nonzero(Q)[0]),1])

    # q = np.zeros(Q.flatten().shape)
    new_Q = np.zeros(original_Q.shape)

    counter = 0
    if removed_zeros:

        for i, j in zip(*original_Q.nonzero()):
            new_Q[(i, j)] = new_q[counter]
            counter+=1

    else:
        for i, j in np.ndenumerate(Q):
            new_Q[(i, j)] = new_q[counter]
            counter += 1

    # q = np.array(q)[:, np.newaxis]

    return new_Q

def generative_bar_plot_link_flows(decoder, Q_ref,Q0, networkname, links_idxs):

    link_flows_Q = decoder.predict(denseQ(Q_ref).T).flatten()[links_idxs-1]
    link_flows_Q0 = decoder.predict(denseQ(Q0).T).flatten()[links_idxs-1]
    #n_links = link_flows_Q.shape[0]
    links_ids = links_idxs #list(np.arange(1,n_links+1))
    # len(list(link_flows))
    # len(list(np.arange(1,n_links)))

    fig = plt.figure()
    plt.bar(links_ids,list(link_flows_Q))
    # plt.ylabel('path flow [veh/hr]')
    plt.xlabel('Link')
    plt.ylabel('Traffic flow [veh/hr]')
    fig.savefig('figures/generative_link_flows_' + networkname +'_Q.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    hist = plt.bar(links_ids,list(link_flows_Q0))
    # plt.ylabel('path flow [veh/hr]')
    plt.xlabel('Link id')
    plt.ylabel('Traffic flow [veh/hr]')
    fig.savefig('figures/vae/generative_link_flows_' + networkname +'_Q0.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # Barplots together

    fig = plt.figure()

    X_axis = np.arange(len(links_ids))

    plt.bar(X_axis - 0.15, list(link_flows_Q), 0.3, label='Reference', color = 'blue')
    plt.bar(X_axis + 0.15, list(link_flows_Q0), 0.3, label='Generated', color = 'red')
    plt.xlabel('Link id')
    plt.ylabel('Traffic flow [veh/hr]')
    plt.xticks(links_ids-1,links_ids)
    plt.legend()
    fig.savefig('figures/generative_link_flows_' + networkname +'.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

def generative_bar_plots_link_flows(decoder, Q_ref,Q0,Q1, networkname, links_idxs):

    link_flows_Q = decoder.predict(denseQ(Q_ref).T).flatten()[links_idxs-1]
    link_flows_Q0 = decoder.predict(denseQ(Q0).T).flatten()[links_idxs-1]
    link_flows_Q1 = decoder.predict(denseQ(Q1).T).flatten()[links_idxs - 1]
    #n_links = link_flows_Q.shape[0]
    links_ids = links_idxs #list(np.arange(1,n_links+1))

    barWidth = 0.25
    br1 = np.arange(len(links_ids))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Barplots together

    fig = plt.figure()

    X_axis = np.arange(len(links_ids))

    plt.bar(br1, list(link_flows_Q), width = barWidth, label='Reference', color = 'blue')
    plt.bar(br2, list(link_flows_Q0),width = barWidth, label='Apartments', color = 'red')
    plt.bar(br3, list(link_flows_Q1), width = barWidth, label='Schools', color='green')
    plt.xlabel('Link id')
    plt.ylabel('Traffic flow [veh/hr]')
    plt.xticks(links_ids-1,links_ids)
    plt.legend()
    fig.savefig('figures/vae/generative_link_flows_' + networkname +'.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

def plot_histogram_path_flows(test_X, networkname,encoder, decoder, F = None,truth = True):

    od_flows = encoder.predict(test_X)
    path_flows = decoder.layers[1](decoder.layers[0](od_flows)).numpy()


    fig = plt.figure()
    hist = plt.hist(path_flows.flatten())
    # plt.ylabel('path flow [veh/hr]')
    plt.xlabel('path flow [veh/hr]')
    plt.ylabel('frequency')
    fig.savefig('figures/vae/predicted_path_flows_' + networkname +'.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # if truth is True:
    # True path flows
    path_flows_true = F.flatten()#[(F.flatten()<2000)]

    fig = plt.figure()
    hist = plt.hist(path_flows_true)
    plt.xlabel('path flow [veh/hr]')
    plt.ylabel('frequency')
    plt.savefig('figures/vae/true_path_flows_' + networkname + '.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # if truth is True:
    fig = plt.figure()
    hist = plt.hist([path_flows_true.flatten(),path_flows.flatten()], label=['true', 'predicted'])
    plt.xlabel('path flow [veh/hr]')
    plt.ylabel('frequency')
    plt.legend(loc='upper right')
    plt.savefig('figures/vae/path_flows_' + networkname + '.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # plt.show()
    # np.max(train_X_paths)
    #
    # np.mean(np.abs(train_X_paths-np.mean(train_X_paths)))

def plot_histogram_link_flows(test_X, X,encoder, decoder, networkname):

    od_flows = encoder.predict(test_X)
    predicted_link_flows = decoder.predict(od_flows)

    # predicted_link_flows_high = predicted_link_flows.flatten()#[(path_flows.flatten()>1000)]

    # np.sum(path_flows)
    #
    # np.mean(np.abs(path_flows-train_X_paths))

    fig = plt.figure()
    hist = plt.hist(predicted_link_flows.flatten())
    # plt.ylabel('path flow [veh/hr]')
    plt.xlabel('link flow [veh/hr]')
    plt.ylabel('frequency')
    fig.savefig('figures/vae/predicted_link_flows_' + networkname +'.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # True path flows
    link_flows_true = X#[(train_X_paths.flatten()>10000)]

    fig = plt.figure()
    hist = plt.hist(link_flows_true.flatten())
    plt.xlabel('link flow [veh/hr]')
    plt.ylabel('frequency')
    plt.savefig('figures/vae/predicted_path_flows_' + networkname + '.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()


    fig = plt.figure()
    hist = plt.hist([link_flows_true.flatten(),predicted_link_flows.flatten()], label=['true', 'predicted'])
    plt.xlabel('link flow [veh/hr]')
    plt.ylabel('frequency')
    plt.legend(loc='upper right')
    plt.savefig('figures/vae/link_flows_' + networkname + '.pdf',bbox_inches='tight')
    fig.tight_layout()
    plt.show()

    # plt.show()
    # np.max(train_X_paths)
    #
    # np.mean(np.abs(train_X_paths-np.mean(train_X_paths)))


# Global parameters
EPOCHS = 10

# Read data
link_flows_df = pd.read_csv('data/sioux-falls/link_flows.csv')
paths_flows_df = pd.read_csv('data/sioux-falls/path_flows.csv')
demand_df = pd.read_csv('data/sioux-falls/demand.csv')

n = len(link_flows_df.columns)-1

# Convert dataframes to numpy
Q = np.array(demand_df.iloc[:,list(np.arange(1,n+1))]).T
X = np.array(link_flows_df.iloc[:,list(np.arange(1,n+1))]).T
F = np.array(paths_flows_df.iloc[:,list(np.arange(1,n+1))]).T

nodes = int(np.sqrt(Q.shape[1]))

Q = Q.reshape((n,nodes,nodes))

# Get dense vector of matrices Q
qs = np.array([denseQ(Q[i], remove_zeros=True).flatten() for i in range(Q.shape[0])])

# Incidence matrices
C = np.genfromtxt('data/sioux-falls/C-SiouxFalls.csv', delimiter=',')
D = np.genfromtxt('data/sioux-falls/D-SiouxFalls.csv', delimiter=',')
M = np.genfromtxt('data/sioux-falls/M-SiouxFalls.csv', delimiter=',')

# General parameters for training
mb_size = 16
# epochs = 1000
train_valid_split = 0.7

# Data processing
data = X
idxs = np.arange(0,data.shape[0])
np.random.shuffle(idxs)
data = data[idxs,:]


train_threshold = int(train_valid_split*n)
train_idxs = idxs[:train_threshold]
test_idxs =  idxs[train_threshold:]

train_X = data[train_idxs, :]
valid_X = data[test_idxs, :]

# Plot functions
def heatmap_OD(Q,filepath):
    Q_plot = Q.reshape((24, 24))
    rows, cols = Q_plot.shape

    od_df = pd.DataFrame({'origin': pd.Series([], dtype=int)
                             , 'destination': pd.Series([], dtype=int)
                             , 'trips': pd.Series([], dtype=int)})

    counter = 0
    for origin in range(0, rows):
        for destination in range(0, cols):
            # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
            od_df.loc[counter] = [int(origin + 1), int(destination + 1), Q_plot[(origin, destination)]]
            counter += 1

    od_df.origin = od_df.origin.astype(int)
    od_df.destination = od_df.destination.astype(int)

    # od_df = od_df.groupby(['origin', 'destination'], sort=False)['trips'].sum()

    od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')

    # uniform_data = np.random.rand(10, 12)
    fig, ax = plt.subplots()
    ax = sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues")
    # plt.show()

    fig.savefig(filepath)

    plt.show()



# BASELINE MODELS

# Normalize data between 0 and 1
normalization_vector = np.amax(train_X, axis = 1, keepdims = True)
train_X  = train_X  / np.amax(train_X, axis = 1, keepdims = True)
valid_X  = valid_X  / np.amax(valid_X, axis = 1, keepdims = True)

train_X = binary_data(train_X)
valid_X = binary_data(valid_X)

# RBM
rbm_1 = RBM(n_visible=X.shape[1], n_hidden = 576, k=1, lr=0.005, minibatch_size=mb_size)

# rbm_1.W = np.random.normal(0, 0.1, (rbm_1.n_hidden, rbm_1.n_visible))

train_errors_rbm,test_errors_rbm = rbm_1.train(trainX = train_X, testX = valid_X, epochs = EPOCHS)

od_pred_valid = rbm_1.sample_h(valid_X)[0]
binary_x_pred_valid, x_pred_valid = rbm_1.sample_v(od_pred_valid)

od_pred_train = rbm_1.sample_h(train_X)[0]

# Confusion matrix

Q_binary = (Q>0.1).astype(int)

y_actu = Q_binary[train_idxs, :].flatten()
y_pred = od_pred_train.flatten()
confusion_matrix(y_actu, y_pred)

y_actu = pd.Series(list(Q_binary[train_idxs, :].flatten()), name='Actual')
y_pred = pd.Series(list(od_pred_train.flatten()), name='Predicted')
df_confusion_rbm = pd.crosstab(y_actu, y_pred)
df_confusion_rbm = df_confusion_rbm/ np.array(df_confusion_rbm.sum(axis=1)).reshape(2,1)
df_confusion_table_rbm = round(100*df_confusion_rbm,2)
df_confusion_table_rbm.to_csv('tables/confusion_matrix_rbm' + '.csv', sep=',', encoding='utf-8')

def plot_confusion_matrix(df_confusion, filepath, cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

    plt.savefig(filepath)

plot_confusion_matrix(df_confusion_rbm, 'figures/rbm/confusion_matrix_rbm.pdf')

plt.show()

# Plots
epochs_x = np.arange(len(train_errors_rbm))

fig = plt.figure()

plt.plot(epochs_x, train_errors_rbm, label="Train error", color='black')
plt.plot(epochs_x, test_errors_rbm, label="Validation error", color='blue')

plt.xlabel('epoch')
plt.ylabel('reconstruction error')
plt.legend()

plt.savefig('figures/rbm/errors_rbm.pdf')

plt.show()

plt.close(fig)

# Heatmap with OD Matrices

## Original
# true_OD = (np.mean(Q[train_idxs,],axis = 0)>0.1).astype(int)
# true_OD = (np.mean(Q[train_idxs,],axis = 0)>0.5).astype(int)
# true_OD = (np.mean(Q_binary[train_idxs,],axis = 0)>0.5).astype(int)
# heatmap_OD(true_OD,filepath = 'figures/rbm/OD_heatmap_true.pdf')

true_OD = Q_binary[1,:]
heatmap_OD(true_OD,filepath = 'figures/rbm/OD_heatmap_true.pdf')

# heatmap_OD(Q_binary[0],filepath = 'figures/rbm/OD_heatmap_true.pdf')
# Prediction
# heatmap_OD((np.mean(od_pred_train,axis = 0)>0.5).astype(int).reshape(24,24),filepath = 'figures/rbm/OD_heatmap_rbm.pdf')
heatmap_OD(od_pred_train[1].reshape(24,24),filepath = 'figures/rbm/OD_heatmap_rbm.pdf')

# DBN
dbn = DBN(n_v=X.shape[1], layers=[X.shape[1], F.shape[1]], k = 1, lr=0.01, mb_size=mb_size)

train_errors_rbm1, train_errors_rbm2, valid_errors_rbm1, valid_errors_rbm2, od_pred = dbn.train(train_X, valid_X, epochs=EPOCHS)

od_pred_train_dbn = dbn.rbm_2.sample_h(dbn.rbm_1.sample_h(train_X)[0])[0]
od_pred_valid_dbn = dbn.rbm_2.sample_h(dbn.rbm_1.sample_h(valid_X)[0])[0]


# Plots
epochs_x = np.arange(len(train_errors_rbm1))

fig = plt.figure()

plt.plot(epochs_x, train_errors_rbm1, label="Train error", color='black')
plt.plot(epochs_x, valid_errors_rbm1, label="Validation error", color='blue')

plt.xlabel('epoch')
plt.ylabel('reconstruction error')
plt.legend()

plt.savefig('figures/dbn/errors_dbn.pdf')

plt.show()

plt.close(fig)


## Original

# true_OD = (np.mean(Q[train_idxs,],axis = 0)>0.5).astype(int)
true_OD = Q_binary[1,:]
heatmap_OD(true_OD,filepath = 'figures/dbn/OD_heatmap_true.pdf')
plt.show()
# Prediction
heatmap_OD(od_pred_train_dbn[1].reshape((24,24)),filepath = 'figures/dbn/OD_heatmap_dbn.pdf')
# heatmap_OD((np.mean(od_pred_train_dbn,axis = 0)>0.5).astype(int).reshape(24,24),filepath = 'figures/dbn/OD_heatmap_dbn.pdf')


# Confusion matrix
y_actu = pd.Series(list(Q_binary[train_idxs, :].flatten()), name='Actual')
y_pred = pd.Series(list(od_pred_train_dbn.flatten()), name='Predicted')
df_confusion_dbn = pd.crosstab(y_actu, y_pred)
df_confusion_dbn = pd.crosstab(y_actu, y_pred) / np.array(df_confusion_dbn.sum(axis=1)).reshape(2,1)
df_confusion_table_dbn = round(100*df_confusion_dbn,2)
df_confusion_table_dbn.to_csv('tables/confusion_matrix_dbn' + '.csv', sep=',', encoding='utf-8')

plot_confusion_matrix(df_confusion_dbn, 'figures/dbn/confusion_matrix_dbn.pdf')

plt.show()










