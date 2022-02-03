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


# VAE
# source: https://www.tensorflow.org/probability/examples/Probabilistic_Layers_VAE
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

BATCH_SIZE = 64 #int(train_X.shape[0]/2) #64
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 100

train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_X))
test_dataset = tf.data.Dataset.from_tensor_slices((valid_X , valid_X))

# Normalize
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_X))

# Masked layers

class LinksToPathsFlowsLayer(tf.keras.layers.Layer):
  def __init__(self,D):
    super(LinksToPathsFlowsLayer, self).__init__()
    self.D = tf.convert_to_tensor(D, dtype = tf.float32)
    # self.num_inputs = M.shape[0]
    # self.num_outputs = M.shape[1]

  def build(self, input_shape):
    self.input_dim = input_shape
    self.kernel = self.add_weight("kernel",
                                  shape=[int(self.D.shape[0]),
                                         int(self.D.shape[1])],dtype = tf.float32)

  def call(self, inputs):
      #TODO: forced sum of weights going out from a link to be equal to 1
    return tf.tensordot(inputs,tf.multiply(self.kernel,self.D),axes = 1)

class PathsToODFlowsLayer(tf.keras.layers.Layer):
  def __init__(self,M):
    super(PathsToODFlowsLayer, self).__init__()
    self.M = tf.convert_to_tensor(M, dtype = tf.float32)
    self.num_inputs = M.shape[1]
    self.num_outputs = M.shape[0]

  def build(self, input_shape):

    self.input_dim = input_shape
    self.kernel = self.add_weight("kernel",
                                  shape=[int(M.shape[0]),
                                         M.shape[1]],dtype = tf.float32)

  def call(self, inputs):
    return tf.tensordot(inputs,tf.transpose(tf.multiply(self.M,self.kernel)),axes = 1)
  # return tf.tensordot(inputs,tf.nn.softmax(tf.multiply(mask,self.kernel),axis = 1),axes = 1)

class ODtoPathsFlowsLayer(tf.keras.layers.Layer):
  def __init__(self, C,M):
    super(ODtoPathsFlowsLayer, self).__init__()
    self.num_outputs = M.shape[1]
    self.num_inputs = M.shape[0]
    self.C = tf.convert_to_tensor(C, dtype = tf.float32)
    self.M = tf.convert_to_tensor(M, dtype=tf.float32)


  def build(self, input_shape):

    self.input_dim = input_shape
    self.kernel = self.add_weight("kernel",
                                  shape=[int(self.num_inputs),
                                         self.num_outputs],dtype = tf.float32)

  def call(self, inputs):


    inputs = tf.cast(inputs, tf.float32)

    vf = tf.multiply(self.M,self.kernel)

    exp_vf = tf.exp(vf) #.float()
    # v = np.exp(np.sum(V_Z, axis=1) + V_Y)

    # Denominator logit functions
    sum_exp_vf = tf.tensordot(self.C, tf.transpose(exp_vf), axes = 1)
    # sum_exp_vf = torch.clamp(torch.mm(torch.tensor(C).float(),exp_vf.t()),-1e7,1e7)

    epsilon = 0  # 1e-3

    p_f = tf.divide(exp_vf, (tf.transpose(sum_exp_vf) + epsilon))
    p_f = tf.multiply(self.M,p_f)

    f = tf.tensordot(inputs, p_f, axes=1)

    # return tf.tensordot(inputs,tf.nn.softmax(tf.multiply(mask,self.kernel),axis = 1),axes = 1)

    return f

class PathsToLinksFlowsLayer(tf.keras.layers.Layer):
  def __init__(self,D):
    super(PathsToLinksFlowsLayer, self).__init__()
    self.D = tf.convert_to_tensor(D, dtype = tf.float32)



  def build(self, input_shape):

    self.input_dim = input_shape
    # self.kernel = self.add_weight("kernel",
    #                               shape=[int(self.num_inputs),
    #                                      self.num_outputs],dtype = tf.float32)

  def call(self, inputs):
    return tf.tensordot(inputs,tf.transpose(self.D),axes = 1)

pathstoodflows_layer = PathsToODFlowsLayer(M=M)

pathstoodflows_layer(tf.convert_to_tensor(np.arange(0,M.shape[1])[np.newaxis,:], dtype = tf.float32)).shape

linktopathsflows_layer = LinksToPathsFlowsLayer(D=D)

linktopathsflows_layer(tf.convert_to_tensor(np.arange(0,D.shape[0])[np.newaxis,:], dtype = tf.float32)).shape

odtopathsflows_layer = ODtoPathsFlowsLayer(C=C, M = M)
pathstolinkflows_layer = PathsToLinksFlowsLayer(D=D)

# odtopathsflows_layer(M[0,:])

odtopathsflows_layer(tf.convert_to_tensor(np.arange(0,M.shape[0])[np.newaxis,:], dtype = tf.float32)).shape

pathstolinkflows_layer(tf.convert_to_tensor(np.arange(0,D.shape[1])[np.newaxis,:], dtype = tf.float32))
inputs = tf.convert_to_tensor(np.arange(0,D.shape[1])[np.newaxis,:], dtype = tf.float32)

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# input_shape = (train_X.shape[1],1)
# input_shape = (1,train_X.shape[1])
input_shape = (train_X.shape[1],)
encoded_size = qs[0].shape[0] # q_size
path_flow_size = F.shape[1] #300

KL_WEIGHT = 0 #10 # 100
# LR = 5e-3
LR = 1e-1


# TODO: A different prior maybe set to better match the distribution of the original OD matrix. I may set a good prior for std as well

# prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
#                         reinterpreted_batch_ndims=1)

# prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=300),
#                         reinterpreted_batch_ndims=1)

# prior = tfd.Independent(tfd.Normal(loc=tf.ones(encoded_size)*np.mean(Q[test_idxs,]), scale=tf.constant(np.std(Q[test_idxs,]), dtype='float32')),reinterpreted_batch_ndims=1)

# prior = tfd.Independent(tfd.Normal(loc=tf.ones(encoded_size)*np.mean(Q[test_idxs,]), scale=tf.constant(np.std(Q[test_idxs,]), dtype='float32')),reinterpreted_batch_ndims=1)

# prior = tfd.Independent(tfd.Normal(loc=tf.convert_to_tensor(np.mean(Q[train_idxs,],axis = 0),dtype='float32'), scale=tf.constant(np.std(Q[train_idxs,]), dtype='float32')),reinterpreted_batch_ndims=1)

prior = tfd.Independent(tfd.Normal(loc=tf.convert_to_tensor(1*np.mean(qs,axis = 0),dtype=tf.float32), scale=tf.constant(np.std(qs), dtype=tf.float32)),reinterpreted_batch_ndims=1)

# prior = tfd.Independent(tfd.Normal(loc=tf.convert_to_tensor(0.3*np.mean(Q[train_idxs,],axis = 0),dtype='float32'), scale=tf.constant(np.std(Q[train_idxs,]), dtype='float32')),reinterpreted_batch_ndims=1)

# prior = tfd.Independent(tfd.Poisson(tf.zeros(encoded_size)),
#                         reinterpreted_batch_ndims=1)

encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    normalizer,
    # tfkl.Lambda(lambda x: tf.cast(x, tf.float32) - 0.5),
    tfkl.Flatten(),
    tfkl.Dense(path_flow_size,
               activation='relu'),
    # LinksToPathsFlowsLayer(D),
    # tfkl.ReLU(),
    # PathsToODFlowsLayer(M),
    # tfkl.Flatten(),
    # tfkl.ReLU(),
    # tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size), activation=None),
    # tfpl.MultivariateNormalTriL(
    #     encoded_size,
    #     activity_regularizer=tfpl.KLDivergenceRegularizer(prior))

    # the following two lines set the output to be a probabilistic distribution (independent normal is less expensive)
    tfkl.Dense(tfpl.IndependentNormal.params_size(encoded_size),
               activation='relu', name='z_params'),
    tfpl.IndependentNormal(encoded_size,
                           convert_to_tensor_fn=tfd.Distribution.sample,
                           activity_regularizer=tfpl.KLDivergenceRegularizer(prior,weight=KL_WEIGHT),
                           name='z_layer'),
    tfkl.ReLU()
    # tfkl.Dense(tfpl.IndependentPoisson.params_size(encoded_size),
    #            activation='relu', name='z_params'),
    # tfpl.IndependentPoisson(encoded_size,
    #                        convert_to_tensor_fn=tfd.Distribution.sample,
    #                        activity_regularizer=tfpl.KLDivergenceRegularizer(prior,
    #                                                                          weight=KL_WEIGHT),
    #                        name='z_layer')
])

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    # tfkl.Reshape([1, 1, encoded_size]),
    tfkl.Flatten(),
    ODtoPathsFlowsLayer(C,M),
    # tfkl.Dense(path_flow_size, activation='relu'),
    # tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
    # tfpl.IndependentNormal(input_shape),
    # tfkl.Dense(input_shape[0], activation='relu'),
    PathsToLinksFlowsLayer(D),
    # pathstolinkflows_layer,
    tfkl.ReLU()
])

vae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))

vae.summary()


# negloglik = lambda x, rv_x: -rv_x.log_prob(x)

# Gaussian likelihood (i.e. MSE) to address continuous outocome
# https://www.kaggle.com/maunish/training-vae-on-imagenet-pytorch
prior_variance = np.std(Q[train_idxs,])**2
negloglik = lambda x, rv_x: (x-rv_x)**2/(2*prior_variance)

# negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=LR),
            loss = negloglik
            )
            # loss=negloglik)


# vae.compile(optimizer=tf.optimizers.Adam(learning_rate=LR),
#             loss = tfk.losses.MeanSquaredError()
#             )
#             # loss=negloglik)

history = vae.fit(train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset)

# history = _

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  # plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

plot_loss(history)
plt.savefig(fname = 'figures/vae/loss.pdf')

plt.show()

# Prediction
pred = vae.predict(test_dataset)

pred.shape

test_data_x = np.hstack(list(test_dataset))[1]
_ = plt.hist(test_data_x-pred)
plt.show()

# RMSE is reasonable
print(np.sqrt(np.sum((test_data_x-pred)**2))/test_data_x.shape[0])
print(np.mean(test_data_x))

# - Get values from estimated OD
model = vae  # include here your original model

od_estimated = encoder.predict(test_data_x)

# Project to non negative orthant
od_estimated = np.maximum(0,od_estimated)

np.max(od_estimated)


# _ = plt.hist(od_estimated)
# plt.show()
#
# _ = plt.hist(Q[test_idxs,])
# plt.show()

#Heatmaps
# heatmap_OD(od_estimated[0],filepath = 'figures/vae/OD_heatmap_predicted.pdf')
# heatmap_OD(Q[test_idxs,][0],filepath = 'figures/vae/OD_heatmap_true.pdf')

# Get original Q matrices

test_Qs = np.array([originalQ(original_Q = Q[j,:,:], new_q = od_estimated[i]) for i,j in zip(range(len(test_idxs)),test_idxs)])

heatmap_OD(np.mean(test_Qs,axis =0),filepath = 'figures/vae/OD_heatmap_predicted.pdf')
plt.show()

# True ODS
heatmap_OD(np.mean(Q[test_idxs,],axis =0),filepath = 'figures/vae/OD_heatmap_true.pdf')
plt.show()

np.mean(Q[test_idxs,])
np.std(Q[test_idxs,])
# od_estimated.shape
# print(np.sqrt(np.sum((od_estimated-Q[test_idxs,])**2))/od_estimated.shape[1])
np.mean(od_estimated)
np.std(od_estimated)

# Worst Loss
print(np.sqrt(np.sum(od_estimated**2))/od_estimated.shape[1])

print(np.sqrt(np.sum((Q[test_idxs,]-np.mean(Q[test_idxs,]))**2))/od_estimated.shape[1])

Q.max()

vae.summary()


# Predict link and path flows

plot_histogram_path_flows(test_X = valid_X, F = F[test_idxs], networkname = 'sioux', decoder = decoder, encoder = encoder)
plot_histogram_link_flows(test_X = valid_X, X = X[test_idxs], networkname = 'sioux', decoder = decoder, encoder = encoder)


# Generative process
Q0 = Q[0,:].copy()

# - FLow exiting forom node 13 (new apartments)
node = 1
Q0[node-1,:] = Q0[node-1,:]*3
# Q0=np.ones(Q.shape)
# generative_bar_plot_link_flows(decoder = decoder, Q0 = Q0, Q_ref = Q[0,:], networkname = 'Sioux_E1', links_idxs = np.arange(1,16))

# - Flow entering node 1 increase (new school)
Q1= Q[0,:].copy()
node = 1
Q1[:,node-1] = Q1[:,node-1]*3

# generative_bar_plot_link_flows(decoder = decoder, Q0 = Q1, Q_ref = Q[0,:], networkname = 'Sioux_E2', links_idxs = np.arange(1,16))

# Bar plots altogether

generative_bar_plots_link_flows(decoder = decoder, Q1 = Q1, Q0 = Q0, Q_ref = Q[0,:], networkname = 'Sioux_E', links_idxs = np.arange(1,16))

# TODO: It may be great to plot a graph where the new link flow shows the intensity of the flow
# TODO: effect of congestion is higher when the flows exiting from the node increase










