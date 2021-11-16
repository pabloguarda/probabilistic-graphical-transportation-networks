from dbn import *
from rbm import *
# import dbn
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix

# Read data
link_flows_df = pd.read_csv('data/link_flows.csv')
paths_flows_df = pd.read_csv('data/path_flows.csv')
demand_df = pd.read_csv('data/demand.csv')

n = len(link_flows_df.columns)-1

# Convert dataframes to numpy

Q = np.array(demand_df.iloc[:,list(np.arange(1,n+1))]).T
X = np.array(link_flows_df.iloc[:,list(np.arange(1,n+1))]).T
F = np.array(paths_flows_df.iloc[:,list(np.arange(1,n+1))]).T

# General parameters for training
mb_size = 16
epochs = 100
train_valid_split = 0.7

# Data processing
data = X
np.random.shuffle(data)

# Normalize data between 0 and 1
normalization_vector = np.amax(data, axis = 1, keepdims = True)

data = data  / normalization_vector

train_idx = int(train_valid_split*n)

train_X = data[:train_idx, :]
valid_X = data[train_idx:, :]

train_X = binary_data(train_X)
valid_X = binary_data(valid_X)


# RBM
rbm_1 = RBM(n_visible=X.shape[1], n_hidden = 576, k=1, lr=0.005, minibatch_size=mb_size)

# rbm_1.W = np.random.normal(0, 0.1, (rbm_1.n_hidden, rbm_1.n_visible))

train_errors_rbm,test_errors_rbm = rbm_1.train(trainX = train_X, testX = valid_X, epochs = epochs)

od_pred_valid = rbm_1.sample_h(valid_X)[0]
binary_x_pred_valid, x_pred_valid = rbm_1.sample_v(od_pred_valid)

od_pred_train = rbm_1.sample_h(train_X)[0]

# Confusion matrix
Q_binary = (Q>0.1).astype(int)

y_actu = Q_binary[:train_idx, :].flatten()
y_pred = od_pred_train.flatten()
confusion_matrix(y_actu, y_pred)

y_actu = pd.Series(list(Q_binary[:train_idx, :].flatten()), name='Actual')
y_pred = pd.Series(list(od_pred_train.flatten()), name='Predicted')
df_confusion_rbm = pd.crosstab(y_actu, y_pred)
df_confusion_rbm = df_confusion_rbm/ df_confusion_rbm.sum(axis=1)
df_confusion_rbm.to_csv('tables/confusion_matrix_rbm' + '.csv', sep=',', encoding='utf-8')

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

plt.close(fig)

# Heatmap with OD Matrices

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

## Original
heatmap_OD(Q_binary[0],filepath = 'figures/rbm/OD_heatmap_true.pdf')
# Prediction
heatmap_OD(od_pred_train[0].reshape((24,24)),filepath = 'figures/rbm/OD_heatmap_rbm.pdf')



# DBN
dbn = DBN(n_v=X.shape[1], layers=[X.shape[1], F.shape[1]], k = 1, lr=0.01, mb_size=mb_size)

train_errors_rbm1, train_errors_rbm2, valid_errors_rbm1, valid_errors_rbm2, od_pred = dbn.train(train_X, valid_X, epochs=epochs)

od_pred[:train_idx, :]
Q.round()

od_pred_train_dbn = dbn.rbm_2.sample_h(dbn.rbm_1.sample_h(train_X)[0])[0]
od_pred_valid_dbn = dbn.rbm_2.sample_h(dbn.rbm_1.sample_h(valid_X)[0])[0]

# x_pred_train_dbn = dbn.sample_v(train_X, k =1)[0]



print(np.sum((Q_binary[:train_idx, :]-od_pred_train_dbn != 0).astype(int))/od_pred_train_dbn.size)
print(np.sum((Q_binary[train_idx:, :]-od_pred_valid_dbn != 0).astype(int))/od_pred_valid_dbn.size)


# Plots
epochs_x = np.arange(len(train_errors_rbm1))

fig = plt.figure()

plt.plot(epochs_x, train_errors_rbm1, label="Train error", color='black')
plt.plot(epochs_x, valid_errors_rbm1, label="Validation error", color='blue')

plt.xlabel('epoch')
plt.ylabel('reconstruction error')
plt.legend()

plt.savefig('figures/dbn/errors_dbn.pdf')

plt.close(fig)


## Original
heatmap_OD(Q_binary[0],filepath = 'figures/dbn/OD_heatmap_true.pdf')
# Prediction
heatmap_OD(od_pred_train_dbn[0].reshape((24,24)),filepath = 'figures/dbn/OD_heatmap_dbn.pdf')


# Confusion matrix
y_actu = pd.Series(list(Q_binary[:train_idx, :].flatten()), name='Actual')
y_pred = pd.Series(list(od_pred_train_dbn.flatten()), name='Predicted')
df_confusion_dbn = pd.crosstab(y_actu, y_pred)
df_confusion_dbn = pd.crosstab(y_actu, y_pred) / df_confusion_dbn.sum(axis=1)
df_confusion_dbn.to_csv('tables/confusion_matrix_dbn' + '.csv', sep=',', encoding='utf-8')
plot_confusion_matrix(df_confusion_dbn, 'figures/dbn/confusion_matrix_dbn.pdf')