# Used for Test 4 d) only
# train with datasets B0005, B0006, B0007 and test on B0018 only
# merges feature data together, normalises data, randomises and splits it, then passes through the GPR model
# training and evaluation code taken from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e


import torch
import gpytorch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot




# ------------------------- Sorting data input ------------------------- #
# pklfiles = ['dataset1.pkl', 'dataset2.pkl', 'dataset3.pkl', 'dataset4.pkl']
pklfiles = ['dataset4.pkl', 'dataset3.pkl', 'dataset2.pkl', 'dataset1.pkl']

frames = []

for filename in pklfiles:
    # import dataframes from main
    with open(filename, 'rb') as handle:
        frames1 = pickle.load(handle)

    frames.append(frames1)

# ------------------------- Sorting data input ------------------------- #
df1 = frames[0][0]
df2 = frames[0][1]
df3 = frames[0][2]
df4 = frames[1][0]
df5 = frames[1][1]
df6 = frames[1][2]
df7 = frames[2][0]
df8 = frames[2][1]
df9 = frames[2][2]
df10 = frames[3][0]
df11 = frames[3][1]
df12 = frames[3][2]

frameset1 = [df1, df2, df3]
frameset2 = [df4, df5, df6]
frameset3 = [df7, df8, df9]
frameset4 = [df10, df11, df12]

allframes = [frameset1, frameset2, frameset3, frameset4]
dfcomb_final = pd.DataFrame()

# combining all dataframes
for frameset in allframes:
    dfcomb = pd.concat(frameset, axis=1)
    print(dfcomb.shape)

    dfcomb = dfcomb.drop(['SOH charge cycles', 'SOH discharge cycles', 'SOH discharge cycles 2', 'SOH discharge 2'], axis=1)
    print(dfcomb.shape)

    # fixing issue of value stored as lists:
    dfcomb = dfcomb.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

    # find average SOH for charge and discharge, then remove those 2 columns
    dfcomb['Average SOH'] = dfcomb[['SOH charge', 'SOH discharge']].mean(axis=1)
    dfcomb = dfcomb.drop(['SOH charge', 'SOH discharge'], axis=1)

    dfcomb_final = pd.concat([dfcomb, dfcomb_final], axis=0)

# # saving dataframe to use to plot at end of script:
# with open('dfcomb_final.pkl', 'wb') as handle:
#     pickle.dump(dfcomb_final, handle, protocol=pickle.HIGHEST_PROTOCOL)


# remove NaNs from the whole dataset:
print('Shape before:', dfcomb_final.shape)



# dfcomb_final = dfcomb_final[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'Max ICA', 'Max ICA voltage', 'Av volt discharge', 'Average SOH']]
# dfcomb_final = dfcomb_final.drop(['Max ICA of 2nd peak', 'Max ICA of 2nd peak voltage', 'Max ICA (2nd)', 'Max ICA voltage (2nd)', 'Max temp time'], axis=1)

# --------****** showing improvements, with 1) areas, and 2) max temperature time and 3) areas and max temperature time ******----------

# dfcomb_final = dfcomb_final[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'Max ICA', 'Max ICA voltage', 'Av volt discharge', 'area1', 'area2', 'area3', 'area4', 'Average SOH']]
# dfcomb_final = dfcomb_final[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'Max ICA', 'Max ICA voltage', 'Av volt discharge', 'Max temp time', 'Average SOH']]
dfcomb_final = dfcomb_final[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'Max ICA', 'Max ICA voltage', 'Av volt discharge', 'area1', 'area2', 'area3', 'area4', 'Max temp time', 'Average SOH']]



print('No. NaN values: ', dfcomb_final.isnull().sum().sum())

dfcomb_final = dfcomb_final.dropna(axis=0, how='any')

print('Shape after:', dfcomb_final.shape)

# saving dataframe to use to plot at end of script:
with open('dfcomb_final.pkl', 'wb') as handle:
    pickle.dump(dfcomb_final, handle, protocol=pickle.HIGHEST_PROTOCOL)


# normalising data:
scaler = MinMaxScaler()

col_names = list(dfcomb_final.columns)
row_num = list(dfcomb_final.index)

dfcomb_final_scaled = scaler.fit_transform(dfcomb_final.to_numpy())
dfcomb_final = pd.DataFrame(dfcomb_final_scaled, columns=col_names, index=row_num)





#                                               get the x and y data:

# get three different dataframes and randomise order of rows

# for B0018:

df_training = dfcomb_final[:-129]
df_test = dfcomb_final.tail(129)


df_training = df_training.sample(frac=1, random_state=22)
df_test = df_test.sample(frac=1, random_state=22)


# X and y data split for testing and training

X_train = df_training.drop('Average SOH', axis=1)
y_train = df_training['Average SOH']

X_test = df_test.drop('Average SOH', axis=1)
y_test = df_test['Average SOH']


print('yeee')


# ------------------------- Feature selection ------------------------- #

# FIRST METHOD - https://machinelearningmastery.com/feature-selection-for-regression-data/

# configure to select all features
fs = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)


# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xticks([i for i in range(len(fs.scores_))], X_train.columns.values, rotation='vertical')
pyplot.ylabel('Correlation feature importance')
pyplot.show()







# ------------------------- GPR ------------------------- #
torch.manual_seed(0)

class ExactGPModel(gpytorch.models.ExactGP):
#     # Initialize the parameter
    def __init__(self, X_train_tensor, y_train_tensor, likelihood):
        super(ExactGPModel, self).__init__(X_train_tensor, y_train_tensor, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



# convert to pytorch tensors:

# convert X_train and X_test to numpy arrays

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# training and test data
# method 1:
# train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=100)

# test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=100)


# method 2:
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=100)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=100)

# method 3:
# train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=100)
# test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=100)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()


# Training model test:
num_epochs = 120
training_losses = []
validation_losses = []

# GPytorch documentation method for training:
# for epoch in range(num_epochs):
#     # Zero gradients from previous iteration
#     optimizer.zero_grad()
#     # Output from model
#     pred = model(X_train_tensor)
#     loss = -loss_fn(pred, y_train_tensor)
#     loss.mean().backward()
#     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#         epoch + 1, num_epochs, loss.mean().item(),
#         model.covar_module.base_kernel.lengthscale.item(),
#         model.likelihood.noise.item()
#     ))
#     optimizer.step()



val_results = []
observed_pred = []
trainmse = []
valmse = []

for epoch in range(num_epochs):
    batch_loss = []
    train_batchloss = []

    for X, y in train_dataloader:
        model = ExactGPModel(X, y, likelihood)

        # get MSE before training:
        model.eval()
        untrained_pred = likelihood(model(X))
        train_mse = gpytorch.metrics.mean_squared_error(untrained_pred, y, squared=True)
        train_batchloss.append(train_mse.mean().item())

        # training mode
        model.train()
        likelihood.train()

        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-2)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        # training losses:
        optimizer.zero_grad()
        model.train()

        pred = model(X)

        loss = -loss_fn(pred, y)
        batch_loss.append(loss.mean().item())
        loss.sum().backward()
        optimizer.step()
    training_loss = np.mean(batch_loss)
    training_losses.append(training_loss)

    train_loss_new = np.mean(train_batchloss)
    trainmse.append(train_loss_new)

    # Validation:

    val_results = []

    with torch.no_grad():
        val_losses = []
        model.eval()
        likelihood.eval()
        for X, y in test_dataloader:

            outputs = model(X)

            # val_results.append(outputs.numpy())
            observed_pred = likelihood(model(X))
            # get MSE:
            val_mse = gpytorch.metrics.mean_squared_error(observed_pred, y, squared=True)
            observed_pred = observed_pred.mean.reshape(-1, 1)
            val_results.append(observed_pred.numpy())

            val_loss = -loss_fn(outputs, y)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

        valmse.append(val_mse.numpy())


    val_results = np.concatenate(val_results, axis=0)

    print(f"[{epoch+1}] Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}\t Training MSE: {train_mse:.7f}\t Validation MSE: {val_mse:.7f}")

plt.figure(1)
plt.plot(trainmse, label='Training Loss')
plt.plot(valmse, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# ------------------------- Plot predicted values against cycle number ------------------------- #

# Get the scaling range for the final column of dfcomb_final
soh_range = scaler.data_range_[-1]
soh_min = scaler.data_min_[-1]


# plot prediction results

with torch.no_grad():

    unnormalized_val_results = val_results * soh_range + soh_min

    plt.figure(2)

    # plot predicted results
    plt.scatter(X_test.index, unnormalized_val_results, label='Predicted SOH', s=10)

    # plot true results on same graph:
    with open('dfcomb_final.pkl', 'rb') as handle:
        dfcomb_final = pickle.load(handle)

    # plt.scatter(dfcomb_final.index, dfcomb_final['Average SOH'], label='True SOH', s=15)
    plt.scatter(dfcomb_final.tail(129).index, dfcomb_final.tail(129)['Average SOH'], label='True SOH', s=20)


    plt.xlabel('Cycle Number')
    plt.ylabel('SOH %')
    plt.legend()
    plt.show()


# store values to plot in new script - GPR followed by true SOH
gprpred_plot = {}
gprpred_plot[0] = unnormalized_val_results
gprpred_plot[1] = X_test.index

# with open("gprpred4d_plot", "wb") as fp:   #Pickling
#     pickle.dump(gprpred_plot, fp)


# true SOH:
# true2a_plot = {}
# true2a_plot[0] = dfcomb['Average SOH']
# true2a_plot[1] = dfcomb.index
#
# with open("true2a_plot", "wb") as f:   #Pickling
#     pickle.dump(true2a_plot, f)

print('hello')