# best model to compare results for each individual dataset with a predicted model

import torch
import gpytorch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# import dataframes from main - CHOOSE DATASET BASED ON NUMBER AFTER FRAMES
with open('dataset1.pkl', 'rb') as handle:
    frames = pickle.load(handle)

# ------------------------- Sorting data input ------------------------- #
df1 = frames[0]
df2 = frames[1]
df3 = frames[2]

allframes = [df1, df2, df3]

# combining all dataframes
dfcomb = pd.concat(allframes, axis=1)
print(dfcomb.shape)

dfcomb = dfcomb.drop(['SOH charge cycles', 'SOH discharge cycles', 'SOH discharge cycles 2', 'SOH discharge 2'], axis=1)
print(dfcomb.shape)

# fixing issue of value stored as lists:
dfcomb = dfcomb.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

# find average SOH for charge and discharge, then remove those 2 columns
dfcomb['Average SOH'] = dfcomb[['SOH charge', 'SOH discharge']].mean(axis=1)
dfcomb = dfcomb.drop(['SOH charge', 'SOH discharge'], axis=1)




# select the original features only, which are used in the report
# dfcomb = dfcomb[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'Max ICA', 'Max ICA voltage', 'Av volt discharge', 'Average SOH']]
dfcomb = dfcomb[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'ICA delta 1', 'ICA delta 2', 'Av volt discharge', 'Average SOH']]




# remove NaNs from the whole dataset:

print('No. NaN values: ', dfcomb.isnull().sum().sum())

dfcomb = dfcomb.dropna(axis=0, how='any')

print('Shape after:', dfcomb.shape)






# normalising data:
scaler = MinMaxScaler()

col_names = list(dfcomb.columns)
row_num = list(dfcomb.index)

dfcomb_final_scaled = scaler.fit_transform(dfcomb.to_numpy())
dfcomb_final = pd.DataFrame(dfcomb_final_scaled, columns=col_names, index=row_num)








# get the x and y data:
X = dfcomb_final.drop('Average SOH', axis=1)
print(X.shape)

y = dfcomb_final['Average SOH']
print(y.shape)

# Split data into x and y, with 30% for testing and randomly shuffling data
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=22)

print('yeee')


# ------------------------- GPR ------------------------- # https://richardcsuwandi.medium.com/gaussian-process-regression-using-gpytorch-2c174286f9cc
torch.manual_seed(0)

# convert X_train and X_test to numpy arrays

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)




class MyModule (gpytorch.models.ExactGP):
    # Initialize the parameter
    def __init__(self, X_train_tensor, y_train_tensor, likelihood):
        super(MyModule, self).__init__(X_train_tensor, y_train_tensor, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=6)  # Construct the kernel function
        self.cov.initialize_from_data(X_train_tensor, y_train_tensor)  # Initialize the hyperparameters from data

    # Forward pass
    def forward(self, input):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(input)
        cov_x = self.cov(input)

        # Reshape the mean and covariance tensors
        # to have shape [batch_size, 1, num_mixtures]
        mean_x = mean_x.unsqueeze(-1).expand(-1, -1, self.cov.num_mixtures)
        cov_x = cov_x.unsqueeze(-1).expand(-1, -1, self.cov.num_mixtures)

        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# model = MyModule(X_train_tensor, y_train_tensor, likelihood)

model = MyModule(X_train_tensor, y_train_tensor, likelihood)
# model = model.mean


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

# criterion = torch.nn.MSELoss(size_average=False)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# Training loop
model.train()
likelihood.train()

training_iterations = 500
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = -likelihood(output, y_train_tensor).log_prob(y_train_tensor).mean()
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))



# convert to pytorch tensors:

# convert X_train and X_test to numpy arrays

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# adding another dimension for LSTM model:
X_train_tensor = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, X_train_tensor.shape[1]))

X_test_tensor = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, X_test_tensor.shape[1]))


# training and test data
train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32)

test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=32)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Training model test:
num_epochs = 700
training_losses = []
validation_losses = []

val_results = []

for epoch in range(num_epochs):
    batch_loss = []
    # training losses:
    for X, y in train_dataloader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    training_loss = np.mean(batch_loss)
    training_losses.append(training_loss)

    # Validation:

    val_results = []

    with torch.no_grad():
        val_losses = []
        model.eval()
        for X, y in test_dataloader:
            outputs = model(X)
            val_results.append(outputs.numpy())
            val_loss = loss_fn(outputs, y)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    val_results = np.concatenate(val_results, axis=0)

    print(f"[{epoch+1}] Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}")

plt.figure(1)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.title('Battery B0007')
plt.legend()


# ------------------------- Plot predicted values against cycle number ------------------------- #

# get the inverse scaler first:

# Get the scaling range for the final column of dfcomb_final
soh_range = scaler.data_range_[-1]
soh_min = scaler.data_min_[-1]

# Unnormalize val_results
unnormalized_val_results = val_results * soh_range + soh_min



print(X_test.index, unnormalized_val_results)
plt.figure(2)
plt.scatter(X_test.index, unnormalized_val_results, label='Predicted SOH')
plt.xlabel('Cycle Number')
plt.ylabel('SOH %')
plt.legend()



# plot true results on same graph:
plt.scatter(dfcomb.index, dfcomb['Average SOH'], label='True SOH')
# plt.title('Battery B0007')
plt.legend()


plt.show()



print('hello')