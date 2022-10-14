# credit to https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/
# Convert the flavor transformation data to one with reduced dimensionality to make it easier to train on
# Run from the directory containin the joint dataset
import h5py
import re
import numpy as np
import time
import copy
from sklearn.model_selection import train_test_split
import torch
from torch import nn

input_filename = "many_sims_database_RUN_lowres_sqrt2_RUN_standard.h5"
N_Nbar_tolerance = 1e-3
NF=2

#===============================================#
# read in the database from the previous script #
#===============================================#
f_in = h5py.File(input_filename,"r")
#ijkList_growthrate = np.array(f_in["ijk_growthrate"])
growthRateList     = np.array(f_in["growthrate(1|s)"])
#ijkList_F4         = np.array(f_in["ijk_F4"])
F4_initial_list    = np.array(f_in["F4_initial_Nsum1"]) # [ind, xyzt, nu/nubar, flavor]
F4_final_list      = np.array(f_in["F4_final_Nsum1"])
f_in.close()

# N-Nbar must be preserved
N_Nbar_initial = F4_initial_list[:,3,0,:] - F4_initial_list[:,3,1,:]
N_Nbar_final = F4_final_list[:,3,0,:] - F4_final_list[:,3,1,:]
N_Nbar_error = np.max(np.abs(N_Nbar_initial - N_Nbar_final))
print("N_Nbar_error = ", N_Nbar_error)
assert(N_Nbar_error < N_Nbar_tolerance)

# define the input (X) and output (y) for the neural network
nsims = F4_initial_list.shape[0]
IO_shape = F4_initial_list.shape[1:]
number_predictors = np.product(IO_shape)
X = F4_initial_list
y = F4_final_list

# split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# create test ("Fiducial" simulation)
F4_test = np.zeros(IO_shape)
F4_test[3,0,0] = 1
F4_test[3,1,0] = 1
F4_test[2,0,0] = 1/3
F4_test[2,1,0] = -1/3
F4_test /= np.sum(F4_test[3])

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_test = torch.Tensor(y_test).to(device)

# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32,16)
        )

    def forward(self,x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# instantiate the model
model = NeuralNetwork().to(device)
print(model)

# define the loss function
loss_fn = nn.MSELoss(reduction='sum')

# use the adam optimizer
optimizer = torch.optim.Adam(model.parameters())

# create list of equivalent simulations
# X = [isim, xyzt, nu/nubar, flavor]
flip_antimatter_list = [True, False] # possible orderings of neutrinos and antineutrinos
def augment_data(X,y):
    Xlist = torch.zeros_like(X)
    ylist = torch.zeros_like(y)
    for reflect0 in [-1,1]:
        for reflect1 in [-1,1]:
            for reflect2 in [-1,1]:
                for flip_antimatter in flip_antimatter_list:
                    thisX = copy.deepcopy(X)
                    thisy = copy.deepcopy(y)

                    # perform nu/nubar reordering
                    if flip_antimatter:
                        thisX = torch.flip(thisX, [2])
                        thisX = torch.flip(thisy, [2])
                    
                    # perform reflection operations
                    thisX[0,0,:,:] *= reflect0
                    thisX[0,1,:,:] *= reflect1
                    thisX[0,2,:,:] *= reflect2
                    thisy[0,0,:,:] *= reflect0
                    thisy[0,1,:,:] *= reflect1
                    thisy[0,2,:,:] *= reflect2
                    Xlist = torch.cat((Xlist, thisX))
                    ylist = torch.cat((ylist, thisy))

    # flatten the input/output. Torch expects the last dimension size to be the number of features.
    Xlist = torch.flatten(Xlist,start_dim=1)
    ylist = torch.flatten(ylist,start_dim=1)

    # remove first element (which is just zeros - should make this more elegant)
    Xlist = Xlist[1:]
    ylist = ylist[1:]

    return Xlist, ylist
    
# function to train the dataset
def train(Xlist,ylist, model, loss_fn, optimizer):
    nsims = len(Xlist)
    model.train()

    training_loss = 0
    for isim in range(nsims):
        # select a single data point (batch size 1)
        loc = isim
        X = Xlist[loc:loc+1]
        y = ylist[loc:loc+1]

        # AUGMENT DATA HERE
        X,y = augment_data(X,y)
        
        # compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss = max(training_loss, loss)
    print(f"Training Error: {training_loss:>7f}")

# function to test the model performance
def test(Xlist,ylist, model, loss_fn):
    nsims = len(Xlist)
    model.eval()
    with torch.no_grad():
        X = torch.flatten(Xlist, start_dim=1)
        y = torch.flatten(ylist, start_dim=1)
        pred = model(X)
        test_loss = loss_fn(pred,y).item() / nsims
        print(f"Test Error: {test_loss:>8f}")

# training loop
epochs = 20
for t in range(epochs):
    print("----------------------------------")
    print(f"Epoch {t+1}")
    train(X_train, y_train, model, loss_fn, optimizer)
    test(X_test, y_test, model, loss_fn)
print("Done!")
exit()

#===============================#
# Abstract out boilerplate code #
#===============================#
def run_ML_model(estimator, param_grid, label):

    # run the grid search
    StartTime = time.time()
    #grid_search.fit(X,y)
    EndTime = time.time()

    print("Total time:",EndTime-StartTime,"seconds.")

    before = F4_test
    after = F4_test #grid_search.predict(F4_test.reshape((1,number_predictors))).reshape(IO_shape)
    print("N Before:", before[3].flatten(), np.sum(before[3,0]), np.sum(before[3,1]), np.sum(before[3]))
    print("N After :", after[3].flatten(), np.sum(after[3,0]), np.sum(after[3,1]), np.sum(after[3]))
    print()
    print("Fz Before:", before[2].flatten(), np.sum(before[2]))
    print("Fz After :", after[2].flatten(), np.sum(after[2]))
    print()
    print("Fy Before:", before[1].flatten(), np.sum(before[1]))
    print("Fy After :", after[1].flatten(), np.sum(after[1]))
    print()
    print("Fx Before:", before[0].flatten(), np.sum(before[0]))
    print("Fx After :", after[0].flatten(), np.sum(after[0]))
    print()

