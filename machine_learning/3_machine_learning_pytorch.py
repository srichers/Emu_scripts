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
from itertools import permutations

input_filename = "many_sims_database_RUN_lowres_sqrt2_RUN_standard.h5"
N_Nbar_tolerance = 1e-3
NF=2
epochs = 5

#===============================================#
# read in the database from the previous script #
#===============================================#
f_in = h5py.File(input_filename,"r")
growthRateList     = np.array(f_in["growthrate(1|s)"])
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
print("split train shape = ", X_train.shape, y_train.shape)
print("split test shape = ",X_test.shape, y_test.shape)
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)
X_test = torch.Tensor(X_test).to(device)
y_test = torch.Tensor(y_test).to(device)

# create list of equivalent simulations
# X = [isim, xyzt, nu/nubar, flavor]
def augment_data(X,y):
    # number of augmentations
    # 2 reflections in each direction
    # permutations for anti/matter, direction, and flavor
    matter_permutations    = list(permutations(range(2)))
    direction_permutations = list(permutations(range(3)))
    flavor_permutations    = list(permutations(range(NF)))
    n_augment = 2**3 * len(matter_permutations) * len(direction_permutations) * len(flavor_permutations)

    # augmented arrays. [iaug, isim, xyzt, nu/nubar, flavor]
    Xaugmented = X.repeat(n_augment,1,1,1,1)
    yaugmented = y.repeat(n_augment,1,1,1,1)

    iaug = 0
    for reflect0 in [-1,1]:
        for reflect1 in [-1,1]:
            for reflect2 in [-1,1]:
                for mperm in matter_permutations:
                    for dperm in direction_permutations:
                        for fperm in flavor_permutations:
                            # permute which direction is which
                            Xaugmented[iaug,:,0:3] = Xaugmented[iaug,:,dperm]
                            yaugmented[iaug,:,0:3] = yaugmented[iaug,:,dperm]
                        
                            # permute which flavor is which
                            Xaugmented[iaug] = Xaugmented[iaug,:,:,:,fperm]
                            yaugmented[iaug] = yaugmented[iaug,:,:,:,fperm]
                        
                            # perform nu/nubar reordering
                            Xaugmented[iaug] = Xaugmented[iaug,:,:,mperm]
                            yaugmented[iaug] = yaugmented[iaug,:,:,mperm]
                    
                            # perform reflection operations
                            Xaugmented[iaug,:,0] *= reflect0
                            Xaugmented[iaug,:,1] *= reflect1
                            Xaugmented[iaug,:,2] *= reflect2
                            yaugmented[iaug,:,0] *= reflect0
                            yaugmented[iaug,:,1] *= reflect1
                            yaugmented[iaug,:,2] *= reflect2
                            
    # flatten the input/output. Torch expects the last dimension size to be the number of features.
    #Xaugmented = torch.flatten(Xaugmented,start_dim=2)
    #yaugmented = torch.flatten(yaugmented,start_dim=2)

    # switch the array index order to the simulation index is first and the augmentation index is second
    Xaugmented = torch.transpose(Xaugmented,0,1)
    yaugmented = torch.transpose(yaugmented,0,1)

    return Xaugmented, yaugmented

# augment both datasets
X_train, y_train = augment_data(X_train, y_train)
X_test,  y_test  = augment_data(X_test,  y_test )
print("augmented training data shape:",X_train.shape, y_train.shape)
print("augmented testing data shape:",X_test.shape, y_test.shape)


# function to train the dataset
def train_chunked(Xlist,ylist, model, loss_fn, optimizer):
    nsims = Xlist.shape[0]
    model.train()

    training_loss = 0
    for isim in range(nsims):
        # select a single simulation, including all augmentations
        X = Xlist[isim]
        y = ylist[isim]

        # compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss = max(training_loss, loss.item())
    print(f"Training Error: {training_loss:>7f}")

# function to train the dataset
def train(Xlist,ylist, model, loss_fn, optimizer):
    nsims = Xlist.shape[0]
    model.train()

    # compute the prediction error
    pred = model(Xlist)
    loss = loss_fn(pred, ylist)

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    training_loss = loss.item()
    print(f"Training Error: {training_loss:>7f}")

# function to test the model performance
def test(Xlist,ylist, model, loss_fn):
    nsims = Xlist.shape[0]
    model.eval()
    with torch.no_grad():
        pred = model(Xlist)
        test_loss = loss_fn(pred,ylist).item()
        print(f"Test Error: {test_loss:>8f}")

def cyclic_permute(A, steps):
    ndims = len(A.shape)
    if(steps>0): steps -= ndims
    permutation = [range(ndims)[i + steps] for i in range(ndims)]
    return torch.permute(A,permutation)
    
# generate y from final distributions
# y quantifies the amount of flow from flavor i to flavor i+1
# We do not consider antineutrino number density flux, because it must be identical to that for neutrinos
# For NF flavors, there are NF-1 flow quantities for each of 7 components
# assumes final three indices index into distribution properties and other indices identify samples
Ny = (NF-1)*7
def y_from_F4(F4_initial, F4_final):
    # take the difference between the final and initial solutions
    delta_F4 = F4_final - F4_initial # [isim1, isim2,..., xyzt, nu/nubar, flavor]

    # permute the indices so the sample indices are at the end and we can use the same expressions for one data point and many data points
    input_rank = len(delta_F4.shape)
    single_element_rank = 3
    assert(input_rank >= single_element_rank)
    shift = input_rank - single_element_rank
    delta_F4 = cyclic_permute(delta_F4, shift) # [xyzt, nu/nubar, flavor, isim1, isim2, ...]

    # calculate the flow from one flavor to the next
    delta_F4_flow = delta_F4[:,:,1:NF] - delta_F4[:,:,0:NF-1]

    # create y array of correct size
    # [iy, isim1, isim2, ...]
    shape = [Ny]
    if input_rank>single_element_rank:
        for dim in range(3,input_rank):
            shape.append(delta_F4.shape[dim])
    y = torch.zeros(shape).to(device)

    # fill in y
    diy = (NF-1)*2
    for i in range(3):
        iystart = diy*i
        y[iystart:iystart+diy] = delta_F4_flow[i].flatten(start_dim=0, end_dim=1)
    iystart = diy*3
    y[iystart:iystart+(NF-1)] = delta_F4_flow[3,0]

    # shift back to have simulation index first
    y = cyclic_permute(y,-shift) # [isim1, isim1, ... , iy]
    return y

#def F4_from_y(F4_initial, y):
#    F4_final = copy.deepcopy(F4_initial)

#    F4_final[]

y_train = y_from_F4(X_train, y_train)
y_test = y_from_F4(X_test, y_test)
X_train = torch.flatten(X_train,start_dim=2)
X_test = torch.flatten(X_test,start_dim=2)

print("Restructured train shape:",X_train.shape, y_train.shape)
print("Restructured test shape:",X_test.shape, y_test.shape)

# define the NN model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,Ny)
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

# training loop
for t in range(epochs):
    print("----------------------------------")
    print(f"Epoch {t+1}")
    #train_chunked(X_train, y_train, model, loss_fn, optimizer)
    train(X_train, y_train, model, loss_fn, optimizer)
    test(X_test, y_test, model, loss_fn)
print("Done!")

#=====================================#
# create test ("Fiducial" simulation) #
#=====================================#
F4_test = np.zeros(IO_shape)
F4_test[3,0,0] = 1
F4_test[3,1,0] = 1
F4_test[2,0,0] = 1/3
F4_test[2,1,0] = -1/3
F4_test /= np.sum(F4_test[3])

before = torch.Tensor(F4_test).to(device)
shape = F4_test.shape
print("before:",before.shape)
after = model(before.flatten(start_dim=0)).reshape(shape)
print("after:",after.shape)


before = before.to('cpu').detach().numpy()
after = after.to('cpu').detach().numpy()

print("N Before:", before[3].flatten(), np.sum(before[3]))
print("N After :", after[3].flatten() , np.sum(after[3]) )
print()
print("Fz Before:", before[2].flatten(), np.sum(before[2]))
print("Fz After :", after[2].flatten() , np.sum(after[2]) )
print()
print("Fy Before:", before[1].flatten(), np.sum(before[1]))
print("Fy After :", after[1].flatten() , np.sum(after[1]) )
print()
print("Fx Before:", before[0].flatten(), np.sum(before[0]))
print("Fx After :", after[0].flatten() , np.sum(after[0]) )
print()

