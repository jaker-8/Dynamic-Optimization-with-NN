#%% Import and create differential equation

import numpy as np
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dxdt(x,t):
    dxdt = 0.5/(x + 2)
    return dxdt

# Length of time to solve for (time points = tf + 1, each pt = 1 min)
tf = 100 # min
timepts = tf + 1
tvect = np.linspace(0, tf, timepts)

x_list = np.linspace(0, 10, 100000)

x_inputs = np.zeros(len(x_list))
x_outputs = np.zeros(len(x_list))
row = 0

for x_ind, x in enumerate(x_list):

    x_inputs[x_ind] = x
    x_outputs[x_ind] = odeint(dxdt, x, t=[0,1])[1][0]

    row += 1

print(row)

#%% Define and train neural network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import copy

x_inputs = np.vstack((x_inputs))
print(np.shape(x_inputs))
x_outputs = np.vstack((x_outputs))
print(np.shape(x_outputs))
x_inputs_tens = torch.tensor(x_inputs, dtype=torch.float32)

x_outputs_tens = torch.tensor(x_outputs, dtype=torch.float32)

# Define neural network
class x_next(nn.Module):
    def __init__(self):
        super(x_next, self).__init__()
        self.layer1 = nn.Linear(1, 80)  # Input layer to hidden layer
        self.layer2 = nn.Linear(80, 1) # Hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = (self.layer2(x))
        return x
    
# Initialize the model
x_next_model = x_next()

# Create data set and data loader to use batches
dataset = TensorDataset(x_inputs_tens, x_outputs_tens)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

# Define loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(x_next_model.parameters(), lr=0.01)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

best_loss = float('inf')
patience = 50
patience_left = patience

# Train model
num_epochs = 500
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # Evaluate model
        outputs = x_next_model(x_batch)
        # Calculate loss
        loss = loss_func(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step the scheduler and print the learning rate if it changes
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f'Epoch {epoch+1}: reducing learning rate from {current_lr:.5f} to {new_lr:.5f}')

    # Early stopping criteria
    if loss <= best_loss:
        best_loss = loss
        patience_left = patience
        best_model_weights = copy.deepcopy(x_next_model.state_dict())
        best_epoch = epoch
    else:
        patience_left -= 1
        if patience_left == 0:
            break

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.10f}')

# Use the best weights in the model
x_next_model.load_state_dict(best_model_weights)

# Determine the best epoch and lost value when complete
print(f"Best epoch: {best_epoch}")
print(f"Smallest loss value: {best_loss:.8f}")


#%% Save as ONNX file and add input bounds
import tempfile
import torch.onnx
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds

x_input_bounds = [(0,10)]

# Make random data for batch size, input size (dummy input to trace the model)
x_trace = torch.randn(1000, 1, requires_grad=True)

with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    torch.onnx.export(
        x_next_model,
        x_trace,
        f,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    write_onnx_model_with_bounds(f.name, None, x_input_bounds)
    print(f"Wrote PyTorch model to {f.name}")
    x_pytorch_model = f.name


#%% Load the ONNX file to use
import os
from omlt.neuralnet import FullSpaceNNFormulation
import pyomo.environ as pyo
from omlt.neuralnet.activations import ComplementarityReLUActivation
from omlt import OmltBlock

# Import necessary libraries
from pyomo.environ import *  # Pyomo for optimization modeling
from pyomo.dae import *  # Differential algebraic equations module in Pyomo
from pyomo.gdp import Disjunct, Disjunction
import numpy as np  # Numpy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting results
import math
from scipy.integrate import trapz
import time

# Import NN from ONNX with specified bounds
x_model_pyo = load_onnx_neural_network_with_bounds(x_pytorch_model)
# Remove the temporary file created
os.remove(x_pytorch_model)

# Confirm layers and bounds
print(x_model_pyo.scaled_input_bounds)
for layer_id, layer in enumerate(x_model_pyo.layers):
    print(f"{layer_id}\t{layer}\t{layer.activation}")


#%% Test using original function
# Evaluate NN
def x_next_eval(x1):

    NN_input = np.vstack((x1))
    NN_input = torch.tensor(NN_input, dtype=torch.float32)

    x_next_model.eval()
    with torch.no_grad():
        x_test = x_next_model(NN_input).item()

    return x_test


x_init = 1
x_t = np.zeros(timepts)
x_t[0] = x_init

for t_ind in range(tf):
    x_val = np.array([x_t[t_ind]])
    x_t[t_ind+1] = x_next_eval(x_val)

plt.figure()
plt.plot(tvect, x_t)



#%% Use NN in Pyomo model via OMLT
# Import necessary libraries
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from omlt.neuralnet import ReducedSpaceSmoothNNFormulation
import pyomo.kernel

# --------------------------- Save PyTorch NN and create NN formulation ---------------------- #

# model.nn_x = OmltBlock()
x_input_bounds = [(0,10)]

# Make random data for batch size, input size (dummy input to trace the model)
x_trace = torch.randn(1000, 1, requires_grad=True)

with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    torch.onnx.export(
        x_next_model,
        x_trace,
        f,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    write_onnx_model_with_bounds(f.name, None, x_input_bounds)
    print(f"Wrote PyTorch model to {f.name}")
    x_pytorch_model = f.name

# Import NN from ONNX with specified bounds
x_model_pyo = load_onnx_neural_network_with_bounds(x_pytorch_model)
# Remove the temporary file created
os.remove(x_pytorch_model)

# Neural network formulation - used loaded ONNX file
x_formulation = ReducedSpaceSmoothNNFormulation(x_model_pyo)


# ---------------------------------- Set up Pyomo model --------------------------------- #


# Time horizon for the simulation
start_time = 0.0
end_time = 100.0

# Initialize the Concrete Model
model = ConcreteModel()

# Define an index set
model.block_indices = RangeSet(0, 100)

# Create an indexed Block to hold the OmltBlock instances
model.nn_x = Block(model.block_indices, rule=lambda b, i: b.add_component("omlt_block", OmltBlock()))

# Now build formulations for each OmltBlock
for i in model.block_indices:
    model.nn_x[i].omlt_block.build_formulation(x_formulation)


# Make discrete time points within specified bounds
time_points = list(range(int(start_time), int(end_time) + 1)) 
model.t = Set(initialize=time_points)
model.x = Var(model.t, domain=Reals)

# Initial value for x
x_init = 1
model.x[int(start_time)].fix(x_init)


# Create constraint for each time point
def input_connect(m, t):
    if t == m.t.last():
        print(f'Input constraint skipped for last time point, {t}... nothing fed to NN\n')
        return Constraint.Skip  # Skip the last time point
    return m.nn_x[t].omlt_block.inputs[0] == m.x[t]

model.input_constraint = Constraint(model.t, rule=input_connect)

# Constraint to connect outputs of the neural network block to the state variable
def output_connect(m, t):
    if t == m.t.first():
        print(f'Output constraint skipped for first time point, {t}... output is the known initial condition\n')
        return Constraint.Skip  # Skip the first time point
    t_prev = m.t.prev(t)
    return m.x[t] == m.nn_x[t_prev].omlt_block.outputs[0]

model.output_constraint = Constraint(model.t, rule=output_connect)


# --------------------------------- Solve and Plot --------------------------------- #


# Objective function (trivial)
def obj_fxn(m):
    return 1.0

model.obj = Objective(rule=obj_fxn)

# Solver setup and solution
solver = SolverFactory('ipopt')
results = solver.solve(model, tee=True)
print(results)

# Extract and plot results
times = np.array([t for t in model.t])
x_pts = np.array([model.x[t].value for t in times])



# Compare to NN evaluations

x_t = np.zeros(timepts)
x_t[0] = x_init

for t_ind in range(int(end_time)):
    x_val = np.array([x_t[t_ind]])
    x_t[t_ind+1] = x_next_eval(x_val)



plt.figure()
plt.plot(times, x_pts)
plt.plot(times, x_t, linestyle='--')
plt.xlabel('Time')
plt.ylabel('x')
plt.legend(['With OMLT & Pyomo', 'Without Pyomo'])
plt.show()

