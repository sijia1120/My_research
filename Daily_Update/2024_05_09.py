!pip install blitz-bayesian-pytorch

# ----------  Import Dataset  -----------------------------
import torch
from torch import nn
from torch.nn import functional as F
from blitz.modules.base_bayesian_module import BayesianModule
from blitz.modules.weight_sampler import TrainableRandomDistribution, PriorWeightDistribution

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np

#from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


# ---------- Import Dataset -------------------
BATCH_SIZE = 16
boston = fetch_openml(name='boston', version=1)
X, y = boston.data, boston.target
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(np.expand_dims(y, -1))
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.25,
                                                    random_state=42)
X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

ds_train = torch.utils.data.TensorDataset(X_train, y_train)
dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)

ds_test = torch.utils.data.TensorDataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)



# ---------- Linear Regression + Lasso -------------
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error

# Instantiate the Linear Regression model
model = LinearRegression()
model.fit(X_train.numpy(), y_train.numpy())
predictions = model.predict(X_test.numpy())
mse = mean_squared_error(y_test.numpy(), predictions)
print("Mean Squared Error:", mse)

# ------------   Instantiate the Lasso Regression model   -------------------------
lasso_model = Lasso(alpha=0.000001)  # You can adjust the alpha parameter for regularization strength
lasso_model.fit(X_train.numpy(), y_train.numpy())
lasso_predictions = lasso_model.predict(X_test.numpy())
# Calculate Mean Squared Error (MSE)
lasso_mse = mean_squared_error(y_test.numpy(), lasso_predictions)
print("Lasso Mean Squared Error:", lasso_mse)

ridge_model = Ridge(alpha=50.0)  # You can adjust the alpha parameter for regularization strength
ridge_model.fit(X_train.numpy(), y_train.numpy())
ridge_predictions = ridge_model.predict(X_test.numpy())
# Calculate Mean Squared Error (MSE)
ridge_mse = mean_squared_error(y_test.numpy(), ridge_predictions)
print("Ridge Mean Squared Error:", ridge_mse)


class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer, implements the linear layer proposed on Weight Uncertainity on Neural Networks
    (Bayes by Backprop paper).

    Its objective is be interactable with torch nn.Module API, being able even to be chained in nn.Sequential models with other non-this-lib layers

    parameters:
        in_fetaures: int -> incoming features for the layer
        out_features: int -> output features for the layer
        bias: bool -> whether the bias will exist (True) or set to zero (False)
        prior_sigma_1: float -> prior sigma on the mixture prior distribution 1
        prior_sigma_2: float -> prior sigma on the mixture prior distribution 2
        prior_pi: float -> pi on the scaled mixture prior
        posterior_mu_init float -> posterior mean for the weight mu init
        posterior_rho_init float -> posterior mean for the weight rho init
        freeze: bool -> wheter the model will start with frozen(deterministic) weights, or not

    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 prior_sigma_1 = 0.4, # 0.1,
                 prior_sigma_2 = 0.1,  # 0.4,
                 prior_pi = 0.7, # 1,
                 posterior_mu_init = 0., # 0,
                 posterior_rho_init = -7.0,
                 freeze = False,
                 prior_dist = None):
        super().__init__()

        #our main parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.freeze = freeze


        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init

        #parameters for the scale mixture prior
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.prior_dist = prior_dist

        # Variational weight parameters and sample
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        self.weight_sampler = TrainableRandomDistribution(self.weight_mu, self.weight_rho)

        # Variational bias parameters and sample
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
        self.bias_sampler = TrainableRandomDistribution(self.bias_mu, self.bias_rho)

        # Priors (as BBP paper)
        self.weight_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.bias_prior_dist = PriorWeightDistribution(self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist = self.prior_dist)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x):
        # Sample the weights and forward it

        #if the model is frozen, return frozen
        if self.freeze:
            return self.forward_frozen(x)

        w = self.weight_sampler.sample()

        if self.bias:
            b = self.bias_sampler.sample()
            b_log_posterior = self.bias_sampler.log_posterior()
            b_log_prior = self.bias_prior_dist.log_prior(b)

        else:
            b = torch.zeros((self.out_features), device=x.device)
            b_log_posterior = 0
            b_log_prior = 0

        # Get the complexity cost
        self.log_variational_posterior = self.weight_sampler.log_posterior() + b_log_posterior
        self.log_prior = self.weight_prior_dist.log_prior(w) + b_log_prior

        return F.linear(x, w, b)



@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(in_features= input_dim, out_features = 100)
        self.blinear2 = BayesianLinear(in_features= 100, out_features = 100)
        self.blinear3 = BayesianLinear(in_features= 100, out_features = output_dim)
        self.acti = activation

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = self.acti(x_)
        x_ = self.blinear2(x_)
        x_ = self.acti(x_)
        return self.blinear3(x_)




def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()

    mse = torch.mean((means - y) ** 2)
    upper_bound_acc = (ci_upper >= y).float().mean()
    lower_bound_acc = (ci_lower <= y).float().mean()
    print(f"MSE: {mse}")
    return ic_acc, ic_acc, upper_bound_acc, lower_bound_acc,mse



num_features = X.shape[1]
num_epochs = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
regressor = BayesianRegressor(input_dim= num_features, output_dim=1, activation =nn.Softplus()).to(device)
optimizer = optim.Adam(regressor.parameters(), lr=0.01)
# Adam optimization with a learning rate of 0.01
criterion = torch.nn.MSELoss()


def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


from sklearn.metrics import mean_squared_error
iteration = 0
ensemble_size = 100

for epoch in range(100):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = regressor.sample_elbo(inputs=datapoints.to(device),
                                      labels=labels.to(device),
                                      criterion=criterion,
                                      sample_nbr=3,
                                      complexity_cost_weight=1/X_train.shape[0])
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 100 == 0:
            # Initialize lists to store predictions and losses
            all_preds_test = []
            all_preds_train = []
            all_losses_test = []
            all_losses_train = []

            # Loop over ensemble size
            for _ in range(ensemble_size):
                eval_result = evaluate_regression(regressor, X_test.to(device),
                                                  y_test.to(device),
                                                  samples=25,
                                                  std_multiplier=3)
                ic_acc, under_ci_upper, over_ci_lower = eval_result
                preds_train = regressor(X_train)[:, 0].unsqueeze(1)
                preds_test = regressor(X_test)[:, 0].unsqueeze(1)
                loss_train = criterion(preds_train, y_train)
                loss_test = criterion(preds_test, y_test)

                all_preds_test.append(preds_test.detach().cpu().numpy())
                all_losses_test.append(loss_test.item())
                all_preds_train.append(preds_train.detach().cpu().numpy())
                all_losses_train.append(loss_train.item())

            # Calculate average predictions and loss
            avg_preds_test = np.mean(all_preds_test, axis=0)
            avg_loss_test = np.mean(all_losses_test)
            avg_preds_train = np.mean(all_preds_train, axis=0)
            avg_loss_train = np.mean(all_losses_train)

            # Print evaluation results
            print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))
            print("Average train Loss: {:.4f} Average test Loss: {:.4f}".format(avg_loss_train, avg_loss_test))
            print(mean_squared_error(avg_preds_test, y_test.detach().numpy()))
            print(mean_squared_error(avg_preds_train, y_train.detach().numpy()))
            #print("Iteration: {} Val-loss: {:.4f}".format(str(iteration), criterion(avg_preds, y_test)))
            print()


