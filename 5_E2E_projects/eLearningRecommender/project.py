# %% [markdown]
# # AI RECOMMENDATION AGENT FOR PERSONALIZED E-LEARNING
# 
# ---
# 
# <h3>Project Overview of the proposed AI Recommendation Agent</h3>
# 
# This project suggests creating a customised AI recommendation agent for e-learning that makes use of methodologies like `Item Response Theory (IRT)` and `Artificial Neural Networks (ANN)`. By offering `personalized recommendations` based on the `student's academic performance` and `learning preferences`.
# 
# The suggested method seeks to overcome the difficulty of locating relevant and interesting content on an e-learning platform. ANN will adaptively suggest learning resources based on the student's academic ability level and learning preferences after using IRT to model the student's academic ability level.
# 

# %% [markdown]
# With the use of criteria like `accuracy, coverage, and novelty`, the system's performance will be assessed for the project. 

# %% [markdown]
# `Artificial neural networks (ANN) and item response theory (IRT)` have emerged as cutting-edge solutions to overcome these issues and boost the precision and efficacy. 
# 
# The statistical framework (mathematical model) known as IRT, often referred to as Latent Response Theory, simulates the link between `latent qualities` and their manifestations.
# 
# 
# IRT can be used in e-learning to model a `student's academic prowess` and the `degree of difficulty of the course materials` to deliver customized reading content recommendations.

# %% [markdown]
# The system will use `IRT to model the student's academic proficiency` and `ANN to adaptively offer learning materials` that are appropriate for both their academic proficiency and learning preferences.
# 
# ### Links
# [IRT in Python](https://www.linkedin.com/pulse/item-response-theory-modeling-python-part-1-andrew-f/)

# %% [markdown]
# ### IRT Model
# 
# <p>IRT will be used to model the student's academic ability level and the difficulty of the learning materials. The IRT model will be trained using the student's responses to a set of test criteria / items.</p>
# 
# `P(ij) = c + (1 - c) * (e^(a*(tj-bi)) / (1 + e^(a*(tj-bi))))`
# 
# where:
# ==> Pij is the prob-academic ability of a student j correctly answering an item i.
# 
# ==> c is a guessing parameter, representing the prob-academic ability of guessing the correct answer.
# 
# ==> a is the discrimination parameter, representing how well the item discriminates between high and low academic ability students.
# 
# ==> tj is the academic ability level of student j.
# 
# ==> bi is the difficulty level of item I.
# 
# 
# `This would output the probability of the student correctly answering the item, given their academic ability level and the item difficulty level.`

# %% [markdown]
# To build a machine learning model based on the IRT formula, we need to first define the inputs and outputs of the model.
# <br>
# 
# `Inputs:` The inputs to the model will be the student's responses to a set of test criteria / items, along with the difficulty level of each item.
# 
# `Outputs:` The output of the model will be the predicted academic ability level of the student.
# 
# To train the model, we can use a supervised learning approach where we provide the model with labeled data consisting of the student's responses to the test items and their corresponding academic ability levels. 
# 
# We can then use a regression algorithm to learn the relationship between the inputs (test item responses and item difficulty levels) and the output (academic ability level).

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score,confusion_matrix
import numpy as np

np.random.seed(0) # Set seed for reproducibility

# %%
# TEST~ IRT Formula to predict the probability of a student answering a question correctly in code
# Define the IRT formula
def irt_formula(c, a, tj, bi):
    """
        c: guessing parameter
        a: discrimination parameter
        tj: academic ability level of student j
        bi: difficulty level of item i
    """
    return c + (1 - c) * (np.exp(a * (tj - bi)) / (1 + np.exp(a * (tj - bi))))

p_ = irt_formula(0.25, 1.5, 3, 1)
print(p_)  

# %%
# Generate some sample data : from formula
num_students = 4000
num_items = 50

item_difficulties = np.random.normal(0, 1, size=num_items)
student_abilities = np.random.normal(0, 1, size=num_students) # for each student we have a different ability in the range of 0 to 1

c = 0.25
a = 1.0

# %%
responses = np.random.binomial(1, irt_formula(c, a, student_abilities.reshape(-1, 1), item_difficulties), size=(num_students, num_items))

# %%
print(len(responses))

# %%
print(item_difficulties)

# %%
print(student_abilities)

# %%
# Train the model
model = LinearRegression()
model.fit(responses, student_abilities)

# %%
# Test the model
test_student_abilities = np.random.normal(0, 1, size=num_students)
test_responses = np.random.binomial(1, irt_formula(c, a, test_student_abilities.reshape(-1, 1), item_difficulties), size=(num_students, num_items))

# %%
predictions = model.predict(test_responses)

# %%
print(predictions)

# %%
print(predictions)

print(student_abilities)

# %%
# Evaluate the model
mse = np.mean((predictions - test_student_abilities)**2)
rmse = np.sqrt(mse)
r2 = model.score(test_responses, test_student_abilities)

# %%
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r2)

# %% [markdown]
# --- SUMMARY  ---
# 
# ---
# In this example, we first define the IRT formula as a function. We then generate some sample data consisting of 1000 students and 50 test items with randomly generated difficulty levels and student abilities. We use the IRT formula to generate the student responses to the test items. We then train a simple linear regression model using scikit-learn's LinearRegression class on the generated data. Finally, we test the model on a separate set of randomly generated test data and evaluate its performance using mean squared error, root mean squared error, and coefficient of determination (R^2).
# 
# ---

# %% [markdown]
# ### Artificial Neural Networks (ANN)
# 
# <p>The ANN will be used to provide tailored content recommendations based on the student's academic ability level and learning preferences. <br> 
# The ANN will be trained on the preprocessed data, and it will learn to make recommendations that match with high accuracy to the student's academic ability level and learning preferences.</p>
# 
# Neural network formula
# 
# `z = f(Wx + b)`
# 
# where:
# 
# ==> z is the output of the Neural network.
# 
# ==> W is the weight matrix connecting the input layer to the hidden layer.
# 
# ==> x is the input vector representing the user's profile and the item's features.
# 
# ==> b is the bias vector added to the hidden layer.
# 
# ==> f is the activation function applied to the hidden layer output
# 

# %% [markdown]
# For the Artificial Neural Network we will use Pytorch to build a simple neural network with 2 hidden layers. 
# 
# The input layer will have 2 nodes, one for the student's academic ability level and one for the item difficulty level. 
# 
# The output layer will have 1 node, which will output the probability of the student correctly answering the item.
# 
# This will compute the output z of the ANN for the given input x. <br>
# 
# Note that the input data x should be a PyTorch tensor of appropriate size, and the output z will also be a PyTorch tensor of size output_size
# 

# %%
import torch
import torch.nn as nn

# %%
# Define the ANN class
class Recommender(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Recommender, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        hidden = self.activation(self.hidden(x))
        output = self.output(hidden)
        return output

# %%
# Set up the training data
input_data = torch.randn(100, 10)  # 100 samples with 10 features~ this woulkd be the student ability
labels = torch.randn(100, 5)  # 100 samples with 5 output classes ~ this would be the item difficulty

# %%
input_data[:1]

# %%
labels[:1]

# %%
# Set up the model and training parameters
model = Recommender(input_size=10, hidden_size=20, output_size=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%

# Train the model
for epoch in range(100):
    # Forward pass
    outputs = model(input_data)
    loss = criterion(outputs, labels)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))



# %%
# Use the model to make predictions
test_input = torch.randn(1, 10)  # 1 sample with 10 features~ this woulkd be the student ability
predicted_output = model(test_input)
print('Predicted output:', predicted_output)

# %% [markdown]
# --- SUMMARY  ---
# 
# ---
# In this example, we first define the Artificial Neural Network as a class taking in a certain number of inputs, hidden layers and output prediction. 
# 
# We then generate some sample data consisting of 100 students and 10 features and 5 labels
# 
# We then train the model using the generated data. which gives a recomendation on the student's academic ability level and learning preferences.
# 
# ---

# %% [markdown]
# ## Collaborative Filtering
# 
# Collaborative filtering algorithms are used to recommend personalized content based on user preferences and past performance. 
# 
# These algorithms analyze your behavior and the behavior of other users with similar interests to generate recommendations. 
# 
# Item response theory models are used to assess a student's academic performance level and generate personalized recommendations based on their academic performance. 

# %%
import numpy as np

# Load data into a numpy array
# this will represent the student ability rating on each item/class
data = np.array([
    [3.5, 4.0, 2.5, 3.0, 4.5],
    [2.5, 3.5, 4.0, 2.0, 3.5],
    [4.5, 2.5, 3.5, 4.0, 2.0],
    [3.0, 4.5, 2.5, 3.5, 4.0],
    [2.0, 3.0, 3.5, 4.0, 2.5]
])


# %%
# Calculate the mean rating of each item and user
item_means = np.mean(data, axis=0)
user_means = np.mean(data, axis=1)

print(
    {
        "itm_means": item_means,
        "user_means": user_means
    }
)

# %%
# Subtract the mean rating of each item and user from the data
centered_data = data - user_means[:, np.newaxis] - item_means

# %%
centered_data

# %%
# Define hyperparameters for the model
num_factors = 2
num_iterations = 100
learning_rate = 0.01
regularization = 0.1

# %%
# Initialize the latent factor matrices
item_factors = np.random.normal(scale=1/num_factors, size=(data.shape[1], num_factors))
user_factors = np.random.normal(scale=1/num_factors, size=(data.shape[0], num_factors))

# %%
# Train the model using stochastic gradient descent

# Following is a simple implementation of SGD [https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/]
for p in range(num_iterations):
    for u in range(data.shape[0]):
        for i in range(data.shape[1]):
            if data[u, i] > 0:
                error = data[u, i] - item_means[i] - user_means[u] - np.dot(item_factors[i, :], user_factors[u, :])
                item_factors[i, :] += learning_rate * (error * user_factors[u, :] - regularization * item_factors[i, :])
                user_factors[u, :] += learning_rate * (error * item_factors[i, :] - regularization * user_factors[u, :])

        print("-------------- Epoch: [", p,f"/{num_iterations} ]   ===> Error: ", error, " --------------")




# %%
# Predict the ratings for each user and item
predicted_ratings = item_means + user_means[:, np.newaxis] + np.dot(user_factors, item_factors.T)

# %%
predicted_ratings

# %%
# Get recommendations for a specific user by their user ID
user_id = 2
recommendations = np.argsort(predicted_ratings[user_id, :])[::-1]

print("Top recommendations for user {}: {}".format(user_id, recommendations))

# %% [markdown]
# --- SUMMARY  ---
# 
# ---
# 
# The stochastic gradient descent method is used to train the collaborative filtering model using the for loop. The error between the expected rating and the actual rating of each item by student is calculated by iterating over each user-item pair in the data. 
# 
# The gradient of the error with respect to each component is scaled by the learning rate and regularized to avoid overfitting before the error is used to update the latent factor matrices for the item and user. 
# 
# The loop is used to repedetly update the latent component matrices for each user and item in the data, minimizing the difference between the expected and actual ratings using stochastic gradient descent.
# 
# This is a test of the collaborative filtering model. The model is used to predict the ratings of the items that have not been rated by the users.
# 
# The predicted ratings are compared with the actual ratings to evaluate the performance of the model.
# 
# The mean squared error is used to evaluate the performance of the model. The mean squared error is calculated by taking the average of the squared difference between the predicted and actual ratings.
# 
# There are better methods like ANN we have explored in this project for the same purpose.
# ---


