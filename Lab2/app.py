import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ---------------------------
# ACTIVATION FUNCTIONS
# ---------------------------


def step_function(x):
    return np.where(x >= 0, 1, 0)


def sigmoid_binary(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_bipolar(x):
    return (2 / (1 + np.exp(-x))) - 1


def tanh_function(x):
    return np.tanh(x)


def relu_function(x):
    return np.maximum(0, x)

# Derivatives (for backprop)


def sigmoid_derivative(x):
    s = sigmoid_binary(x)
    return s * (1 - s)


def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# ---------------------------
# VISUALIZATION
# ---------------------------


def plot_activation(func, name, x_range=(-10, 10)):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = func(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.grid(True)
    ax.set_title(f"{name} Activation Function")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    st.pyplot(fig)

# ---------------------------
# SIMPLE NEURAL NETWORK (XOR)
# ---------------------------


class SimpleNN:
    def __init__(self, activation='sigmoid', hidden_size=4, learning_rate=0.1, epochs=10000):
        self.activation_name = activation
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._set_activation()

        # XOR input and output
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize weights
        self.w1 = np.random.randn(2, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, 1)
        self.b2 = np.zeros((1, 1))

    def _set_activation(self):
        if self.activation_name == 'sigmoid':
            self.activation = sigmoid_binary
            self.derivative = sigmoid_derivative
        elif self.activation_name == 'tanh':
            self.activation = tanh_function
            self.derivative = tanh_derivative
        elif self.activation_name == 'relu':
            self.activation = relu_function
            self.derivative = relu_derivative
        else:
            raise ValueError("Unknown activation function")

    def train(self):
        for _ in range(self.epochs):
            # Forward pass
            z1 = np.dot(self.X, self.w1) + self.b1
            a1 = self.activation(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = sigmoid_binary(z2)  # Output layer is always sigmoid

            # Backpropagation
            dz2 = a2 - self.y
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = np.dot(dz2, self.w2.T) * self.derivative(z1)
            dw1 = np.dot(self.X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Update weights
            self.w2 -= self.learning_rate * dw2
            self.b2 -= self.learning_rate * db2
            self.w1 -= self.learning_rate * dw1
            self.b1 -= self.learning_rate * db1

    def predict(self, X=None):
        if X is None:
            X = self.X
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.activation(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = sigmoid_binary(z2)
        return (a2 > 0.5).astype(int)

    def evaluate(self):
        preds = self.predict()
        return accuracy_score(self.y, preds)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Exploring Activation Functions in Neural Networks")
st.markdown("This interactive demo helps visualize and compare different **activation functions** and their performance on the classic **XOR classification task** using a simple neural network.")

st.header("Activation Function Visualizations")
activation_choice = st.selectbox(
    "Select activation function to visualize",
    ["Step", "Sigmoid (Binary)", "Sigmoid (Bipolar)", "Tanh", "ReLU"]
)

if activation_choice == "Step":
    plot_activation(step_function, "Step")
elif activation_choice == "Sigmoid (Binary)":
    plot_activation(sigmoid_binary, "Sigmoid (Binary)")
elif activation_choice == "Sigmoid (Bipolar)":
    plot_activation(sigmoid_bipolar, "Sigmoid (Bipolar)")
elif activation_choice == "Tanh":
    plot_activation(tanh_function, "Tanh")
elif activation_choice == "ReLU":
    plot_activation(relu_function, "ReLU")

st.header("Neural Network XOR Training")
st.markdown("Let's train a **1-hidden-layer neural network** with different activation functions and compare accuracy on the XOR problem.")

selected_activation = st.selectbox(
    "Choose activation for hidden layer",
    ["sigmoid", "tanh", "relu"]
)

epochs = st.slider("Training Epochs", 1000, 20000, 10000, step=1000)
lr = st.slider("Learning Rate", 0.001, 1.0, 0.1)

if st.button("Train Network"):
    nn = SimpleNN(activation=selected_activation,
                  epochs=epochs, learning_rate=lr)
    nn.train()
    acc = nn.evaluate()
    st.success(
        f"Training complete with '{selected_activation.upper()}' activation. Accuracy on XOR: {acc*100:.2f}%")

    st.write("### Predictions")
    preds = nn.predict()
    st.write(f"Inputs:\n{nn.X}")
    st.write(f"Predictions:\n{preds}")
    st.write(f"True Output:\n{nn.y}")
