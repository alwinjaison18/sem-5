import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---- Perceptron Helper Functions ----


def step_function(x):
    return np.where(x >= 0, 1, 0)


def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    np.random.seed(42)
    weights = np.random.randn(X.shape[1])
    bias = np.random.randn()

    for _ in range(epochs):
        for i in range(len(X)):
            linear_output = np.dot(X[i], weights) + bias
            y_pred = step_function(linear_output)
            error = y[i] - y_pred
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
    return weights, bias


def plot_decision_boundary(X, y, weights, bias, title):
    # Create a mesh grid for plotting decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = step_function(np.dot(grid, weights) + bias)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=100)
    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()

# ---- Dataset Definitions ----


datasets = {
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    "OR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    "AND-NOT": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 1, 0])),
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0]))
}

# ---- Streamlit UI ----

st.set_page_config(page_title="Logic Gates using Perceptron", page_icon="🧠")
st.title("Logic Gates using Single Layer Perceptron")
st.markdown("""
This interactive app demonstrates how a **Single Layer Perceptron** can learn the behavior of basic logic gates (AND, OR, AND-NOT), 
and why it **fails for XOR** due to non-linear separability.
""")

# Sidebar for gate selection
gate = st.sidebar.selectbox(
    "Select Logic Gate",
    ["AND", "OR", "AND-NOT", "XOR"]
)

# Sidebar for training params
learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
epochs = st.sidebar.slider("Epochs", 1, 50, 10)

# Load dataset
X, y = datasets[gate]

# Train perceptron
weights, bias = train_perceptron(X, y, learning_rate, epochs)

st.subheader(f"Selected Gate: {gate}")
st.write(f"**Trained Weights:** {weights}")
st.write(f"**Trained Bias:** {bias}")

# Display results table
st.subheader("Truth Table & Perceptron Output")
table = []
for x_i, target in zip(X, y):
    pred = step_function(np.dot(x_i, weights) + bias)
    table.append([x_i[0], x_i[1], target, int(pred)])
st.table(
    {
        "Input 1": [row[0] for row in table],
        "Input 2": [row[1] for row in table],
        "Target Output": [row[2] for row in table],
        "Perceptron Output": [row[3] for row in table],
    }
)

# Plot decision boundary (except XOR - still show failure)
plot_decision_boundary(X, y, weights, bias, f"{gate} Gate Decision Boundary")

# XOR explanation
if gate == "XOR":
    st.warning("""
    ❌ The Single Layer Perceptron cannot correctly classify XOR gate outputs because **XOR is not linearly separable**.
    
    ✅ To solve XOR, you need a **Multi-Layer Perceptron (MLP)** with at least one hidden layer and a non-linear activation function (e.g., sigmoid, ReLU).
    """)

st.markdown("---")
st.markdown(
    "Developed for Neural Network Lab • Demonstration using **Streamlit** 🚀")
