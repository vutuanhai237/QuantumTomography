## I. Overview

Quantum process tomography (QPT) is a foundational technique for characterizing unknown quantum operations (gates or channels). However, traditional QPT approaches often struggle with scalability, excessive resource demands, and sensitivity to noise.

We propose a methodology for quantum process tomography based on universal compilation. The main idea is to decompose a target quantum process into an optimized representation using Kraus operators and Choi matrices, leveraging compilation techniques to guide the estimation. By doing so, the method aims to reduce the computational and experimental overhead while improving accuracy and robustness.

This repository aims to provide an implementation of the universal-compilation-based QPT method, including:
- Core algorithms for decomposition and estimation
- Simulation tools and benchmarks
- Example workflows
- Utility modules for Choi/Kraus handling, noise models, and metrics

**Paper:** Advancing quantum process tomography through universal compilation
https://arxiv.org/abs/2504.14958

## II. Installation & Setup

1. Install [Git](https://git-scm.com/) or [GitHub Desktop](https://desktop.github.com/), and an editor such as Visual Studio Code.

2. Python 3.10+, download on python.org.

3. Clone the codes to your local computer (**qoop** is a core package of our various repositories) (using terminal or cmd)
```
git clone https://github.com/vutuanhai237/QuantumBattery.git
cd QuantumBattery
rm -rf qoop (delete folder qoop)
git clone https://github.com/vutuanhai237/qoop.git
```

Make sure that you see the **qoop** folder in the project

4. Install all the needed packages.
```
pip install -r requirements.txt
```
## III. Usage examples
### Random Process Tomography
This example demonstrates how to perform **Kraus-operator-based quantum process tomography (QPT)** using `qtomo`.
The goal is to train a set of **Kraus operators** that approximate a target *unitary process* (here, a random Haar unitary).

> For a full working version, see: [`src/kraus_tomography.py`](./src/kraus_tomography.py)


#### 1. Initialize the Tomography Model

Create a `KrausTomography` object for a single-qubit process with one Kraus operator and three random probe states.

```python
from qtomo.tomography import KrausTomography

# Single-qubit, 1 Kraus operator, 3 probe states
tomog = KrausTomography(num_qubits=1, num_rho=3, num_kraus=1)
tomog.set_logging(True)  # enable training progress logging
```

The **Kraus parameterization** allows modeling both *unitary* and *non-unitary* (noisy) processes, offering more flexibility than a strictly unitary Choi representation.

#### 2. Define the Target Process

We define an **ideal process** represented by a random Haar-distributed unitary, called `epsilon`.
The model will learn Kraus operators that approximate this transformation.

```python
import qtomo.generator as generator

epsilon = generator.haar(n=1)  # 1-qubit Haar-random 2x2 unitary
```

#### 3. Describe the Training Transformation

Each probe state is transformed in two steps:

1. Apply the current **Kraus operators** (the model’s learned process)
2. Apply the **inverse** of the ideal unitary `epsilon`

This setup drives the learned process toward approximating the target inverse transformation.

```python
import qtomo.quantum_ops as quantum_ops

def data_process(rho_set, kraus_ops, epsilon):
    """Apply current Kraus process, then invert epsilon."""
    rho_out = []
    for rho in rho_set:
        rho_k = quantum_ops.apply_kraus_operators(rho, kraus_ops)
        rho_final = quantum_ops.apply_unitary_dagger(rho_k, epsilon)
        rho_out.append(rho_final)
    return rho_out
```

#### 4. Train the Kraus Operators

We train for 400 epochs using the **Riemannian gradient** and **mean infidelity** as the loss function, optimized with Adam.

```python
import qtomo.gradient as gradient
import qtomo.optimizer as optimizer
import qtomo.metric as metric

opt = optimizer.AdamOptimizer(alpha=0.01)

final_kraus, loss_trace = tomog.fit(
    epochs=400,
    data_process_fn=lambda rho, K: data_process(rho, K, epsilon),
    gradient_fn=gradient.riemann,
    loss_fn=metric.mean_infidelity,
    optimizer=opt,
)

print(f"Final loss: {loss_trace[-1]:.6f}")
```

During training, the model iteratively refines its Kraus operators to minimize the distance between the processed and target probe states.

#### 5. Evaluate on Test States

After training, we test the learned process on new probe states and compare it with the ideal target `epsilon`.

```python
rho_test_set = generator.haar_probe_states(1, 6)

for rho in rho_test_set:
    rho_out, rho_target, fidelity = tomog.evaluate(
        rho_input=rho,
        target_process_fn=lambda r: quantum_ops.apply_unitary(r, unitary=epsilon),
        metric_fn=metric.compilation_trace_fidelity,
    )
    print(f"Fidelity (Kraus vs. ε): {fidelity:.6f}")
```

A fidelity close to **1.0** indicates that the learned Kraus representation successfully reproduces the target unitary dynamics.

#### Summary

| Step                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| **Initialize model** | Define a Kraus-based tomography model            |
| **Define target**    | Specify the ideal unitary process (`epsilon`)    |
| **Data process**     | Apply current Kraus + inverse of target          |
| **Train**            | Optimize Kraus operators via Riemannian gradient |
| **Evaluate**         | Compare learned and ideal transformations        |

### Quantum dephasing
This example demonstrates how to use the Choi-based quantum process tomography framework to learn and invert a single-qubit dephasing channel.

> For a full working version, see [`src/dephasing_tomography.py`](./src/dephasing_tomography.py).
#### 1. Initialize the Tomography Model

We start by creating a **ChoiTomography** object for a single qubit and a few Haar-random probe states that will be used for training.

```python
from qtomo.tomography import ChoiTomography

# One-qubit process tomography using 3 random probe states
tomog = ChoiTomography(num_qubits=1, num_rho=3)
```

The `ChoiTomography` class parameterizes a quantum process via its **unitary representation** in Choi space, allowing gradient-based optimization during training.


#### 2. Define the Training Transformation

Each probe state is:

1. Passed through a **dephasing channel** with noise strength γ
2. Then inverted by the **learned unitary** (the tomography model’s current estimate)

```python
import qtomo.quantum_ops as quantum_ops

def data_process(rho_set, unitary, gamma):
    """Apply dephasing noise followed by inverse of learned unitary."""
    result = []
    for rho in rho_set:
        rho_noisy = quantum_ops.dephasing_channel(rho, gamma)
        rho_out = quantum_ops.apply_unitary_dagger(rho_noisy, unitary)
        result.append(rho_out)
    return result
```

This function defines *how the model learns*: by attempting to undo the noise applied to the quantum states.


#### 3. Train the Model for One Noise Level

We can now train the model using an optimizer, gradient rule, and loss metric.

```python
import qtomo.gradient as gradient
import qtomo.metric as metric
import qtomo.optimizer as optimizer

# Adam optimizer with learning rate 0.01
opt = optimizer.AdamOptimizer(alpha=0.01)

# Train for one dephasing strength (γ = 0.2)
gamma = 0.2

_, loss_trace = tomog.fit(
    epochs=200,
    data_process_fn=lambda rho, U: data_process(rho, U, gamma),
    gradient_fn=gradient.riemann,
    loss_fn=metric.mean_infidelity,
    optimizer=opt,
)

print(f"Final training loss: {loss_trace[-1]:.6f}")
```

The `fit()` function automatically:

* Computes **gradients** on the unitary manifold (Riemannian gradient)
* Minimizes the **mean infidelity** between reconstructed and target states
* Updates the **unitary parameters** using the Adam optimizer

#### 4. Evaluate the Learned Process

After training, we can compare the learned unitary with the true dephasing channel using a test state.

```python
import qtomo.generator as generator

rho_test = generator.haar_probe_state(1)

rho_learned, rho_target, fidelity = tomog.evaluate(
    rho_input=rho_test,
    target_process_fn=lambda r: quantum_ops.dephasing_channel(r, gamma),
    metric_fn=metric.compilation_trace_fidelity,
)

print(f"Fidelity (Learned vs True): {fidelity:.6f}")
```

If training is successful, the fidelity will approach **1.0**, meaning the learned process effectively reproduces the dephasing behavior.
#### Summary

| Step                    | Purpose                                                      |
| ----------------------- | ------------------------------------------------------------ |
| **Initialize model**    | Define a Choi-based representation of the quantum process    |
| **Define data process** | Describe how noise and inversion are applied to probe states |
| **Train**               | Optimize the unitary to minimize mean infidelity             |
| **Evaluate**            | Compare learned and true processes via fidelity              |