# Physics-Informed Neural Network for Modeling the Gravitational Potential of an Irregular Body
![Python](https://img.shields.io/badge/Python-3.11-blue) 
![Physics](https://img.shields.io/badge/Field-Physics%20%7C%20AI-purple) 
![Astrophysics](https://img.shields.io/badge/Domain-Astro%20%7C%20Gravitational%20Potential-red) 
![PINNs](https://img.shields.io/badge/Model-PINN-orange) 
![Deep Learning](https://img.shields.io/badge/Framework-PyTorch%20%7C%20DeepXDE-lightblue) 
![License](https://img.shields.io/badge/License-MIT-yellow) 
![Status](https://img.shields.io/badge/Status%20Complete-brightgreen)


Why do we need a better theory of gravity than Newton’s? At an observational level, it is
because Newtonian gravity fails at a certain level of accuracy, for example, for predicting the
orbit of Mercury. More conceptually, Newtonian gravity conflicts with the fundamental
principle of special relativity that no signal should travel faster than light. We will start there.
Consider a particle of mass $m$ in a gravitational field **$Φ(x, t)$**. The force it experiences is given by $F = −m∇Φ$, where the gravitational field satisfies the Poisson equation

<img align="right" alt="Coding" width="250" src="https://i.pinimg.com/736x/59/e5/1f/59e51fa2a7a205a98caa53b97c6c7168.jpg">


**$$∇2ϕ(x,y)=4πGρ(x,y)$$**


This project uses Physics-Informed Neural Networks (PINNs) to model the gravitational potential field of an irregular mass distribution (e.g., asteroid).
The model learns the potential field $φ(x, y)$ while respecting Poisson’s equation, combining physics and machine learning in a single framework.

Where:

- **$𝜙(𝑥,𝑦)$** = gravitational potential
- **$𝜌(𝑥,𝑦)$** = mass density
- **$𝐺$** = gravitational constant

*We’ll enforce this equation in the loss function, making it **“physics-informed.”***


**Tech stack:** Python, PyTorch, DeepXDE, NumPy, Matplotlib  

**Goal:** Predict gravitational potential fields accurately without traditional solvers, demonstrating physics-guided machine learning.

---
