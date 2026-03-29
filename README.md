# Quantum Graph Coloring

A hybrid quantum-classical framework for solving the Graph Coloring Problem using classical, quantum, and quantum-inspired optimization methods.

---

## 🧠 Problem

The Graph Coloring Problem is an NP-hard combinatorial optimization problem where nodes of a graph must be colored such that no two adjacent nodes share the same color.

---

## ⚙️ Approach

This project models graph coloring as a QUBO problem and converts it into an Ising Hamiltonian for quantum optimization.

### Implemented Solvers:

* **Greedy (Classical baseline)**
* **QAOA (Quantum Approximate Optimization Algorithm)**
* **Qudit-inspired Gradient Descent**

---

## 🔄 Modes

### Comparative Mode

Runs all solvers independently and compares results.

### Hybrid Mode

Uses greedy first and invokes quantum methods only when necessary.

---

## 📊 Features

* QUBO → Ising conversion
* QAOA simulation
* Scalable qudit optimization
* Visualization:

  * Graph colorings
  * Convergence plots
  * QUBO heatmaps
  * Summary comparisons

---

## 🚀 Run the Project

### CLI

```bash
python main.py
```

### Streamlit UI (optional)

```bash
streamlit run app.py
```

---

## 📌 Example Results

* Triangle K₃ → 3 colors
* Cycle C₄ → 2 colors
* Petersen graph → 3 colors
* Random graphs → scalable results

---

## 🔬 References

* Farhi et al. (2014) — QAOA
* Oh et al. (2019) — Graph Coloring with QAOA
* Jansen et al. (2024) — Qudit-inspired optimization

---

## ⚡ Future Work

* Run on real quantum hardware (IBM Qiskit / AWS Braket)
* Explore deeper QAOA circuits
* Optimize hybrid strategies

---

## 👨‍💻 Author

Your Name
