#Battery RUL Prediction using IPSO-PF

This project implements a hybrid approach for **Remaining Useful Life (RUL)** prediction of lithium-ion batteries using:

* Dual-Exponential Degradation Model
* Particle Filter (PF)
* Improved Particle Swarm Optimization (IPSO)
* Hybrid IPSO-PF Algorithm

The implementation is based on NASA battery datasets and reproduces results similar to research literature.

---

##  Project Structure

```
battery-rul/
│
├── src/
│   ├── data_loading.py     # Load NASA .mat battery data
│   ├── model.py            # Dual-exponential model
│   ├── ipso.py             # IPSO parameter optimization
│   ├── pf.py               # Standard Particle Filter
│   ├── ipso_pf.py          # Hybrid IPSO-PF
│   ├── rul.py              # RUL prediction logic
│   ├── experiment.py       # Main experiment pipeline
│   └── plotting.py         # All plots (Figures)
│
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
└── README.md
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/battery-rul.git
cd battery-rul
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Outputs

The code generates the following figures:

### 1. Capacity Estimation

* Actual vs PF vs IPSO-PF
* Error comparison

### 2. RUL Prediction (K = 80)

* Future capacity prediction
* Failure threshold detection
* RUL estimation comparison

### 3. RUL Prediction (K = 78)

* Early prediction scenario

All figures are saved as:

* `fig_capacity_estimation.png`
* `fig_rul_prediction_k58.png`
* `fig_rul_prediction_k78.png`

---

##  Methodology

###  Dual-Exponential Model

Battery degradation is modeled as:

y_k = a e^{bk} + c e^{dk}

---

###  IPSO (Improved PSO)

* Adaptive inertia weight
* Dynamic learning factors
* Better global convergence

Used to estimate optimal model parameters.

---

###  Particle Filter (PF)

* Sequential Bayesian estimation
* Uses importance sampling
* Suffers from particle degeneracy

---

###  IPSO-PF (Proposed Method)

* Uses IPSO to guide particle updates
* Improves particle diversity
* Reduces degeneracy
* Achieves better prediction accuracy

---

##  Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Square Error)**

Comparison is performed between:

* PF
* IPSO-PF

---

## ⚠ Notes

* Ensure NASA `.mat` files are placed correctly (or synthetic data fallback is implemented).
* The `synthetic_battery()` function must be defined if real data is not available.
* This project focuses on research-level implementation rather than production deployment.

---

## References

* NASA Battery Dataset
* Research papers on IPSO-PF for battery RUL prediction

---

# battery-rul
