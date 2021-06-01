# CoSIR: Managing an Epidemic via Optimal AdaptiveControl of Transmission Policy

The code for our paper **CoSIR: Managing an Epidemic via Optimal Adaptive Control of Transmission Policy**
## [CoSIR: Managing an Epidemic via Optimal Adaptive Control of Transmission Policy](https://www.medrxiv.org/content/10.1101/2020.11.10.20211995v1.full.pdf)

[Harsh Maheshwari](https://harshm121.github.io), [Shreyas Shetty](https://in.linkedin.com/in/shreyasshetty), [Nayana Bannur](https://www.linkedin.com/in/nayana-bannur/) and [Srujana Merugu](https://www.linkedin.com/in/srujana-merugu-a7243819/).


## Proposed Method
![Proposed method](figures/cosir.png)

## Web Application for CoSIR
[Link](http://cosir.herokuapp.com/)

### Code description

1. *models.py*: The epidemilogical model equations
2. *solvers.py*: Discrete and continuous solvers for the epidemilogical given inital conditions
3. *plots.py*: The scripts to create plots presented in the paper
4. *createPlots.ipynb*: Wrapper notebook and contains the parameters used to generate simulation plots
5. *baselines.ipynb*: Evaluates CoSIR against other baselines.



### Model Equations

In the *models.py* file, these are the model equations for various models:

```python
1. def sir_equations(y, t, N, beta, gamma): # SIR equations, returns derivative of S, I and R w.r.t time. 

2. def lv_equations(y, t, r, e, b, d): # LV equations, returns derivative of Prey and Predator w.r.t time. 

3. def sir_lv_equations(y, t, N, r, e, gamma): # LVSIR equations, returns derivative of S, I, R and beta w.r.t time. 

4. def sir_lv_control_equations(y, t, N, r, e, gamma, eta): # CoSIR equations, returns derivative of S, I, R and beta w.r.t time
```



### Equation Solvers

In the *solvers.py*, these functions solve the various ODEs:

```python
1. def solve_sir(S0, I0, R0, N, beta, gamma, t): # Solves SIR Equations using scipy's odeint.

2. def solve_lv(X0, Y0, r, e, b, d, t): # Solves LV Equations using scipy's odeint.

3. def solve_sir_lv(S0, I0, R0, beta0, N, r, e, gamma, t): # Solves LVSIR Equations using scipy's odeint.
 
4. def solve_sir_lv_control(S0, I0, R0, beta0, N, r, e, gamma, eta, t): # Solves CoSIR Equations using scipy's odeint.

5. def solve_discrete_sir(init, beta, gamma, N, timesteps, delta_t=0.001): # Solves SIR Equations dicretely with the given delta t value. 

6. def solve_discrete_SEIR(init, beta, gamma, sigma, N, timesteps, delta_t=0.001): # Solves SEIR Equations dicretely with the given delta t value. 
  
7. def solve_discrete_delayedSir(init, beta, gamma, N, timesteps, delta_t=0.001, tau = 4): # Solves delayedSIR Equations dicretely with the given delta t value. 
```



### Clone this repository

```
git clone https://github.com/dsindiavscovid/sir-control.git
```

## SteamLit Application

1. Install Streamlit, version 0.68.1
2. Run streamlit on your local machine using the following command:
```
streamlit run mainStreamLit.py 
```
3. Use the dynamic interface to see the plots for different parameter settings

![streamlit app screenshot](figures/streamlit.png)


## Citation
```
@article {Maheshwari2020.11.10.20211995,
	author = {Maheshwari, Harsh and Shetty, Shreyas and Bannur, Nayana and Merugu, Srujana},
	title = {CoSIR: Managing an Epidemic via Optimal Adaptive Control of Transmission Policy},
	elocation-id = {2020.11.10.20211995},
	year = {2020},
	doi = {10.1101/2020.11.10.20211995},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2020/11/13/2020.11.10.20211995},
	eprint = {https://www.medrxiv.org/content/early/2020/11/13/2020.11.10.20211995.full.pdf},
	journal = {medRxiv}
}
```
