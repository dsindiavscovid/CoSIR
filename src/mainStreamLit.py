import altair as alt
import pandas as pd
import streamlit as st

from solvers import *

# SIDEBAR

st.set_page_config(page_title='CoSIR')
st.sidebar.markdown('Parameters')
gamma = float(st.sidebar.slider('gamma (Inverse of infectious period)', 0., 1., 0.1))
beta0 = float(st.sidebar.slider('beta0 (Initial transmission rate)', 0., 5., 0.3))
t_end = int(st.sidebar.slider('T_end (Simulation time)', 10, 1000, 300))
t = np.linspace(0, t_end - 1, t_end)
I0 = int(st.sidebar.text_input("Init I (Initial Infectious count)", 200000))
R0 = int(st.sidebar.text_input("Init R (Initial Recovered count)", 200000))
N = int(st.sidebar.text_input("Population", 13000000))
r = float(st.sidebar.text_input("r (LV Reproducibility)", 3e-1))
e = float(st.sidebar.text_input("e (LV Consumption rate)", 2e-6))
S0 = N - I0 - R0

if st.sidebar.checkbox('Control (CoSIR and LVSIR)'):
    eta = st.sidebar.slider('eta (Learning Rate)', 0., 20., 0.5)
else:
    eta = 0

# MAIN CONTENT

st.header('CoSIR')

# System evolution plot
st.markdown('Select variables')
var_plot_components = {
    'variables': ['S', 'I', 'R', 'J', 'beta'],
    'labels': ['Susceptible', 'Infectious', 'Removed', 'Susceptible contacts', 'Transmission rate'],
    'defaults': [True, True, True, True, False],
    'key': [None] * 5,
    'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

var_cols = st.beta_columns(5)
to_plot = dict()
for i in range(5):
    to_plot[var_plot_components['variables'][i]] = var_cols[i].checkbox(var_plot_components['labels'][i],
                                                                        value=var_plot_components['defaults'][i],
                                                                        key=var_plot_components['key'][i])

S, I, R, beta = solve_sir_lv_control(S0, I0, R0, beta0, N, r, e, gamma, eta, t)
J = beta * S
variableValues = {"S": S, "I": I, "J": J, "R": R, "beta": beta}
graphDict = {}
for v in to_plot:
    if to_plot[v]:
        graphDict[v] = variableValues[v]

df = pd.DataFrame.from_dict(graphDict)
df = df.stack().reset_index()
df.columns = ['time', 'variable', 'value']

st.markdown('### System Evolution')
sys_evol_chart = alt.Chart(df).mark_line().encode(
    x='time',
    y='value',
    color=alt.Color('variable',
                    scale=alt.Scale(
                        domain=var_plot_components['variables'],
                        range=var_plot_components['colors'])
                    ),
    tooltip=alt.Tooltip('value:Q')
).properties(
    width=700,
    height=400
)
st.altair_chart(sys_evol_chart)

# Phase plot

Istar = r / e
Jstar = gamma * N
phasePlotDict = {"J / J*": J / Jstar, "I / I*": I / Istar, "Time": t}
phasePlotDf = pd.DataFrame.from_dict(phasePlotDict)

st.markdown('### Normalised Phase Plot')
scatter_chart = st.altair_chart(
    alt.Chart(phasePlotDf)
        .mark_circle(size=60)
        .encode(x="J / J*", y="I / I*", color=alt.Color('Time', scale=alt.Scale(scheme='reds')))
        .properties(width=500, height=500)
)
