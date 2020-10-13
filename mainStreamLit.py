import altair as alt
import pandas as pd
import streamlit as st

from solvers import *

st.sidebar.markdown('Parameters')
gamma = float(st.sidebar.slider('gamma', 0., 1., 0.1))
beta0 = float(st.sidebar.slider('beta0', 0., 5., 0.3))
t_end = int(st.sidebar.slider('T_end', 10, 1000, 300))
t = np.linspace(0, t_end - 1, t_end)
I0 = int(st.sidebar.text_input("Init I", 200000))
R0 = int(st.sidebar.text_input("Init R", 200000))
N = int(st.sidebar.text_input("Population", 13000000))
r = float(st.sidebar.text_input("r", 3e-1))
e = float(st.sidebar.text_input("e", 2e-6))
S0 = N - I0 - R0

if st.sidebar.checkbox('Control'):
    eta = st.sidebar.slider('eta', 0., 20., 0.5)
else:
    eta = 0

# MAIN CONTENT

st.header('CoSIR')

# PLOT OF S-I-J-R-beta
st.markdown('Select variables')
var_plot_components = {
    'variables': ['S', 'I', 'R', 'J', 'beta'],
    'defaults': [True, True, True, True, False],
    'key': [None] * 5
}
var_cols = st.beta_columns(5)
to_plot = dict()
for i in range(5):
    to_plot[var_plot_components['variables'][i]] = var_cols[i].checkbox(var_plot_components['variables'][i],
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
st.markdown('### System Evolution')
st.line_chart(df)
Istar = (r / e)
Jstar = gamma * N
phasePlotDict = {"J / JStar": J / Jstar, "I / Istar": I / Istar, "Time": t}
phasePlotDf = pd.DataFrame.from_dict(phasePlotDict)

st.markdown('### Phase Plot')
scatter_chart = st.altair_chart(
    alt.Chart(phasePlotDf)
        .mark_circle(size=60)
        .encode(x="J / JStar", y="I / Istar", color=alt.Color('Time', scale=alt.Scale(scheme='reds')))
        .properties(width=500, height=500)
)
