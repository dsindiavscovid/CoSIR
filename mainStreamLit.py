import streamlit as st
import numpy as np
import pandas as pd
from solvers import *
import altair as alt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

modes = ['About', 'Tool']
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", modes)
if(selection == 'About'):
    st.markdown("# We can write something about our work here")
    
if(selection == 'Tool'):
#     graphs = ['SJIR', 'betaI', 'Phase']
#     graphSelected = st.sidebar.radio("Display", graphs)
    gamma = float(st.sidebar.slider('gamma', 0., 1., 0.1))
    beta0 = float(st.sidebar.slider('beta0', 0., 5., 0.3))
    t_end = int(st.sidebar.slider('T_end', 10, 1000, 300))
    t = np.linspace(0, t_end-1, t_end)
    I0 = int(st.sidebar.text_input("Init I", 200000))
    R0 = int(st.sidebar.text_input("Init R", 200000))
    N = int(st.sidebar.text_input("Population", 13000000))
    r = float(st.sidebar.text_input("r", 3e-1))
    e = float(st.sidebar.text_input("e", 2e-6))
    S0 = N-I0-R0

    if st.sidebar.checkbox('Control'):
        eta = st.sidebar.slider('eta', 0., 20., 0.5)
    else:
        eta = 0
    st.markdown("## Variables to plot")
    toPlot = {}
    toPlot['S'] = st.checkbox("S", value=True, key=None)
    toPlot['I'] = st.checkbox("I", value=True, key=None)
    toPlot['J'] = st.checkbox("J", value=True, key=None)
    toPlot['R'] = st.checkbox("R", value=True, key=None)
    toPlot['beta'] = st.checkbox("beta", value=False, key=None)
    
    S, I, R, beta = solve_sir_lv_control(S0, I0, R0, beta0, N, r, e, gamma, eta, t)
    J = beta*S
    variableValues = {"S":S, "I":I, "J":J, "R":R, "beta":beta}
    graphDict = {}
    for v in toPlot:
        if(toPlot[v]):
            graphDict[v] = variableValues[v]
    
    df = pd.DataFrame.from_dict(graphDict)
    st.line_chart(df)
    phasePlotDict = {"J":J, "I":I, "Time":t}
    phasePlotDf = pd.DataFrame.from_dict(phasePlotDict)
    num = int(64/3)
    colorsList1 = [(1., 0.8, 0.5-(x/2+0.)/num) for x in range(num)]
    colorsList2 = [(1., 0.8-0.8*(x+0.)/num, 0) for x in range(num)]
    colorsList3 = [(1.-(x/2+0.)/num, 0, 0) for x in range(num)]
    colorsList1.extend(colorsList2)
    colorsList1.extend(colorsList3)
    CustomCmap = ListedColormap(colorsList1)
    C = np.linspace(0, 1, t_end+1)
    cmap = CustomCmap
#     st.write(phasePlotDf)
    
    scatter_chart = st.altair_chart(
        alt.Chart(phasePlotDf)
            .mark_circle(size=60)
            .encode(x="J", y="I", color = alt.Color('Time', scale=alt.Scale(scheme='reds')))
            .interactive()
    )   
        
    
    
       
#     if(graphSelected == 'SJIR'):
#         SIR = {"S":S, "I":I, "R":R, "J":J}
#         df = pd.DataFrame.from_dict(SIR)
#         st.line_chart(df)
#     elif(graphSelected == 'betaI'):
#         betaI = {"beta":beta, "I":I}
#         df = pd.DataFrame.from_dict(betaI)
#         st.line_chart(df)