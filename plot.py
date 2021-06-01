import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('./presentation.mplstyle')
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

CLR_S = 'orange'
CLR_I = 'r'
CLR_R = 'g'
CLR_J = 'b'
CLR_PREY = CLR_S
CLR_PRED = CLR_I
CLR_X = CLR_J
CLR_Y = CLR_I
CLR_beta = 'purple'

font = {'family':'serif','size':28}
plt.rc('font',**font)

def getTimePeriodV2(series):
    timestamps = []
    for i in range(1, len(series)-1):
        if(series[i]>series[i-1] and series[i] > series[i+1]):
            timestamps.append(i)
    T = []
    for i in range(1, len(timestamps)):
        T.append(timestamps[i] - timestamps[i-1])

    T = np.mean(T)
    return T

def plot_sir_withoutJ(S, I, R, t, plt_title, fig_name='sir-plot.png'):
    fig, ax = plt.subplots(figsize = (10,8))
    ax.plot(t, S, color = CLR_S, label='Susceptible (S)')
    ax.plot(t, I, color = CLR_I, label='Infectious (I)')
    ax.plot(t, R, color = CLR_R, label='Removed (R)')
    ax.set_xlabel('Time', weight = 'bold')
    ax.set_ylabel('Case counts', weight = 'bold')
#     ax.grid(linestyle='dashdot', linewidth=0.5)
    legend = ax.legend()
    fig.tight_layout()
    plt.title(plt_title)
    plt.savefig(fig_name)
    
def plot_sir(S, I, R, J, t, plt_title, fig_name='sir-plot.png'):
    fig, ax = plt.subplots()
    ax.plot(t, S, color = CLR_S, label='Susceptible (S)')
    ax.plot(t, J, color = CLR_J, label='Beta*Susceptible (J)')
    ax.plot(t, I, color = CLR_I, label='Infectious (I)')
    ax.plot(t, R, color = CLR_R, label='Removed (R)')
    ax.set_xlabel('Time', weight = 'bold')
    ax.set_ylabel('Case counts', weight = 'bold')
#     ax.grid(linestyle='dashdot', linewidth=0.5)
    legend = ax.legend()
    fig.tight_layout()
    plt.title(plt_title)
    plt.savefig(fig_name)


def plot_lv(X, Y, t, plt_title, fig_name='lv_plot.png'):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(t, X, color = CLR_X,  label='Prey (p)')
    ax.plot(t, Y, color = CLR_Y,  label='Predator (q)')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlabel('Time', weight = 'bold')
    ax.set_ylabel('Population', weight = 'bold')
#     ax.grid(linestyle='dashdot', linewidth=0.5)
    legend = ax.legend(loc='upper right')
    fig.tight_layout()
    plt.title(plt_title)
    plt.savefig(fig_name)


def plot_sir_lv(S, I, R, J, t,
                plt_title,
                fig_name='sir-lv_control.png'):
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.plot(t, S, color = CLR_S, label='Susceptible (S)')
    ax.plot(t, J, color = CLR_J, label='Beta*Susceptible (J)')
    ax.plot(t, I, color = CLR_I, label='Infectious (I)')
    ax.plot(t, R, color = CLR_R, label='Removed (R)')
    ax.tick_params(axis="y", labelsize=28)
    ax.tick_params(axis="x", labelsize=28)
    ax.set_xlabel('Time', fontsize = 28, weight = 'bold')
    ax.set_ylabel('Case counts', fontsize = 28, weight = 'bold')
#     ax.grid(linestyle='dashdot', linewidth=0.5)
#     legend = ax.legend(loc='upper right', fontsize = 20, prop={'weight': 'bold'})
    fig.tight_layout()
    plt.title(plt_title)
    plt.savefig(fig_name)
    
def plot_sir_lv_withBeta(S, I, R, J, beta, t,
                plt_title,
                fig_name='sir-lv_control.png'):
#     fig, ax = plt.subplots(figsize = (10, 8))
#     ax.plot(t, S, color = CLR_S, label='Susceptible (S)')
#     ax.plot(t, J, color = CLR_J, label='Beta*Susceptible (J)')
#     ax.plot(t, I, color = CLR_I, label='Infectious (I)')
#     ax.plot(t, R, color = CLR_R, label='Removed (R)')
#     ax.tick_params(axis="y", labelsize=18)
#     ax.tick_params(axis="x", labelsize=18)
#     ax.set_xlabel('Time', fontsize = 16)
#     ax.set_ylabel('Case counts', fontsize = 16)
#     ax.grid(linestyle='dashdot', linewidth=0.5)
#     legend = ax.legend(loc='upper right', fontsize = 16)
#     fig.tight_layout()
#     plt.title(plt_title)
#     plt.savefig(fig_name)
    
    fig, ax1 = plt.subplots(figsize = (10, 8))
    
    ax1.set_xlabel('Time', weight = 'bold')
    ax1.set_ylabel('Case counts', weight = 'bold')
    ax1.plot(t, S, color = CLR_S, label='Susceptible (S)')
    ax1.plot(t, J, color = CLR_J, label='Beta*Susceptible (J)')
    ax1.plot(t, I, color = CLR_I, label='Infectious (I)')
    ax1.plot(t, R, color = CLR_R, label='Removed (R)')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=28)
#     ax1.grid(linestyle='dashdot', linewidth=0.5)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Transmission rate ($\beta$)', color=CLR_beta, weight = 'bold')  
    ax2.plot(t, beta, color=CLR_beta, label = r'Transmission rate ($\beta$)',)
    ax2.tick_params(axis='y', labelcolor=CLR_beta)
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
#     plt.legend(handles, labels, loc= 'upper left')
    fig.tight_layout()
    plt.title(plt_title, pad=20)
    plt.savefig(fig_name)

def plot_sir_lv_oneCycle(S, I, R, betaS, beta, t, NEWS, Tperiod, fig_name):
    north, east, west, south, west1 = NEWS
    start = west
    stop = west+Tperiod
#     plt.figure(figsize = (10, 8))
#     plt.grid(linestyle='dashdot', linewidth=0.5)
#     plt.plot(t[start:stop], S[start:stop], label = "Susceptible (S)", color = CLR_S)
#     plt.plot(t[start:stop], I[start:stop], label = "Infectious (I)", color = CLR_I)
#     plt.plot(t[start:stop], R[start:stop], label = "Removed (R)", color = CLR_R)
#     plt.plot(t[start:stop], betaS[start:stop], label = "beta x S (J)", color = CLR_J)

#     plt.axvline(x=west, color = 'black',  linestyle='--')
#     plt.axvline(x=west1, color = 'black',  linestyle='--')
#     plt.axvline(x=north, color = 'black',  linestyle='--')
#     plt.axvline(x=south, color = 'black',  linestyle='--')
#     plt.axvline(x=east, color = 'black', linestyle='--')

#     plt.annotate("North", (north+0.3, 0.9*1e+7))
#     plt.annotate("South", (south+0.3, 0.9*1e+7))
#     plt.annotate("East", (east+0.3, 0.9*1e+7))
#     plt.annotate("West", (west+0.3, 0.9*1e+7))
#     plt.annotate("West", (west1+0.3, 0.9*1e+7))
#     plt.xticks(fontsize = 18)
#     plt.yticks(fontsize = 18)
    
#     plt.xlabel("Time", fontsize = 16)
#     plt.ylabel("Case counts", fontsize = 16)
#     plt.legend()
#     plt.savefig(fig_name)

    fig, ax1 = plt.subplots(figsize = (10, 8))
    
#     Snew = S[start:stop+300]
#     for i in range(stop-start, stop+300-start):
#         Snew[i] = np.nan
        
    ax1.set_xlabel('Time', weight = 'bold')
    ax1.set_ylabel('Case counts',  weight = 'bold')
    ax1.plot(t[start:stop], S[start:stop], label = "Susceptible (S)", color = CLR_S)
    ax1.plot(t[start:stop], betaS[start:stop], label = "Beta x Susceptible (J)", color = CLR_J)
    ax1.plot(t[start:stop], I[start:stop], label = "Infectious (I)", color = CLR_I)
    ax1.plot(t[start:stop], R[start:stop], label = "Removed (R)", color = CLR_R)

    ax1.tick_params(axis='y', labelcolor='black')
    
#     ax1.grid(linestyle='dashdot', linewidth=0.5)
    ax1.tick_params(axis='x', labelcolor='black')
    ax1.set_xlim(start-1, stop+2)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Transmission rate ($\beta$)', color=CLR_beta, weight='bold')  
    ax2.plot(t[start:stop], beta[start:stop], color=CLR_beta, label = r'Transmission rate ($\beta$)')
    ax2.set_ylim(0.0, 0.8)
    ax2.tick_params(axis='y', labelcolor=CLR_beta)
    
    ax1.axvline(x=west, color = 'black',  linestyle='--')
    ax1.axvline(x=west1, color = 'black',  linestyle='--')
    ax1.axvline(x=north, color = 'black',  linestyle='--')
    ax1.axvline(x=south, color = 'black',  linestyle='--')
    ax1.axvline(x=east, color = 'black', linestyle='--')
    ax1.annotate("North", (north+0.3, 0.75*1e+7), fontsize=24)
    ax1.annotate("South", (south+0.3, 0.75*1e+7), fontsize=24)
    ax1.annotate("East", (east+0.3, 0.75*1e+7), fontsize=24)
    ax1.annotate("West", (west+0.3, 0.75*1e+7), fontsize=24)
    ax1.annotate("West", (west1+0.3, 0.75*1e+7), fontsize=24)
    plt.xticks(fontsize = 28)
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
#     plt.legend(handles, labels, loc = (0.04, 0.65), fontsize = 18)
    fig.tight_layout()
    plt.savefig(fig_name)

def plot_limit_cycle(X, Y, plt_title, 
                     marker='',
                     label='Predator vs Prey', 
                     xlabel='Prey',
                     ylabel='Predator',
                     fig_name='lv_limit_cycle.png'):
    fig, ax = plt.subplots()
    ax.plot(X, Y, f'r{marker}', alpha=0.5, lw=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linestyle='dashdot', linewidth=0.5)
    legend = ax.legend(loc='upper right')
    fig.tight_layout()
    plt.title(plt_title)
    plt.savefig(fig_name)

def plot_limit_cycle_color_gradient(X, Y, plt_title, xlabel, ylabel, fig_name, xyscale, start, stop, eta, factor = 5):
    num = int(64/3)
    colorsList1 = [(1., 0.8, 0.5-(x/2+0.)/num) for x in range(num)]
    colorsList2 = [(1., 0.8-0.8*(x+0.)/num, 0) for x in range(num)]
    colorsList3 = [(1.-(x/2+0.)/num, 0, 0) for x in range(num)]
    colorsList1.extend(colorsList2)
    colorsList1.extend(colorsList3)
    CustomCmap = ListedColormap(colorsList1)
    C = np.linspace(0, 1, stop-start+1)
    cmap = CustomCmap
    plt.figure(figsize = (10, 8))
    plt.grid(linestyle='dashdot', linewidth=0.5)
    plt.yscale(xyscale)
    plt.xscale(xyscale)

    for x, y, c in zip(X[start:stop],Y[start:stop], C):
        plt.plot(x, y, 'o', c=cmap(c), markersize =  8)
    
    if(eta):
        if(xyscale == "linear"):
            plt.xlim(0.63, 1.45)
            plt.ylim(0.63, 1.45)
        if(xyscale == "log"):
            plt.xlim(0.2*1e-1, 0.9*1e+1)
            plt.ylim(0.8*1e-1, 0.9*1e+1)
            
        plt.plot(1, 1, 'x', markersize = 8, c = 'black', label = r'$\eta = $ {}'.format(eta))
        plt.legend(handlelength = 0, markerscale = 0)
    else:
        plt.plot(1, 1, 'x', markersize = 8, c = 'black')
    
    plt.annotate("Equilibrium \n (1, 1)", (1, 1), xytext = (0.95, 0.94), fontsize = 16)
    plt.xlabel("                           " + xlabel+"        Time", weight = 'bold')
    plt.ylabel(ylabel, weight = 'bold')
#     plt.xticks()
#     plt.yticks()
    norm = plt.Normalize(0, (stop-start+1)/factor)
    color = plt.colorbar(ScalarMappable(norm = norm, cmap=CustomCmap))
#     color.set_label("Time", position = (-10000, -0.03), size=18, weight = 'bold', rotation = 0)
#     color.ax.set_ylabel("Time", loc= 'bottom')
    color.ax.tick_params(labelsize=28)
    plt.title(plt_title)
    plt.savefig(fig_name)
    
def plot_single_limit_cycle_color_gradient(X, Y, startCycle, Tperiod, factor, NEWS, plt_title, xlabel, ylabel, fig_name, xyscale):
    num = int(64/3)
    colorsList1 = [(1., 0.8, 0.5-(x/2+0.)/num) for x in range(num)]
    colorsList2 = [(1., 0.8-0.8*(x+0.)/num, 0) for x in range(num)]
    colorsList3 = [(1.-(x/2+0.)/num, 0, 0) for x in range(num)]
    colorsList1.extend(colorsList2)
    colorsList1.extend(colorsList3)
    CustomCmap = ListedColormap(colorsList1)
    C = np.linspace(0, 1, Tperiod)
    cmap = CustomCmap#plt.get_cmap('white')
    # the simplest way of doing this is to just do the following:
    plt.figure(figsize = (12,9.6))
    plt.grid(linestyle='dashdot', linewidth=0.5)
    plt.yscale(xyscale)
    plt.xscale(xyscale)
    north, east, west, south = NEWS
    for x, y, c in zip(X[startCycle:startCycle+Tperiod],Y[startCycle:startCycle+Tperiod], C):
        plt.plot(x, y, 'o', c=cmap(c), markersize =  8)

        
    plt.plot(X[north], Y[north], 'X', c=cmap(C[north-startCycle]), markersize = 12, markeredgewidth=1, markeredgecolor=(0, 0, 0, 1))
    plt.annotate("North", (X[north], Y[north]), xytext = (X[north]-0.02, Y[north] - 0.08))

    plt.plot(X[west], Y[west], 'X', c=cmap(C[west-startCycle]), markersize = 12, markeredgewidth=1, markeredgecolor=(0, 0, 0, 1))
    plt.annotate("West", (X[west], Y[west]), xytext = (X[west]+0.05, Y[west]-0.005))

    plt.plot(X[south], Y[south], 'X', c=cmap(C[south-startCycle]), markersize = 12, markeredgewidth=1, markeredgecolor=(0, 0, 0, 1))
    plt.annotate("South", (X[south], Y[south]), xytext = (X[south]-0.03, Y[south] + 0.03))

    plt.plot(X[east], Y[east], 'X', c=cmap(C[east-startCycle]), markersize = 12, markeredgewidth=1, markeredgecolor=(0, 0, 0, 1))
    plt.annotate("East", (X[east], Y[east]), xytext = (X[east]-0.2, Y[east] - 0.004))

    plt.plot(1, 1, 'x', markersize = 8, c = 'black')
    plt.annotate("Equilibrium \n (1, 1)", (1, 1), xytext = (0.95, 0.82), fontsize = 20)
    plt.xlabel("                         " + xlabel+"    Time", weight = 'bold', fontsize=34.5)
    plt.ylabel(ylabel, weight = 'bold', fontsize=34.5)
#     plt.xticks(fontsize = 18)
#     plt.yticks(fontsize = 18)
    norm = plt.Normalize(west/factor, (west+Tperiod)/factor)
    color = plt.colorbar(ScalarMappable(norm = norm, cmap=CustomCmap))
#     color.set_label("Time", size=16)
    color.ax.tick_params(labelsize=34.5)
    plt.savefig(fig_name)
    plt.savefig(fig_name)

def plot_beta_I(beta, I, t, Ilim, betalim, Istar, 
                plt_title='beta/I vs t', betaMeanFlag = True,
                fig_name='beta-I-vs-t.png'):
    """
    Double axes plot with shared x-axis(time).

    Code borrowed from
    https://matplotlib.org/gallery/api/two_scales.html
    """
    factor = len(beta)/t[-1]
    
    
    fig, ax1 = plt.subplots(figsize = (12, 9.6))
    
    ax1.set_xlabel('Time', weight= 'bold', fontsize=33)
    ax1.tick_params(axis='x', labelsize = 33)
    ax1.set_ylabel('Infectious (I)', color=CLR_I, weight = 'bold', fontsize=33)
    ax1.axhline(y = Istar, color = CLR_I, ls = '--', lw = 2)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.plot(t, I, color=CLR_I)
    ax1.tick_params(axis='y', labelcolor=CLR_I, labelsize = 33)
    ax1.tick_params(axis='x',  labelsize = 33)
    if(Ilim != None):
        ax1.set_ylim(Ilim[0], Ilim[1])
#     ax1.grid(linestyle='dashdot', linewidth=0.5)
    ax2 = ax1.twinx()
    
    ax2.set_ylabel(r'Transmission rate ($\beta)$ ', color=CLR_beta, weight = 'bold')  
    ax2.plot(t, beta, color=CLR_beta)
    if(betaMeanFlag):
        Tperiod = int(getTimePeriodV2(beta))
        betaMean = []
        Ts = []
        for i in range(0, len(beta)-Tperiod, Tperiod):
            betaMean.append(np.mean(beta[i:i+Tperiod]))
            Ts.append((i + Tperiod/2)/factor)
        Ts.append(np.floor(t[-1]))
        betaMean.append(np.mean(beta[-10:]))
        ax2.plot(Ts, betaMean, color=CLR_beta, linestyle = '--', linewidth = 2)
    if(betalim != None):
        ax2.set_ylim(betalim[0], betalim[1])
    ax2.tick_params(axis='y', labelcolor=CLR_beta, labelsize = 33)
#     ax2.grid()
    
    fig.tight_layout()
    plt.title(plt_title, pad=20)
    plt.savefig(fig_name)

def plot_beta_I_oneCycle(beta, I, t, NEWS, Tperiod, fig_name):
    north, east, west, south, west1 = NEWS
    start = west
    stop = west+Tperiod
    fig, ax1 = plt.subplots(figsize = (7.5,5))
    plt.grid(linestyle='dashdot', linewidth=0.5)
    color = CLR_beta
    ax1.set_xlabel('time')
    ax1.set_ylabel('beta', color=color)
    ax1.plot(t[start:stop], beta[start:stop], color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = CLR_I
    ax2.set_ylabel('Infectious', color=color)  # we already handled the x-label with ax1
    ax2.plot(t[start:stop], I[start:stop], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.annotate("North", (north+0.3, 170000))
    ax2.annotate("South", (south+0.3, 170000))
    ax2.annotate("East", (east+0.3, 170000))
    ax2.annotate("West", (west+0.3, 170000))
    ax2.annotate("West", (west1+0.3, 170000))

    plt.axvline(x=west, color = 'black',  linestyle='--')
    plt.axvline(x=west1, color = 'black',  linestyle='--')
    plt.axvline(x=north, color = 'black',  linestyle='--')
    plt.axvline(x=south, color = 'black',  linestyle='--')
    plt.axvline(x=east, color = 'black', linestyle='--')

    plt.savefig(fig_name)


# def plot_S_I_beta(S, I, beta, t, plt_title, fig_name='S-I-beta-plot.png'):
#     fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, sharey=False)
#     markers = ['-', '-', '-']  # ['-', '-.', ':']
#     colors = ['b', 'g', 'r']
#     for i, key in enumerate(S.keys()):
#         axes[0].plot(t, S[key], markers[i], color=colors[i], alpha=0.5, lw=2, label=key)
#         axes[1].plot(t, I[key], markers[i], color=colors[i], alpha=0.5, lw=2, label=key)
#         axes[2].plot(t, beta[key], markers[i], color=colors[i], alpha=0.5, lw=2, label=key)
#     axes[2].set_xlabel('Time')
#     axes[0].set_ylabel('Case counts')
#     axes[1].set_ylabel('Case counts')
#     axes[2].set_ylabel('Values', labelpad=30)
#     axes[0].set_title('Susceptible')
#     axes[1].set_title('Infectious')
#     axes[2].set_title('Beta')
#     axes[0].ticklabel_format(style='plain')
#     for ax in axes:
#         ax.grid(linestyle='dashdot', linewidth=0.5)
#     handles, labels = axes[2].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='center right')
#     fig.tight_layout()
#     plt.title(plt_title)
#     plt.savefig(fig_name)

def plot_case_counts_line(regions):
    data = pd.read_csv('/Users/nayana/projects/covid/SEIRControl/data/case_counts.csv')
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, region in enumerate(regions):
        region_data = data[data['region'] == region[0]]
        if region[1] is None:
            sub_region_data = region_data[pd.isna(region_data['sub_region'])]
        else:
            sub_region_data = region_data[region_data['sub_region'] == region[1]]
        region_name = region[1] if isinstance(region[1], str) else region[0]
        sub_region_data.loc[:, 'date'] = pd.to_datetime(sub_region_data.loc[:, 'date'])
        sub_region_data = sub_region_data[sub_region_data['total_infected'] > 0]
        ax.plot(sub_region_data['date'], sub_region_data['new'], '-', label=region_name)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Daily Cases', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend()
    fig.tight_layout()
    plt.savefig(f'case_counts_plot.png', bbox_inches='tight')
    
    
    
def plot_SIR_dSIR_SEIR(t, S_SIR, S_delayedSIR, S_SEIR, I_SIR, I_delayedSIR, I_SEIR, R_SIR, R_delayedSIR, R_SEIR, fig_name):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    axs[0].plot(t, S_SIR, label = "Susceptible (S)", color = CLR_S)
    axs[0].plot(t, I_SIR, label = "Infectious (I)", color = CLR_I)
    axs[0].plot(t, R_SIR, label = "Removed (R)", color = CLR_R)
    axs[0].grid(linestyle='dashdot', linewidth=0.5)
    axs[0].set_title("SIR")
    axs[0].legend()

    axs[1].plot(t, S_delayedSIR, label = "Susceptible (S)", color = CLR_S)
    axs[1].plot(t, I_delayedSIR, label = "Infectious (I)", color = CLR_I)
    axs[1].plot(t, R_delayedSIR, label = "Removed (R)", color = CLR_R)
    axs[1].grid(linestyle='dashdot', linewidth=0.5)
    axs[1].set_title("delayed SIR")
    axs[1].legend()
    
    axs[2].plot(t, S_SEIR, label = "Susceptible (S)", color = CLR_S)
    axs[2].plot(t, I_SEIR, label = "Infectious (I)", color = CLR_I)
    axs[2].plot(t, R_SEIR, label = "Removed (R)", color = CLR_R)
    axs[2].grid(linestyle='dashdot', linewidth=0.5)
    axs[2].set_title("SEIR")
    axs[2].legend()
    
    plt.savefig(fig_name)
    
    
def plot_S_I_beta(S, I, beta, t, plt_title, spikes = None, fig_name='S-I-beta-plot.png'):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, sharey=False)
    markers = ['-', '-', '-', '-']  # ['-', '-.', ':']
    colors = [('b', 0.9), ('#009900', 0.7), ('#9ACD32', 0.8)]
    for i, key in enumerate(S.keys()):
        axes[0].plot(t, S[key], markers[i], color=colors[i][0], alpha=colors[i][1], lw=2.5, label=key)
        axes[1].plot(t, I[key], markers[i], color=colors[i][0], alpha=colors[i][1], lw=2.5)
        axes[2].plot(t, beta[key], markers[i], color=colors[i][0], alpha=colors[i][1], lw=2.5, label=key)
    axes[0].legend(fontsize = 20, frameon = False)
#     axes[2].set_title(r'Transmission rate ($\beta$)', fontsize = 14)
    axes[2].set_xlabel('Time (days)', fontsize = 18)
    axes[2].set_ylim(0.0, 1.1)
    axes[1].axhline(y = 150000, linestyle = '--', lw = 2, color = 'black', alpha = 0.5, label = r"$I^{target}_{avg}$")
    axes[1].legend(fontsize = 20, loc = 'upper right', numpoints = 2, frameon=False)

    axes[0].set_ylabel('S', fontsize = 24, rotation = 0, labelpad = 20, weight = 'bold')
    axes[1].set_ylabel('I', fontsize = 24, rotation = 0, labelpad = 50, weight = 'bold')
    axes[2].set_ylabel(r'$\beta$', fontsize = 24, rotation = 0, labelpad = 50, weight = 'bold')

    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].ticklabel_format(style='sci')
    axes[2].ticklabel_format(style='plain')
    axes[1].tick_params(axis="y", labelsize=24)
    axes[0].tick_params(axis="y", labelsize=24)
    axes[2].tick_params(axis="y", labelsize=24)
    for ax in axes:
        ax.grid(linestyle='dashdot', linewidth=0.5)
#     handles, labels = axes[2].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper left')
#     fig.tight_layout()
    if(spikes):
        for t, s in spikes:
            axes[0].axvline(x = t, linestyle = '--', lw = 1, color = 'black', alpha = 0.5)
            axes[1].axvline(x = t, linestyle = '--', lw = 1, color = 'black', alpha = 0.5)
            axes[2].axvline(x = t, linestyle = '--', lw = 1, color = 'black', alpha = 0.5)
#     plt.title(plt_title)
    axes[1].annotate(r"$\uparrow$ spike", (spikes[0][0], 4e+5), fontsize = 18)
    axes[1].annotate(r"$\downarrow dip$" , (spikes[1][0], 4e+5), fontsize = 18)
#     axes[1].annotate(r"$I^{target}_{avg}$" , (180, 2e+5), fontsize = 18)
    plt.xticks(fontsize=18)
    plt.savefig(fig_name)
    
    
    
def plot_I_beta(I, beta, t, plt_title, spikes = None, fig_name='S-I-beta-plot.png'):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, sharey=False)
    markers = ['-', '-', '-', '-']  # ['-', '-.', ':']
    colors = [('b', 0.9), ('#009900', 0.7), ('#9ACD32', 0.8)]
    for i, key in enumerate(I.keys()):
        axes[0].plot(t, I[key], markers[i], color=colors[i][0], alpha=colors[i][1], lw=2.5, label=key)
        axes[1].plot(t, beta[key], markers[i], color=colors[i][0], alpha=colors[i][1], lw=2.5, label=key)
#     axes[1].legend(fontsize = 20, frameon = False)
#     axes[2].set_title(r'Transmission rate ($\beta$)', fontsize = 14)
    axes[1].set_xlabel('Time (days)', fontsize = 18)
    axes[1].set_ylim(0.0, 1.1)
    axes[0].axhline(y = 150000, linestyle = '--', lw = 2, color = 'black', alpha = 0.5, label = r"$I^{target}_{avg}$")
    axes[1].legend(fontsize = 18, loc = 'upper left', numpoints = 2, frameon=False)
    axes[0].set_ylabel('S', fontsize = 20, rotation = 0, labelpad = 15, weight = 'bold')
    axes[0].set_ylabel('I', fontsize = 20, rotation = 0, labelpad = 20, weight = 'bold')
    axes[1].set_ylabel(r'$\beta$', fontsize = 20, rotation = 0, labelpad = 15, weight = 'bold')
#     axes[0].set_title('Susceptible (S)', fontsize = 14)
#     axes[1].set_title('Infectious (I)', fontsize = 14)
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[0].ticklabel_format(style='sci')
    axes[1].ticklabel_format(style='plain')
    axes[0].tick_params(axis="y", labelsize=28)
    axes[0].tick_params(axis="y", labelsize=28)
    axes[1].tick_params(axis="y", labelsize=28)
    for ax in axes:
        ax.grid(linestyle='dashdot', linewidth=0.5)
#     handles, labels = axes[2].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper left')
#     fig.tight_layout()
    if(spikes):
        for t, s in spikes:
            axes[0].axvline(x = t, linestyle = '--', lw = 1, color = 'black', alpha = 0.5)
            axes[1].axvline(x = t, linestyle = '--', lw = 1, color = 'black', alpha = 0.5)
#     plt.title(plt_title)
    axes[0].annotate(r"$\uparrow$ spike", (spikes[0][0], 4e+5), fontsize = 18)
    axes[0].annotate(r"$\downarrow dip$" , (spikes[1][0], 4e+5), fontsize = 18)
#     axes[1].annotate(r"$I^{target}_{avg}$" , (180, 2e+5), fontsize = 18)
    plt.xticks(fontsize=18)
    plt.savefig(fig_name)
    
    