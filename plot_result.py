import matplotlib.pyplot as plt 
from matplotlib import cm

import numpy as np 
from auxiliary import *
from pathlib import Path

# ------------------------------------------------------------------------------
## Plot the computed value function ###
def plot(name, model, param, V, axis1=0, axis2=1, zmin=-10.,
         zmax=10., numpoints = 50, pad = -15):
    path = f"saves/projects/{name}"
    
    # Make sure the plot folder exists
    Path(f'{path}/plots').mkdir(parents=True, exist_ok=True)

    eps = 0
    interval_size = param['interval_size']
    # define plotting range and mesh
    x = np.linspace(-interval_size + eps, interval_size - eps, numpoints)
    y = np.linspace(-interval_size + eps, interval_size - eps, numpoints)

    X, Y = np.meshgrid(x, y)

    s = X.shape

    Z_predicted = np.zeros(s)
    Z_data = np.zeros(s)
    statedim = param['dimension']
    DT = np.zeros((numpoints ** 2, statedim))

    # convert mesh into point vector for which the model can be evaluated
    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            DT[c, axis1] = X[i, j]
            DT[c, axis2] = Y[i, j]
            c = c + 1

    # evaluate model
    # E_predicted = model.predict(DT, verbose = 0)[:, 0] - model.predict(np.zeros((numpoints ** 2, ocp.statedim)), verbose = 0)[:, 0]
    E_predicted = model.predict(DT, verbose = 0)[:, 0]

    E_data = V.evaluate(DT)
    E_data = np.squeeze(E_data)

    # copy into plottable format
    c = 0
    for i in range(s[0]):
        for j in range(s[1]):
            Z_predicted[i, j] = E_predicted[c]
            Z_data[i, j] = E_data[c]
            c = c + 1

    ### plot the calculated values 

    # Create a 3D plot using Matplotlib
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=21, azim=121) # type: ignore
    # fig.set_size_inches(36, 16) 

    # Define a custom colormap with masked values as transparent
    cmap = plt.get_cmap('viridis')  # You can change 'viridis' to any other colormap
    cmap.set_bad('none')  # Set masked values as transparent

    # Add labels and adjust the z-axis limits
    ax.set_xlabel(r'$x_{100}$', fontsize = 28, labelpad = 15)
    ax.set_ylabel(r'$x_{200}$', fontsize = 28, labelpad = 15)
    # ax.set_xlabel(r'$x_{}$'.format(axis1+1), fontsize = 28, labelpad = 15)
    # ax.set_ylabel(r'$x_{}$'.format(axis2+1), fontsize = 28, labelpad = 15)
    #ax.set_xlabel(r'$x_{}$'.format(1), fontsize = 28, labelpad = 15)
    #ax.set_ylabel(r'$x_{}$'.format(2), fontsize = 28, labelpad = 15)
    ax.set_title(r'$W(x; \theta)$', fontsize = 28, y=1.0, pad = pad) # type: ignore
    ax.set_zlim(zmin, zmax) # type: ignore

    num_ticks = 5
    plt.xticks(np.linspace(-1, 1, num_ticks))
    plt.yticks(np.linspace(-1, 1, num_ticks))

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    ax.tick_params(axis = 'z', labelsize = 16) # type: ignore

    
    #ax.set_xticks([-1, - 0.5, 0, 0.5, 1], fontsize = 16)
    #ax.set_yticks([-1, -0.5,  0, 0.5, 1], fontsize = 16)


    #ax.plot_wireframe(X, Y, Z_data, rstride=5, cstride=5)
    # ax.plot_wireframe(X, Y, Z_data) # type: ignore

    surface = ax.plot_surface(X, Y, Z_predicted, cmap=cmap) # type: ignore

    # Save the plot as a PDF
    pdf_filename = f'{path}/plots/plot_W.pdf'
    plt.savefig(pdf_filename, format="pdf", dpi = 300, transparent=True, bbox_inches = 'tight')


    # Create a 3D plot using Matplotlib
    fig2 = plt.figure(figsize=(11, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.view_init(elev=21, azim=121) # type: ignore
    # fig2.set_size_inches(36, 16) 

    # Define a custom colormap with masked values as transparent
    cmap = plt.get_cmap('viridis')  # You can change 'viridis' to any other colormap
    cmap.set_bad('none')  # Set masked values as transparent

    # Add labels and adjust the z-axis limits
    ax2.set_xlabel(r'$x_{100}$', fontsize = 28, labelpad = 15)
    ax2.set_ylabel(r'$x_{200}$', fontsize = 28, labelpad = 15)
    # ax2.set_xlabel(r'$x_{}$'.format(axis1+1), fontsize = 28, labelpad = 15)
    # ax2.set_ylabel(r'$x_{}$'.format(axis2+1), fontsize = 28, labelpad = 15)
    #ax.set_xlabel(r'$x_{}$'.format(1), fontsize = 28, labelpad = 15)
    #ax.set_ylabel(r'$x_{}$'.format(2), fontsize = 28, labelpad = 15)
    ax2.set_title(r'$V(x)$', fontsize = 28, y=1.0, pad= pad) # type: ignore
    ax2.set_zlim(zmin, zmax) # type: ignore

    num_ticks = 5
    plt.xticks(np.linspace(-1, 1, num_ticks))
    plt.yticks(np.linspace(-1, 1, num_ticks))

    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)

    ax2.tick_params(axis = 'z', labelsize = 16) # type: ignore

    
    #ax.set_xticks([-1, - 0.5, 0, 0.5, 1], fontsize = 16)
    #ax.set_yticks([-1, -0.5,  0, 0.5, 1], fontsize = 16)


    #ax.plot_wireframe(X, Y, Z_data, rstride=5, cstride=5)
    # ax.plot_wireframe(X, Y, Z_data) # type: ignore
    # Z_error = np.square(Z_data - Z_predicted)

    ax2.plot_wireframe(X, Y, Z_data) # type: ignore

    # Save the plot as a PDF
    pdf_filename = f'{path}/plots/plot_V.pdf'
    plt.savefig(pdf_filename, format="pdf", dpi = 300, transparent=True, bbox_inches = 'tight')

    # print(Z_data)
    # print(Z_predicted)

    return

def plot_P_matrix(name, ocp, pad=8):
    path = f"saves/projects/{name}"
    
    # Make sure the plot folder exists
    Path(f'{path}/plots').mkdir(parents=True, exist_ok=True)

    norms = []
    for i in range(0, ocp.statedim):
        submatrix = ocp.P[1, i]
        norm = np.linalg.norm(submatrix)
        norms.append(norm)

    # Plot the norms
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, ocp.statedim+1), norms, marker='o')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('$i$', fontsize = 34)
    # plt.ylabel('Norm', fontsize = 28)
    plt.title(r'$|P[100,i]|$', fontsize = 34, y=1.0, pad = pad) 
    plt.grid(True)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    pdf_filename = f'{path}/plots/plot_decay_dim{ocp.statedim}_bw{ocp.bandwidth}_cont{ocp.continuous}.pdf'
    plt.savefig(pdf_filename, format="pdf", dpi = 300, transparent=True, bbox_inches = 'tight')
    return 

