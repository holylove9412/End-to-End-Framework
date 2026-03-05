import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.linalg import svd
from numpy.linalg import matrix_power, solve
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def create_graph_and_laplacian(weight_adj):
    G = nx.from_numpy_array(weight_adj)
    L = csgraph.laplacian(weight_adj,use_out_degree=True, normed=False)
    return L
def filter_func(A,x):
    U, E, VT = svd(A)
    nu =1
    N =357
    numedges = 367
    samplenodes=24
    bw = 8
    p = numedges/N
    n = samplenodes/N
    k = bw/N
    lmax = max(E)
    return np.exp(-nu*p*n*k*x/lmax)

def compute_heat_kernel(A):
    U, E, VT = svd(A)
    T_g_tmp1 = U*np.diag(filter_func(A,E))*U.T
    return T_g_tmp1

def compute_cgir(psi_i, psi_j):
    return np.sum(np.abs(psi_i * psi_j))
def select_sensors_algorithm(L, num_sensors):
    Psi = compute_heat_kernel(L)
    selected = []
    candidate_indices = list(range(len(Psi)))
    epsilon = 1e-6
    for _ in range(num_sensors):
        scores = []
        for i in candidate_indices:
            redundancy = 0
            if selected:
                redundancy = np.mean([compute_cgir(Psi[i], Psi[j]) for j in selected])
            score = np.mean(Psi)*np.linalg.norm(Psi[i], ord=1, axis=0) / (epsilon + redundancy)
            scores.append(score)
        best_idx = candidate_indices[np.argmax(scores)]
        selected.append(best_idx)
        candidate_indices.remove(best_idx)
    return selected

def compute_chebyshev_coeff(A, m, N=None, arange=[-1, 1]):

    if N is None:
        N = m + 1
    a1 = (arange[1] - arange[0]) / 2
    a2 = (arange[1] + arange[0]) / 2
    c = np.zeros(m + 1)
    for j in range(m + 1):
        x_k = np.cos(np.pi * (np.arange(1, N + 1) - 0.5) / N)
        g_k = filter_func(A,a1 * x_k + a2)
        c[j] = (2 / N) * np.sum(g_k * np.cos(np.pi * j * (np.arange(1, N + 1) - 0.5) / N))
    return c

def chebyshev_polynomial_operator(L, c, arange=[-1, 1]):

    a1 = (arange[1] - arange[0]) / 2
    a2 = (arange[1] + arange[0]) / 2
    N = L.shape[0]
    L_hat = L - a2 * sp.eye(N)
    Twf_old = sp.eye(N)
    Twf_cur = L_hat / a1
    r = 0.5 * c[0] * Twf_old + c[1] * Twf_cur
    for k in range(2, len(c)):
        Twf_new = (2 / a1) * L_hat.dot(Twf_cur) - Twf_old
        r += c[k] * Twf_new
        Twf_old = Twf_cur
        Twf_cur = Twf_new
    return r

def reconstruct_pressure_signal(Psi, f_C, selected_nodes):

    Psi = Psi.toarray() if sp.issparse(Psi) else Psi
    selected_nodes = np.array(selected_nodes)
    all_nodes = np.arange(Psi.shape[0])
    unselected_nodes = np.setdiff1d(all_nodes, selected_nodes)

    Psi_NC = Psi[unselected_nodes][:, selected_nodes]
    Psi_CC = Psi[selected_nodes][:, selected_nodes]
    Psi_CC_pinv = np.linalg.pinv(Psi_CC)
    f_R = Psi_NC @ Psi_CC_pinv @ f_C
    return f_R
def graph_localization_operator(U, g_lambda, center_node):

    N = U.shape[0]
    psi = np.zeros(N)
    for k in range(N):
        uk = U[:, k]
        psi += g_lambda[k] * np.conj(uk[center_node]) * uk
    return np.real(psi)
def calculate_Psi(L):
    eigvals, U = np.linalg.eigh(L)
    U_inv = U.T
    tau = 1
    g_lambda = np.exp(-tau * eigvals)

    PSI = np.zeros((L.shape[0],L.shape[0]))
    for i in range(L.shape[0]):
        psi_i = graph_localization_operator(U, g_lambda, i)
        PSI[i] = psi_i
    return PSI








