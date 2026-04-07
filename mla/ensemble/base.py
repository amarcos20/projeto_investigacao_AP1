import numpy as np

def split(X, y, value, feature=None):
    """Divide o dataset com base num valor e numa feature."""
    left_mask = X[:, feature] < value
    right_mask = X[:, feature] >= value
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def split_dataset(X, y, feature_idx, threshold):
    """Auxiliar para divisão de arrays."""
    left_idx = np.where(X[:, feature_idx] < threshold)[0]
    right_idx = np.where(X[:, feature_idx] >= threshold)[0]
    return left_idx, right_idx

def xgb_criterion(y, left_indices, right_indices, l2_reg=0, gamma=0):
    """
    Calcula o ganho de estrutura do XGBoost (Gain).
    Baseia-se na primeira e segunda derivada (Gradiente e Hessiana).
    """
    # Numa implementação simplificada de MSE, G é a soma dos erros e H é o número de amostras
    # y aqui costuma representar os gradientes/resíduos
    
    G_L = np.sum(y[left_indices])
    G_R = np.sum(y[right_indices])
    H_L = len(left_indices)
    H_R = len(right_indices)
    
    term_l = (G_L ** 2) / (H_L + l2_reg)
    term_r = (G_R ** 2) / (H_R + l2_reg)
    term_all = ((G_L + G_R) ** 2) / (H_L + H_R + l2_reg)
    
    gain = 0.5 * (term_l + term_r - term_all) - gamma
    return gain