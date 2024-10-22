# Constructing the Z matrix from bitstring samples of an optimized QAOA circuit
def Z_matrix_from_bitstrings(bitstrings):

    num_nodes = bitstrings.shape[1]
    Z = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i <= j:
                Z[i,j] = (2*np.sum(bitstrings[:, i] == bitstrings[:, j])/len(bitstrings[:, i]) - 1) * ((i == j) - 1)
                if i != j:
                    Z[j,i] = Z[i,j]

    return Z

# Finding the best eigenvector
def find_best_eigenvector(eigenvectors, obj_function, args):

    num_eigenvectors = eigenvectors.shape[1]
    cost = obj_function(np.sign(eigenvectors[:,0]), **args)
    best_id = 0

    for i in range(num_eigenvectors-1):
        new_cost = obj_function(np.sign(eigenvectors[:,i+1]), **args)
        if new_cost < cost:
            cost = new_cost
            best_id = i+1

    return np.sign(eigenvectors[:,best_id]), cost, best_id

# The QRR algorithm
def relax_and_round(Z, obj_function, args):

    eigenvalues, eigenvectors = np.linalg.eig(Z) # step 3 of finding eigenvectors
    eigenvectors = np.concatenate((np.sign(eigenvectors), np.sign(eigenvectors)*-1), axis=1) # step 4 of sign-rounding
    best_solution, min_cost, best_id = find_best_eigenvector(eigenvectors, obj_function, args) # step 5 of finding the best eigenvector

    return best_solution, min_cost
