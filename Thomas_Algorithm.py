def get_inverse(jac_index):
    jac_matrix = jac_matrices[jac_index]
    # Access diagonal, upper diagonal and lower diagonal elements of each matrix
    diag = jac_matrix.diagonal().copy()
    upper_diag = jac_matrix.diagonal(1).copy()
    lower_diag = jac_matrix.diagonal(-1).copy()

    # Perform In-Place LU factorization using Thomas algorithm
    for j in range(1, len(diag)-1):
        diag[j] -= lower_diag[j-1] * (upper_diag[j-1]/diag[j-1])
        upper_diag[j] /= diag[j]

    # Compute the inverse of the matrix using the In-Place LU factorization
    start_time = time.time()
    inv_jac_matrix = np.zeros_like(jac_matrix)
    inv_jac_matrix[-1,-1] = 1/ diag[-1]
    for j in range(len(diag)-2, -1, -1):
        inv_jac_matrix[j,j] = 1/diag[j]
        inv_jac_matrix[j,j+1] = -upper_diag[j]
        inv_jac_matrix[j+1,j] = -lower_diag[j]
        inv_jac_matrix[j,j+1] /= diag[j+1]
        inv_jac_matrix[j+1,j+1] += upper_diag[j] * inv_jac_matrix[j,j+1]
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.4f} seconds")
    # Print the inverse matrix
    #print(f"Inverse of jac_matrix_{jac_index}: \n{inv_jac_matrix}\n")

    print(inv_jac_matrix.shape)
