import numpy as np

m = 30
n = 5
matrix = np.random.normal(size=(m, n)) # 5 variables of 30 samples
means = []
for i in range(n):
    mean = np.mean(matrix[:, i])
    means.append(mean)

cov_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        i_mean = means[i]
        j_mean = means[j]
        cov = np.mean(np.multiply(matrix[:, i] - i_mean, matrix[:, j] - j_mean))
        cov_matrix[i][j] = cov

# You can observe slight error when compared to numpy result
print (np.cov(matrix[:, 0:2].T)[0][1])
print (cov_matrix[0][1])