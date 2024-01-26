source("functions.R")

lst1 <- read.table("results/jacobi.txt", header = FALSE)
lst2 <- read.table("results/without_theta.txt", header = FALSE)
lst3 <- read.table("results/with_theta.txt", header = FALSE)
matrix1 <- as.matrix(lst1)
matrix2 <- as.matrix(lst2)
matrix3 <- as.matrix(lst3)

plot_vector_norms(matrix1, matrix2, matrix3)

matrix=matrix2
dimension=1
start_index=1
end_index=10
check(matrix, dimension, start_index, end_index)