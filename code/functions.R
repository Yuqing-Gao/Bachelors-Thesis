library(ggplot2)

plot_vector_norms <- function(matrix1, matrix2, matrix3) {

  norm1 <- apply(matrix1, 1, function(row) sqrt(sum(row^2)))
  norm2 <- apply(matrix2, 1, function(row) sqrt(sum(row^2)))
  norm3 <- apply(matrix3, 1, function(row) sqrt(sum(row^2)))
  
  x1 <- 1:nrow(matrix1)
  x2 <- 1:nrow(matrix2)
  x3 <- 1:nrow(matrix3)
  
  plot(norm1~x1,
       type = "l",
       col = "slategrey",
       xlab = "Index", ylab = "L2 Norm",
       main = "Plot of Vector Norms")
  lines(norm2~x2,
        pch = 0,
        lty = 6,
        type = "b",
        col = "tomato")
  lines(norm3~x3,
        pch = 0,
        lty = 6,
        type = "b",
        col = "turquoise")
  legend("bottomright",
         legend = c("jacobi","without theta","with theta"),
         lty = 1,
         pch = c(0, 0),
         col=c("slategrey","tomato","turquoise"),
         bty = "n",
  )
  
  print(plot)
}

check <- function(matrix, dimension, start_index, end_index) {
  selected_value <- matrix[, dimension]
  x_values <- selected_value[start_index:end_index]
  y_values <- diff(x_values)
  x_values <- x_values[-length(x_values)]
  
  data_df <- data.frame(x=x_values, y=y_values)
  
  plot <- ggplot(data_df, aes(x = x, y = y)) +
    geom_point() +
    geom_smooth(method = "lm", se = FALSE) +
    labs(title = paste("scatter plot and regression for dimension", dimension),
         x = "u_k", y = "(u_k+1 - u_k)")
  
  lm_eq <- lm(y ~ x, data = data_df)$coefficients
  intercept <- lm_eq[1]
  slope <- lm_eq[2]
  intersection_x <- -intercept / slope
  
  plot <- plot + geom_vline(xintercept = intersection_x, linetype = "dashed", color = "indianred4")
  plot <- plot + geom_text(aes(x = intersection_x, y = 0, label = intersection_x),
                           vjust = 0, color = "indianred4", size = 3)
  print(plot)
}