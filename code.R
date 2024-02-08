#' Conor Walsh, s1949139

#' Log-Exponential density
#'
#' Compute the density or log-density for a Log-Exponential (LogExp)
#' distribution
#'
#' @param x vector of quantiles
#' @param rate vector of rates
#' @param log logical; if TRUE, the log-density is returned

dlogexp <- function(x, rate = 1, log = FALSE) {
  result <- log(rate) + x - rate * exp(x)
  if (!log) {
    exp(result)
  }
  result
}

#' Log-Sum-Exp
#'
#' Convenience function for computing log(sum(exp(x))) in a
#' numerically stable manner
#'
#' @param x numerical vector

log_sum_exp <- function(x) {
  max_x <- max(x, na.rm = TRUE)
  max_x + log(sum(exp(x - max_x)))
}

# Question 1:

#' CI
#'
#' Function object for confidence interval construction:
#' 
#' @param y observations of the Poisson model
#' @param alpha significance level
#' @return a 2-element vector

CI <- function(y, alpha) { 
  n <- length(y)
  lambda_hat <- mean(y) 
  theta_interval <- lambda_hat - sqrt(lambda_hat / n) * qnorm(c(1 - alpha / 2, alpha / 2))
  pmax(theta_interval, 0)
}

#' estimate_coverage
#' 
#' @param CI the function object as defined above
#' @param N the number of simulation  replications to use for coverage estimate
#' @param alpha (1-alpha) is the intended coverage probability
#' @param n the sample size
#' @param lambda true lambda values for the poisson model

estimate_coverage <- function(CI, N, alpha, n, lambda){
  # set count to 0 before initiating loop
  count_cover = 0
  
  for(i in 1:N){
    y0 = rpois(n, lambda = lambda)
    true_alpha = CI(y0, alpha)
    
    count_cover = count_cover +  ((true_alpha[1] <= lambda) && (lambda <= true_alpha[2]))
  }
  count_cover/N
}

#' Question 2.1
#' 
#' Log-prior density:
#'
#' Log_prior_density:
#' 
#' Evaluates the logarithm of the joint prior density for the four theta_i 
#' parameters
#' 
#' @param theta a vector of length four representing the reparametrization of 
#' the parameter vector beta.
#' @param params a vector of length four containing the parameters for the prior
#' distributions.
#' @return joint_prior a single numerical value that represents the log-prior density value 
#' for the stated input values

log_prior_density <- function(theta, params){
  prior1 <- dnorm(theta[1], 0, sqrt(params[1]), log = TRUE)
  prior2 <- dnorm(theta[2], 1, sqrt(params[2]), log = TRUE)
  prior3 <- dlogexp(theta[3], rate = params[3], log = TRUE)
  prior4 <- dlogexp(theta[4], rate = params[4], log = TRUE)
  
  joint_density <- prior1 + prior2 + prior3 + prior4
}

#' Question 2.2:
#' 
#' Observation log-likelihood:
#' 
#' log_like:
#' 
#' Evaluates the observation log-likelihood for the model.
#' 
#' @param theta a vector of length four representing the reparametrization of 
#' the parameter vector beta.
#' @param x a vector representing the response variables according to the model
#' @param y a vector representing the explanatory variables according to the 
#' model
#' @return a single numerical value that represents the observed log-likelihood
#' value for the stated input values


log_like <- function(theta, x, y){
  sum(dnorm(x = y,
        mean = theta[1] + theta[2]*x,
        sd = sqrt(exp(theta[3]) + (exp(theta[4])*x^2)),
        log = TRUE))
}

#' Question 2.3:
#' 
#' Log-posterior density:
#' 
#' log_posterior_density:
#' 
#' Evaluates the logarithm of the posterior density minus some unevaluated
#' normalisation constant
#' 
#' @param theta a vector of length four representing the reparametrization of 
#' the parameter vector beta.
#' @param x a vector representing the response variables according to the model
#' @param y a vector representing the explanatory variables according to the 
#' model
#' @param params a vector of length four containing the parameters for the prior
#' distributions.
#' @return posterior_density the logarithm of the posterior density, minus some
#' unevaluated normalisation constant

log_posterior_density <- function(theta, x, y, params){
  posterior_density <- log_prior_density(theta, params) + log_like(theta, x, y)
  return(posterior_density)
}

#' Question 2.4:
#'
#'Posterior Mode:
#'
#'posterior_mode:
#'
#'Determines the mode of the log-posterior density and evaluates the Hessian at
#'the mode as well as the inverse of the negated hessian.
#'
#' @param theta_start the original starting values of theta
#' @param x a vector representing the response variables according to the model
#' @param y a vector representing the explanatory variables according to the 
#' model
#' @param params a vector of length four containing the parameters for the prior
#' distributions.
#' @return a list with the following elements:
#' mode, the posterior mode location
#' hessian, the Hessian of the log-density at the mode
#' S, the inverse of the negated Hessian at the mode      

posterior_mode <- function(theta_start, x, y, params){
  opt <- optim(par = theta_start,
               fn = log_posterior_density,
               x = filament1$CAD_Weight, y = filament1$Actual_Weight, 
               params = params,
               hessian = TRUE,
               control = list(fnscale = -1))
  mode <- opt$par
  hess <- opt$hessian
  S <- solve(-opt$hessian)
  
  name_list <- list(mode, hess, S)
  names(name_list) <- c("mode", "hessian", "S")
  
  return(name_list)
}

#' Question 2.5:
#' 
#' Gaussian Approximation:

#' See report.Rmd for code solution.

#' Question 2.6:
#' 
#' Importance Sampling:
#' 
#' do_importance:
#' 
#' computes importance sampling using a multivariate normal approximation over the importance distribution
#' 
#' @param N number of samples to generate
#' @param mu mean vector for the importance distribution
#' @param S covariance matrix
#' @param x a vector representing the response variables according to the model
#' @param y a vector representing the explanatory variables according to the 
#' model
#' @param params a vector of length four containing the parameters for the prior
#' distributions.
#' @return a dataframe with the following five columns; beta1, beta2, beta3,
#' beta4, log weights containing the beta_{i} samples.

do_importance <- function(N, mu, S, x, y, params){
  sample_gen <- rmvnorm(N, mu, S)
  log_weights <- log_prior_density(mu, params) + log_like(mu, x, y) - 
    dmvnorm(sample_gen, mu, S, log = TRUE)
  
  # exponentiate the values of beta_3 and beta_4 to reparametrise:
  
  sample_gen[,3] <- exp(sample_gen[,3])
  sample_gen[,4] <- exp(sample_gen[,4])
  
  # Normalise the weights by maximising:
  
  norm_weights <- log_weights - log_sum_exp(log_weights)
  
  # Create a dataframe:
  
  df <- cbind(sample_gen, norm_weights)
  colnames(df) <- c("beta1", "beta2", "beta3", "beta4", "log_weights")
  df <- as.data.frame(df)
  return(df)
  
}