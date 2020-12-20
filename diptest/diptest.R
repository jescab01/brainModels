# Title     : diptest.R
# Objective : subprocess
# Created by: Jescab01
# Created on: 05/10/2020


# library(ggplot2)
library(diptest, lib.loc=c("C:/Users/F_r_e/Documents/R/win-library/3.6", "C:/Program Files/R/R-3.6.1/library"))

setwd("C:/Users/F_r_e/PycharmProjects/TVBsim-py3.8/diptest")
#path <- commandArgs(trailingOnly = TRUE)
df=read.csv("powers.csv")
colnames(df)="x"

# ggplot(df, aes(x=x)) + 
#   geom_histogram(aes(y=..density..), colour="black", fill="white", bins = 200)+
#   geom_density(alpha=.2, fill="#FF6666") 

result=dip.test(df$x) # where p.value==0, pvalue>=2.2e-16
cat(result$p.value)

#x <- c(rnorm(50), rnorm(50) + 7)
#
### border-line bi-modal ...  BUT (most of the times) not significantly:
#dip.test(x)
