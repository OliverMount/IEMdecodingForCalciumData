# Preferred orientation sorting out and plotting

rm(list = ls())

library(R.matlab)
library(pracma)
library(tidyverse)
library(gridExtra)
library(reshape2)
library(ggpubr)


base_path='/media/olive/Research/oliver/prefDir/' 
pval_path='/media/olive/Research/oliver/pvals/'
save_path='/media/olive/Research/oliver/IEMdecodingForCalciumData/neuron_counts/' 

conds=c('V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135')
percent=c(10,20,40,60,100)
pval_threshold<-0.05
nos<-8 # number of directions

paradigm='task'
setwd(file.path(base_path,paradigm))

conds=list.files()


for (cond in conds){   # for each condition
  cond_name<- gsub("\\.mat", "", cond)
  A<-readMat(cond) 
  
  L<- length(A$prefDir.homo) # No of animals
  Prefered<-data.frame(Tuned=rep(NA,L*),NonTuned=rep(NA,L))
  
  for (k in 1:L){  # for each animal 
    homo<- as.numeric(unlist(A$prefDir.homo[[k]][[1]][[1]]))
    hetero<-  as.numeric(unlist(A$prefDir.hetero[[k]][[1]][[1]]))
    
    if (length(homo)!=0){
      
    }
    
    homo_order<- order(homo) 
    n_homo_tuned <- sum(homo[homo_order] <= pval_threshold)
    n_homo_nontuned<- sum(homo[homo_order] > pval_threshold)
    
    # homo_hetero counts
    n_tuned_tuned <- sum(hetero[order(homo)][1:n_homo_tuned] <= pval_threshold)
    n_tuned_nontuned <- sum(hetero[order(homo)][1:n_homo_tuned] > pval_threshold)
    
    n_nontuned_tuned <- sum(hetero[order(homo)][n_homo_tuned+1: (length(homo_order)-n_homo_tuned)] <= pval_threshold)
    n_nontuned_nontuned <- sum(hetero[order(homo)][n_homo_tuned+1: (length(homo_order)-n_homo_tuned)] > pval_threshold)
    
    tuned_nontuned[k,1]<-n_homo_tuned
    tuned_nontuned[k,2]<-n_homo_nontuned
    
    sub_types[k,1]<-n_tuned_tuned 
    sub_types[k,2]<-n_tuned_nontuned
    sub_types[k,3]<-n_nontuned_tuned
    sub_types[k,4]<-n_nontuned_nontuned 
    
  }
  
  write.csv(tuned_nontuned,file=paste0(save_path,cond_name,'.csv'))
  write.csv(sub_types,file=paste0(save_path,cond_name,'_subtypes.csv')) 
  
}



