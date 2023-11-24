# Compute the tuning curves and statistics as a function of stimulus direction

library(tidyverse)
library(reshape2)
library(pracma)
library(R.matlab)

data_path='/media/olive/Research/oliver/data_down'
setwd(data_path)

save_path='/media/olive/Research/oliver/IEMdecodingForCalciumData/neuron_counts/' 
paradigm<- c('task')
conds=c('V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135')  # Folder names where the raw data is stored
percents=c(10,20,40,60,100)
pval_threshold<-0.05 


# If you include the p-values then watch out for the order of the pvalues and the data loading to avoid mismatch errors
# right now in the current program it is not of concern


# dealing with paradigm on the outer-most loop 
for (para in paradigm){ 
    for(cond in conds){
      
      # List of animals in that condition
      cond_path<-file.path(data_path,paradigm,cond)
      ani_list<- list.files(cond_path)
      
      noa<- length(ani_list)
      cat('No. of animals in this condition is ', noa, '\n')
       
      for (ani in ani_list){  # For each animal in each cell compute the tuning curve
        ani_path<-file.path(cond_path,ani)
        A<- readMat(ani)
        
        # Basic info (trials and units in each condition)
        homo_shape   <- dim(A$sample.data.homo)
        hetero_shape <- dim(A$sample.data.hetero)
        
        homo_trials<- homo_shape[1] 
        homo_units<- homo_shape[2] 
        homo_timpts<- homo_shape[3] 
       
        hetero_trials<- hetero_shape[1] 
        hetero_units<- hetero_shape[2] 
        hetero_timpts<- hetero_shape[3]
        
        
        
        
        
        A$sample.data.hetero
      }  
    } 
} 
 





