# Preferred orientation sorting out and plotting

rm(list = ls())

library(R.matlab)
library(pracma)
library(tidyverse)
library(gridExtra)
library(reshape2)
library(ggpubr)


base_path='/media/olive/Research/oliver/prefDir'  
#save_path='/media/olive/Research/oliver/IEMdecodingForCalciumData/neuron_counts/' 

conds<- c('V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135')
percent<- c(10,20,40,60,100)
pval_threshold<-0.05
nos<-8 # number of directions

########## for TASK data  #############

paradigm<-'task'
setwd(file.path(base_path,paradigm))

conds=list.files(pattern = ".csv")

# run this only one time

for (cond in conds){   # for each condition
  cond_name<- gsub("\\_prefer.csv", "", cond)
  
  # loading preferred direction  file
  A<-read.csv(cond)    
  df<-A %>% group_by(Sub) %>% filter(Group=="homo" & Pvalue  <= 0.05) %>% count(Preference,na=TRUE)
   
  for (k in 1:L){  # for each animal   
    homo<- as.numeric(unlist(A$prefDir.homo[[k]][[1]][[1]]))
    hetero<-  as.numeric(unlist(A$prefDir.hetero[[k]][[1]][[1]]))
    
    ho_p<-as.numeric(unlist(B$homo[[k]]))
    he_p<-as.numeric(unlist(B$hetero[[k]]))
    
    Lf<- length(homo)
    
    # For homo
    temp<-data.frame(Sub=rep(paste0("Animal.",k),Lf),
                     Condition=rep(cond_name,Lf),
                     Group=rep("homo",Lf),
                     Pvalue=ho_p,
                     Preference=homo) 
    
    df<-rbind(df,temp) 
    
    
    # for hetero
    temp<-data.frame(Sub=rep(paste0("Animal.",k),Lf),
                     Condition=rep(cond_name,Lf),
                     Group=rep("hetero",Lf),
                     Pvalue=he_p,
                     Preference=hetero) 
    
    df<-rbind(df,temp) 
    df<- na.omit(df)
    
  }
  
  row.names(df)<- 1:nrow(df)
  write.csv(df,file=paste0(save_path,cond_name,'_prefer.csv')) 
  
}

########## for PASSIVE data  #############

paradigm<-'passive'
setwd(file.path(base_path,paradigm))

conds=list.files(pattern = ".csv")

# run this only one time

for (cond in conds){   # for each condition
  cond_name<- gsub("\\_prefer.csv", "", cond)
  
  # loading preferred direction  file
  A<-read.csv(cond)    
  df<-A %>% group_by(Sub) %>% filter(Group=="homo" & Pvalue  <= 0.05) %>% count(Preference,na=TRUE)
  
  for (k in 1:L){  # for each animal   
    homo<- as.numeric(unlist(A$prefDir.homo[[k]][[1]][[1]]))
    hetero<-  as.numeric(unlist(A$prefDir.hetero[[k]][[1]][[1]]))
    
    ho_p<-as.numeric(unlist(B$homo[[k]]))
    he_p<-as.numeric(unlist(B$hetero[[k]]))
    
    Lf<- length(homo)
    
    # For homo
    temp<-data.frame(Sub=rep(paste0("Animal.",k),Lf),
                     Condition=rep(cond_name,Lf),
                     Group=rep("homo",Lf),
                     Pvalue=ho_p,
                     Preference=homo) 
    
    df<-rbind(df,temp) 
    
    
    # for hetero
    temp<-data.frame(Sub=rep(paste0("Animal.",k),Lf),
                     Condition=rep(cond_name,Lf),
                     Group=rep("hetero",Lf),
                     Pvalue=he_p,
                     Preference=hetero) 
    
    df<-rbind(df,temp) 
    df<- na.omit(df)
    
  }
  
  row.names(df)<- 1:nrow(df)
  write.csv(df,file=paste0(save_path,cond_name,'_prefer.csv')) 
  
} 