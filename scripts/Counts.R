# p-value bar graphs

library(R.matlab)
library(pracma)


base_path='/media/olive/Research/oliver/pvals/' 

conds=c('V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135')
percent=c(10,20,40,60,100)
pval_threshold<-0.05

paradigm='task'
setwd(file.path(base_path,paradigm))

conds=list.files()

for (cond in conds){  
   cond_name<- gsub("\\.mat", "", cond)
   A<-readMat(cond) 
   
   L<- length(A$homo) # No of animals
   
   for (k in 1:L){
     homo<- as.numeric(unlist(A$homo[[k]]))
     hetero<- as.numeric(unlist(A$hetero[[k]]))
     
     homo_order<- order(homo)
     sum(homo[homo_order] <= pval_threshold)
     
     n_homo_tuned <- sum(homo[homo_order] <= pval_threshold)
     n_homo_nontuned<- sum(homo[homo_order] > pval_threshold)
     
     n_hetero_tuned <- sum(hetero[order(homo)][1:n_homo_tuned] <= pval_threshold)
     
     
   }
  
}


