# p-value bar graphs

library(R.matlab)
library(pracma)
library(tidyverse)
library(gridExtra)
library(reshape2)
library(ggpubr)


base_path='/media/olive/Research/oliver/pvals/' 
save_path='/media/olive/Research/oliver/IEMdecodingForCalciumData/neuron_counts/' 

conds=c('V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135')
percent=c(10,20,40,60,100)
pval_threshold<-0.05

paradigm='task'
setwd(file.path(base_path,paradigm))

conds=list.files()



for (cond in conds){   # for each condition
   cond_name<- gsub("\\.mat", "", cond)
   A<-readMat(cond) 
   
   L<- length(A$homo) # No of animals
   tuned_nontuned<-data.frame(Tuned=rep(NA,L),NonTuned=rep(NA,L))
   sub_types<-data.frame(TunedTuned=rep(NA,L),TunedNonTuned=rep(NA,L),NonTunedTuned=rep(NA,L),NonTunedNonTuned=rep(NA,L))
   
   for (k in 1:L){  # for each animal
     
     homo<- as.numeric(unlist(A$homo[[k]]))
     hetero<- as.numeric(unlist(A$hetero[[k]]))
     
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


## Combined the data frames before plotting

setwd(save_path) 
combined<- data.frame(Condition=rep(NA,1),Tuned=rep(NA,1),NonTuned=rep(NA,1))

for (cond in conds){   # for each condition
  cond_name<- gsub("\\.mat", "", cond)
  A<-read.csv(paste0(cond_name,'.csv'))
  L<- nrow(A) # No of animals
  temp<- data.frame(Condition=rep(cond_name,L),Tuned=A$Tuned,NonTuned=A$NonTuned)
  combined<-rbind(combined,temp) 
}
combined<-na.omit(combined)
combined$PercentageTuned <- round((combined$Tuned/(combined$Tuned+combined$NonTuned))*100,3)
combined$PercentageNonTuned <- round((combined$NonTuned/(combined$Tuned+combined$NonTuned))*100,3)
write.csv(combined,file=paste0(save_path,'combined.csv')) 


# combining data for subtypes

combined_subtypes<- data.frame(Condition=rep(NA,1), TunedTuned=rep(NA,1),
                                                    TunedNonTuned=rep(NA,1),
                                                    NonTunedTuned=rep(NA,1),
                                                    NonTunedNonTuned=rep(NA,1)) 

for (cond in conds){   # for each condition
  cond_name<- gsub("\\.mat", "", cond)
  A<-read.csv(paste0(cond_name,'_subtypes.csv'))
  L<- nrow(A) # No of animals
  temp<- data.frame(Condition=rep(cond_name,L),
                    TunedTuned=A$TunedTuned,
                    TunedNonTuned=A$TunedNonTuned,
                    NonTunedTuned=A$NonTunedTuned,
                    NonTunedNonTuned=A$NonTunedNonTuned)
  combined_subtypes<-rbind(combined_subtypes,temp) 
}
combined_subtypes <-na.omit(combined_subtypes)
combined_subtypes$PercentageTunedTuned <- round((combined_subtypes$TunedTuned/(combined_subtypes$TunedTuned+combined_subtypes$TunedNonTuned))*100,3)
combined_subtypes$PercentageTunedNonTuned <- round((combined_subtypes$TunedNonTuned/(combined_subtypes$TunedTuned+combined_subtypes$TunedNonTuned))*100,3)

combined_subtypes$PercentageNonTunedTuned <- round((combined_subtypes$NonTunedTuned/(combined_subtypes$NonTunedTuned+combined_subtypes$NonTunedNonTuned))*100,3)
combined_subtypes$PercentageNoTunedNonTuned <- round((combined_subtypes$NonTunedNonTuned/(combined_subtypes$NonTunedTuned+combined_subtypes$NonTunedNonTuned))*100,3)




write.csv(combined_subtypes,file=paste0(save_path,'combined_subtypes.csv')) 




## plotting
cus_cols <- c("#FF0000", "#0000FF", "#00FF00", "#FFA500", "#800080", "#FFFF00") 
lab_values<-c("V1 45","V1 90","V1 135","PPC 45","PPC 90","PPC 135")
combined$Condition<- factor(combined$Condition,levels = c("V1_45","V1_90","V1_135","PPC_45","PPC_90","PPC_135"),
                            ordered = TRUE)


p1<-ggplot(data=combined,aes(x=Condition,y=PercentageTuned,fill=Condition))+
  geom_boxplot(width=0.6,col="black",outlier.color = "red",
               outlier.shape = NA) + 
  theme_classic()+ 
  #geom_hline(yintercept=0, linetype="dashed", 
  #           color = "red", linewidth=1) + 
  geom_jitter(width=0.2,alpha=1,size=2,color="black",shape=21,fill="grey")+
  stat_summary(fun=mean, geom='point', shape=23, size=3,
               color="black",fill="magenta",alpha=0.7)+
  theme(legend.position = "none",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        plot.title = element_text(size=20,hjust = 0.5), 
        legend.title = element_blank())+ 
  xlab("")+ 
  ylab("Percentage of Tuned Neurons")  +
  scale_fill_manual(values =cus_cols,guide='none') +
  scale_y_continuous(breaks = seq(0,35,10),limits = c(-0.01,35),
                     expand = c(0,0)) +
  scale_x_discrete(labels =lab_values)  
  #scale_x_discrete(labels = function(x) str_wrap(x, width = 0.5))  # If the string is not breaking break is manually
#labs(title=c('Visibility'))  
#ggsave('EEG_visibility.eps')  
print(p1)



p2<-ggplot(data=combined,aes(x=Condition,y=PercentageNonTuned,fill=Condition))+
  geom_boxplot(width=0.6,col="black",outlier.color = "red",
               outlier.shape = NA) + 
  theme_classic()+ 
  #geom_hline(yintercept=0, linetype="dashed", 
  #           color = "red", linewidth=1) + 
  geom_jitter(width=0.2,alpha=1,size=2,color="black",shape=21,fill="grey")+
  stat_summary(fun=mean, geom='point', shape=23, size=3,
               color="black",fill="magenta",alpha=0.7)+
  theme(legend.position = "none",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        plot.title = element_text(size=20,hjust = 0.5), 
        legend.title = element_blank())+ 
  xlab("")+ 
  ylab("Percentage of Non-Tuned Neurons")  +
  scale_fill_manual(values =cus_cols,guide='none') +
  #scale_y_continuous(breaks = seq(0,35,10),limits = c(-0.01,35),
  #                   expand = c(0,0)) +
  scale_x_discrete(labels =lab_values)  
#scale_x_discrete(labels = function(x) str_wrap(x, width = 0.5))  # If the string is not breaking break is manually
#labs(title=c('Visibility'))  
#ggsave('EEG_visibility.eps')  
print(p2)

### Subtypes plot 

p1<-ggplot(data=combined_subtypes,aes(x=Condition,y=PercentageTunedTuned,fill=Condition))+
  geom_boxplot(width=0.6,col="black",outlier.color = "red",
               outlier.shape = NA) + 
  theme_classic()+ 
  #geom_hline(yintercept=0, linetype="dashed", 
  #           color = "red", linewidth=1) + 
  geom_jitter(width=0.2,alpha=1,size=2,color="black",shape=21,fill="grey")+
  stat_summary(fun=mean, geom='point', shape=23, size=3,
               color="black",fill="magenta",alpha=0.7)+
  theme(legend.position = "none",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        plot.title = element_text(size=20,hjust = 0.5), 
        legend.title = element_blank())+ 
  xlab("")+ 
  ylab("")  +
  scale_fill_manual(values =cus_cols,guide='none') +
  scale_y_continuous(breaks = seq(0,35,10),limits = c(-0.01,35),
                     expand = c(0,0)) +
  scale_x_discrete(labels =lab_values)  
#scale_x_discrete(labels = function(x) str_wrap(x, width = 0.5))  # If the string is not breaking break is manually
#labs(title=c('Visibility'))  
#ggsave('EEG_visibility.eps')  
print(p1)



p2<-ggplot(data=combined_subtypes,aes(x=Condition,y=PercentageNonTunedTuned,fill=Condition))+
  geom_boxplot(width=0.6,col="black",outlier.color = "red",
               outlier.shape = NA) + 
  theme_classic()+ 
  #geom_hline(yintercept=0, linetype="dashed", 
  #           color = "red", linewidth=1) + 
  geom_jitter(width=0.2,alpha=1,size=2,color="black",shape=21,fill="grey")+
  stat_summary(fun=mean, geom='point', shape=23, size=3,
               color="black",fill="magenta",alpha=0.7)+
  theme(legend.position = "none",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        plot.title = element_text(size=20,hjust = 0.5), 
        legend.title = element_blank())+ 
  xlab("")+ 
  ylab("")  +
  scale_fill_manual(values =cus_cols,guide='none') +
  #scale_y_continuous(breaks = seq(0,35,10),limits = c(-0.01,35),
  #                   expand = c(0,0)) +
  scale_x_discrete(labels =lab_values)  
#scale_x_discrete(labels = function(x) str_wrap(x, width = 0.5))  # If the string is not breaking break is manually
#labs(title=c('Visibility'))  
#ggsave('EEG_visibility.eps')  
print(p2)

p3<-ggplot(data=combined_subtypes,aes(x=Condition,y=PercentageTunedNoTuned,fill=Condition))+
  geom_boxplot(width=0.6,col="black",outlier.color = "red",
               outlier.shape = NA) + 
  theme_classic()+ 
  #geom_hline(yintercept=0, linetype="dashed", 
  #           color = "red", linewidth=1) + 
  geom_jitter(width=0.2,alpha=1,size=2,color="black",shape=21,fill="grey")+
  stat_summary(fun=mean, geom='point', shape=23, size=3,
               color="black",fill="magenta",alpha=0.7)+
  theme(legend.position = "none",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        plot.title = element_text(size=20,hjust = 0.5), 
        legend.title = element_blank())+ 
  xlab("")+ 
  ylab("")  +
  scale_fill_manual(values =cus_cols,guide='none') +
  #scale_y_continuous(breaks = seq(0,35,10),limits = c(-0.01,35),
  #                   expand = c(0,0)) +
  scale_x_discrete(labels =lab_values)  
#scale_x_discrete(labels = function(x) str_wrap(x, width = 0.5))  # If the string is not breaking break is manually
#labs(title=c('Visibility'))  
#ggsave('EEG_visibility.eps')  
print(p3)

p4<-ggplot(data=combined_subtypes,aes(x=Condition,y=PercentageNoTunedNonTuned,fill=Condition))+
  geom_boxplot(width=0.6,col="black",outlier.color = "red",
               outlier.shape = NA) + 
  theme_classic()+ 
  #geom_hline(yintercept=0, linetype="dashed", 
  #           color = "red", linewidth=1) + 
  geom_jitter(width=0.2,alpha=1,size=2,color="black",shape=21,fill="grey")+
  stat_summary(fun=mean, geom='point', shape=23, size=3,
               color="black",fill="magenta",alpha=0.7)+
  theme(legend.position = "none",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=20),
        axis.title.y = element_text(size=20),
        plot.title = element_text(size=20,hjust = 0.5), 
        legend.title = element_blank())+ 
  xlab("")+ 
  ylab("")  +
  scale_fill_manual(values =cus_cols,guide='none') +
  #scale_y_continuous(breaks = seq(0,35,10),limits = c(-0.01,35),
  #                   expand = c(0,0)) +
  scale_x_discrete(labels =lab_values)  
#scale_x_discrete(labels = function(x) str_wrap(x, width = 0.5))  # If the string is not breaking break is manually
#labs(title=c('Visibility'))  
#ggsave('EEG_visibility.eps')  
print(p4)



############### PLOTTING SLOPES ############

# Combining all the csv files in to a single file

setwd('~/Desktop/res/task/')
flist<-list.files(getwd())

rois<-c('V1','PPC')
conds<-c('45','90','135')
percents<-c('10','20','40','60','100')


df<-data.frame(Condition=rep(NA,1),Percent=rep(NA,1),Homo=rep(NA,1), Hetero=rep(NA,1))

for (roi in rois){
  for(cond in conds){
    for (percent in percents){
      
      cond_name=paste0(roi,'_',cond)
      fname<-paste0(roi,'_',cond,'_',percent,'.csv') 
      temp<-read.csv(fname) 
      colnames(temp)<-c("Homo","Hetero")
      temp$Condition<- rep(cond_name,nrow(temp))
      temp$Percent<-rep(percent,nrow(temp))
      df<- rbind(df,temp)  
      
      }  
  } 
}

df<-na.omit(df)
write.csv(df,file=paste0(save_path,'slopes.csv')) 

 
df<-melt(df)
df$Condition<- factor(df$Condition,levels = c("V1_45","V1_90","V1_135","PPC_45","PPC_90","PPC_135"),
                      labels = c("V1 45","V1 90","V1 135","PPC 45","PPC 90","PPC 135"),
                      ordered = TRUE)
df$Percent<- factor(df$Percent,levels = c("10","20","40","60","100"))

p <- ggplot(df, aes(x = variable, y = value, fill = Percent)) +
  geom_boxplot(position = "dodge",outlier.color = "red",
               outlier.shape = NA) + 
  #geom_jitter(width=0.1,alpha=1,size=1,color="black",shape=21,fill="grey")+
  #stat_summary(fun=mean, geom='point', shape=23, size=3,
  #             color="black",fill="magenta",alpha=0.7)+
  theme_classic()+
  theme(legend.position = "right",
        axis.ticks.length.x = unit(3,'mm'),
        axis.ticks.length.y = unit(3,'mm'), 
        axis.text = element_text(size=20),
        axis.title.x = element_text(size=24),
        axis.title.y = element_text(size=24),
        plot.title = element_text(size=24,hjust = 0.5), 
        legend.title = element_blank(), 
        strip.text.x = element_text(size = 24))+ 
  facet_wrap(~Condition, scales = "free_y") +
  labs(x = "Condition", y = "Average slope")  +
  stat_compare_means(method = "t.test", comparisons = list(c("10", "20"), c("20", "40"), c("40", "60"), c("60", "100")),
                     label = "p.format", label.y = 0.1)  # Adjust label.y for proper positioning


 

# Print the plot
print(p) 

# List of pairs for t-tests
pairs <- list(c(10, 20), c(20, 40), c(40, 60), c(60, 100),c(10,60),c(10,100))

# Loop through conditions and variables
for (condition in levels(df$Condition)) {
  for (variable in unique(df$variable)) {
    cat("Condition:", condition, "Variable:", variable, "\n")
    
    # Subset the data for the current condition and variable
    subset_data <- df[df$Condition == condition & df$variable == variable, ]
    
    # Perform t-tests for each pair
    for (pair in pairs) {
      cat("Pair:", pair, "\n")
      
      # Subset data for the current pair
      subset_pair <- subset_data[subset_data$Percent %in% pair, ]
      
      # Perform t-test
      ttest_result <- t.test(subset_pair$value ~ subset_pair$Percent)
      
      # Print the t-test result
      cat("p-value:", ttest_result$p.value, "\n")
    }
    
    cat("\n")
  }
}


