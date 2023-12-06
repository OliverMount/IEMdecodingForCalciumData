# Population decoding  of calcium imaging data

import sys
import os
 
import numpy as np 
import pandas as pd
import multiprocessing as mp  # For parallel processing
 
import re	
import time	

from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy.io import loadmat,savemat   
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore 

import matplotlib
import matplotlib.pyplot as plt  
matplotlib.rcParams['axes.linewidth'] = 2

# Load local modules
scripts_dir='/media/olive/Research/oliver/IEMdecodingForCalciumData/scripts/'
os.chdir(scripts_dir)
sys.path.append(os.getcwd())
from utils import *	 # it imports the decoder as well

from mne.stats import permutation_cluster_test,permutation_cluster_1samp_test
import mne 

from sklearn import *
from sklearn.model_selection import LeaveOneOut 
from sklearn.utils.validation import check_is_fitted 

# Data path and parameter settings  
data_path =  '/media/olive/Research/oliver/data_down/' 
#p-values
pval_pref_path='/media/olive/Research/oliver/prefDir'


paradigm = 'task'   
data_path_task =  os.path.join(data_path,paradigm)  

paradigm = 'passive' 
data_path_passive =   os.path.join(data_path,paradigm)

decoding_res_path = '../pop_decoding/'
decoding_res_data_path=os.path.join(decoding_res_path,'tuning_curves')
decoding_res_fig_path=os.path.join(decoding_res_path ,'plots')  
decoding_res_slopes_path=os.path.join(decoding_res_path ,'slopes') 

task_save_path=os.path.join(decoding_res_data_path,'task')
passive_save_path=os.path.join(decoding_res_data_path,'passive')

create_dir(decoding_res_path) 
create_dir(os.path.join(decoding_res_data_path))
create_dir(task_save_path)
create_dir(passive_save_path)
create_dir(decoding_res_fig_path) 
create_dir(decoding_res_slopes_path) 


os.chdir(decoding_res_path)
decoding_res_path=os.getcwd() 
decoding_res_data_path=os.path.join(decoding_res_path,'tuning_curves')
decoding_res_fig_path=os.path.join(decoding_res_path ,'plots')  
decoding_res_slopes_path=os.path.join(decoding_res_path ,'slopes') 

task_save_path=os.path.join(decoding_res_data_path,'task')
passive_save_path=os.path.join(decoding_res_data_path,'passive')

 
  
# Adjustable parameter settings for decoding  
fs=20	 # sampling frequency 
ts=1/fs   # sampling time

#Data parameter settings
analyse_time=[-1,5] # sec 
time_values=np.arange(analyse_time[0],analyse_time[1],ts)  
nt=len(time_values)

angles=np.arange(22.5,360,45) # Stimulus angles
ns=len(angles)   # no. of stimulus values 
# Parameters for slope computation
tt=np.arange(-1,5,ts)
trun=2
sig=1
wrap_around=5
slope_angles=[-180,-135,-90,-45,0]
  
Gaussian_smoothening_needed=False
time_resolved_tuning_desired=False
iteration_included=False

nt=120   # 120 time-points for 20 Hz

# parameters for cluster computation
nperm,tail,cluster_alpha=5000,1,0.05
xmin=-0.5
xmax=4

ymin=-0.05
ymax=0.33
# for task
first_sig_task=0.32
second_sig_task=0.31
diff_sig_task=0.30
# for passive
first_sig_passive=0.29
second_sig_passive=0.28
diff_sig_passive=0.27

"""
ymin=-0.05
ymax=0.25 
first_sig=0.24
second_sig=0.23
diff_sig=0.22
"""

lwd=3
plt_lwd=3
alp=0.2


ROIs_hetero=['V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135']

for k in ROIs_hetero:
	create_dir(os.path.join(decoding_res_data_path,'task',k))
	create_dir(os.path.join(decoding_res_data_path,'passive',k)) 
 
# Percentage of cells to use for decoding (tuned + % of untuned cells)
percent_data=[0,10,20,40,60,100]	 
#########################################################################################
############################  ANALYSIS FOR TASK DATA #####################################
#########################################################################################
 
paradigm='task' 
# mouse name order
V1_task=['1R.mat','2L.mat','newV1.mat','Ylbd.mat','Ylbe.mat','Ylbf.mat','Ylbg.mat','2.mat',	'3.mat','4.mat']
PPC_task=['3L.mat','3R.mat','YLbc.mat','YLcj.mat','YLck.mat','YLcl.mat','YLcm.mat']

for roi in ROIs_hetero:   # For each heterogeneous condition
#for roi in ['V1_45']:	
	os.chdir(data_path_task)
	print_status('Dealing with ROI: ' + roi)
	
	os.chdir(roi)
	
	ani_list=os.listdir()
	print(ani_list)
	noa=len(ani_list)
	
	print_status('No. of animals in ' + roi + ' is ' + str(noa)) 
	
	# load the neuron preferences and pvalue for all animals 
	# The csv file is created in R (pls. look at the script PrefDirection.R in the scripts folder)
	B = pd.read_csv(os.path.join(pval_pref_path, paradigm,roi+'_prefer.csv'))
	  
	if roi.startswith('V1'):
		list_for_sorting=V1_task
	else:
		list_for_sorting=PPC_task 
	
	ani_list=[[file for file in ani_list if file.lower().endswith(suffix.lower())] for suffix in list_for_sorting]
	ani_list= [ani[0] for ani in ani_list] 
	 
	for p in range(len(ani_list)):   # For each animal
	#for p in range(1):	
		# load that animal data in homo and hetero group  
		# This is not sorted yet using p-value
		df_homo=B[(B['Sub']=='Animal.'+str(p+1))  & (B['Group'] =='homo')]
		df_hetero=B[(B['Sub']=='Animal.'+str(p+1))  & (B['Group'] =='hetero')] 
		
		# resetting the index is needed after subsetting the data
		df_homo.reset_index(drop=True, inplace=True)
		df_hetero.reset_index(drop=True, inplace=True)  # reset the indices
		 
		# arrange accoring to the p-value first
		idx_homo=np.argsort(df_homo['Pvalue'])
		df_homo_sorted=df_homo.loc[idx_homo]
		df_hetero_sorted=df_hetero.loc[idx_homo] 
		 
		# to get a data frame that is tuned in both the condition
		# finding indices that match both conditions
		final=pd.merge(df_homo_sorted[df_homo_sorted['Pvalue'] <= 0.05], df_hetero_sorted[df_hetero_sorted['Pvalue'] <= 0.05], left_index=True, right_index=True)
		#print(final.shape)
		TunedIndices=final.index.to_numpy() 
		
		
		# pref_df in only for tuned-tuned
		pref_df = final[['Preference_x','Preference_y']]
		pref_df = pref_df.copy()  # to avoid the copy warnings
		pref_df['Neuron'] = final.index.to_list()
		pref_df.rename(columns={'Preference_x': 'Pref.Homo', 'Preference_y': 'Pref.Hetero'}, inplace=True)
		pref_df.reset_index(drop=True, inplace=True) 
		 
		print_status('Dealing with the animal ' + ani_list[p])
		
		A=loadmat(ani_list[p])	 # Load the data
		
		# homo and hetero data
		homo_data=A['sample_data_homo']	   #orig.data:	 trials X units X time-pts
		homo_data=homo_data.transpose(1,0,2)   #for decoding:  units X trials X time-pts  
		
		hetero_data=A['sample_data_hetero']
		hetero_data =hetero_data.transpose(1,0,2)  
		
		# homo and hetero data labels
		homo_labels=np.squeeze(A['dirIdx_homo'])  
		hetero_labels=np.squeeze(A['dirIdx_hetero'])  
		
		del A
		
		# Shuffle labels and data before decoding 
		# Not necessary for population decoding
		idx=np.random.permutation(len(homo_labels))
		homo_labels=homo_labels[idx]
		homo_data=homo_data[:,idx,:]
		
		idx=np.random.permutation(len(hetero_labels))
		hetero_labels=hetero_labels[idx]
		hetero_data=hetero_data[:,idx,:] 
		
		for pp in percent_data: 
		
			path_name=os.path.join(decoding_res_data_path,paradigm,roi,str(pp))
			create_dir(path_name)
			fname=path_name+'/Mouse'+str(p+1) + '.npy'  
			
			if not os.path.exists(fname):

				# Get the indices of the other neurons and use them for decoding
				# Only homo data is enough to sort out the neurons
				# as we always choose homotuned neurons and choose the same in the 
				# hetero condition
				if pp!=0:  # Other than tuned on both homo and hetero 
					 
					 
					# first remove the tuned tuned one from the original df
					df_homo_sorted_pp=df_homo_sorted.drop(TunedIndices) 
					df_hetero_sorted_pp=df_hetero_sorted.drop(TunedIndices)
					
					# additional neurons
					nper_cells = int(np.ceil((pp/100)*df_homo_sorted_pp.shape[0])) # get the number of cells 
					
					additional_neurons=df_homo_sorted_pp[:nper_cells].index.to_numpy()
					 
					
					# Neurons used for decoding
					NeuronIndices=np.concatenate((TunedIndices,additional_neurons))  
					
					update_df= pd.DataFrame()
					update_df['Pref.Homo']=df_homo_sorted_pp[:nper_cells]['Preference']
					update_df['Pref.Hetero']=df_hetero_sorted_pp[:nper_cells]['Preference']
					
					update_df['Neuron']=df_homo_sorted_pp[:nper_cells].index
					
					# update the  pref_df (to include the new additional neurons)
					PrefDirInfo=pd.concat([pref_df, update_df], ignore_index=True)
					
				else:
					NeuronIndices=TunedIndices
					PrefDirInfo=pd.DataFrame()
					PrefDirInfo=pref_df
				
				homo_data_p=homo_data[NeuronIndices,:,:]  # arranging homo data accroding to sorted pvalue 
				hetero_data_p=hetero_data[NeuronIndices,:,:]  # arranging hetero data accroding to sorted pvalue 
				
				print_status('Homo. shape before decoding is ' + str(homo_data_p.shape))
				print_status('Hetero. shape before decoding is ' + str(hetero_data_p.shape))   
 
				# preference direction computation from data 
				for neu_idx in range(homo_data_p.shape[0]): 
					PrefDirInfo['Pref.Homo'][neu_idx]=get_preference(homo_data_p[neu_idx,:,:],homo_labels)[0]			   
					PrefDirInfo['Pref.Hetero'][neu_idx]=get_preference(hetero_data_p[neu_idx,:,:],hetero_labels)[0]
				
				# Parallel decoding begings here
				st = time.time() 
				A=run_parallel_the_pop_decoding(homo_data_p,hetero_data_p,homo_labels,hetero_labels,PrefDirInfo,nt)  # (Homo result, Hetero. result)
				ed = time.time()
					
				elapsed_time = ed - st
				print_status('Execution time: ' + str(elapsed_time) + ' for animal ' + str(p))   
					 
				print('Shape of A is ', A.shape)
				print_status('Saving the tuning curves') 
				path_name=os.path.join(decoding_res_data_path,paradigm,roi,str(pp))
				create_dir(path_name)
				fname=path_name+'/Mouse'+str(p+1) + '.npy'
				print(fname)
							
				np.save(fname,np.squeeze(A))
				del A 
					
				print_status('Done with ' + str(p+1) + '/' + str(noa) + ' for the percentage '+ str(pp),'') 
					 
			else:
				print_status('Already done with ' + str(p+1) + '/' + str(noa) + ' for the percentage '+ str(pp),'') 

# Centering and computing slopes for task case
 
# Slope dynamics computation and storing the results
for roi in ROIs_hetero:   # For each condition  
	for pp in percent_data: # For each percentage of data 
		
		os.chdir(os.path.join(decoding_res_data_path, paradigm))
		print_status('Computing slopes for  ROI' + roi +' for the percentage  ',str(pp)) 
	
		os.chdir(os.path.join(roi,str(pp)))

		ani_list=os.listdir()
		noa=len(ani_list)

		print_status('No. of animals in ' + roi + ' is ' + str(noa))

		st_roi = time.time()

		slopes=np.zeros((noa,nt,2))  # 2 conditions

		for p in range(len(ani_list)):   # For each animal
		
			save_dir=os.path.join(decoding_res_slopes_path,paradigm,roi,str(pp))
			create_dir(save_dir) 
			fname=os.path.join(save_dir,'slopes.npy')
			
			if not os.path.exists(fname): 

				A=np.load(ani_list[p])  # Load the tuning curve data  
				B=zero_center_the_tcs(A,shift=wrap_around)
				C=B.transpose((0,2,1))
				D=avg_across_zero_centered_tcs(C, shift=wrap_around)

				if Gaussian_smoothening_needed:
					print_status('Gaussian smoothening desired!') 
					S=Gaussian_smoothener(D,sig=sig,trun=trun ,ts=ts) 
				else:
					print_status('NO Gaussian smoothening desired!') 
					S=D  

				## Estimate the slope 
				# Homo-case
				print_status('Computing slopes..') 
				for time_pts in range(nt): 
					slopes[p,time_pts,0]=esti_slope(slope_angles,S[:,time_pts,0],intercept=True, standardise=False) 

				# Hetero-case 
				for time_pts in range(nt): 
					slopes[p,time_pts,1]=esti_slope(slope_angles,S[:,time_pts,1],intercept=True, standardise=False)   
				 
				
			else:
				print_status('Already done with slope computations') 
				
		np.save(fname,slopes)   
		
############## passive data decoding   ################## 
 

paradigm='passive' 
# mouse name order
V1_passive=['1L.mat','1R.mat','2L.mat','Ylce.mat','Ylcf.mat','Ylcg.mat']
PPC_passive=['3L.mat','3R.mat','Ylcb.mat', 'Ylch.mat','Ylci.mat','YLck.mat','Ylcl.mat','Ylcm.mat']

for roi in ROIs_hetero:   # For each heterogeneous condition
#for roi in ['V1_45']:	
	os.chdir(data_path_passive)
	print_status('Dealing with ROI: ' + roi)
	
	os.chdir(roi)
	
	ani_list=os.listdir()
	print(ani_list)
	noa=len(ani_list)
	
	print_status('No. of animals in ' + roi + ' is ' + str(noa)) 
	
	# load the neuron preferences and pvalue for all animals 
	# The csv file is created in R (pls. look at the script PrefDirection.R in the scripts folder)
	B = pd.read_csv(os.path.join(pval_pref_path, paradigm,roi+'_prefer_passive.csv'))
	  
	if roi.startswith('V1'):
		list_for_sorting=V1_passive
	else:
		list_for_sorting=PPC_passive 
	
	ani_list=[[file for file in ani_list if file.lower().endswith(suffix.lower())] for suffix in list_for_sorting]
	ani_list= [ani[0] for ani in ani_list] 
	 
	for p in range(len(ani_list)):   # For each animal
	#for p in range(1):	
		# load that animal data in homo and hetero group  
		# This is not sorted yet using p-value
		df_homo=B[(B['Sub']=='Animal.'+str(p+1))  & (B['Group'] =='homo')]
		df_hetero=B[(B['Sub']=='Animal.'+str(p+1))  & (B['Group'] =='hetero')] 
		
		# resetting the index is needed after subsetting the data
		df_homo.reset_index(drop=True, inplace=True)
		df_hetero.reset_index(drop=True, inplace=True)  # reset the indices
		 
		# arrange accoring to the p-value first
		idx_homo=np.argsort(df_homo['Pvalue'])
		df_homo_sorted=df_homo.loc[idx_homo]
		df_hetero_sorted=df_hetero.loc[idx_homo] 
		 
		# to get a data frame that is tuned in both the condition
		# finding indices that match both conditions
		final=pd.merge(df_homo_sorted[df_homo_sorted['Pvalue'] <= 0.05], df_hetero_sorted[df_hetero_sorted['Pvalue'] <= 0.05], left_index=True, right_index=True)
		TunedIndices=final.index.to_numpy()  
		
		# pref_df in only for tuned-tuned
		pref_df = final[['Preference_x','Preference_y']]
		pref_df = pref_df.copy()  # to avoid the copy warnings
		pref_df['Neuron'] = final.index.to_list()
		pref_df.rename(columns={'Preference_x': 'Pref.Homo', 'Preference_y': 'Pref.Hetero'}, inplace=True)
		pref_df.reset_index(drop=True, inplace=True) 
		 
		print_status('Dealing with the animal ' + ani_list[p])
		
		A=loadmat(ani_list[p])	 # Load the data
		
		# homo and hetero data
		homo_data=A['sample_data_homo']	   #orig.data:	 trials X units X time-pts
		homo_data=homo_data.transpose(1,0,2)   #for decoding:  units X trials X time-pts  
		
		hetero_data=A['sample_data_hetero']
		hetero_data =hetero_data.transpose(1,0,2)  
		
		# homo and hetero data labels
		homo_labels=np.squeeze(A['dirIdx_homo'])  
		hetero_labels=np.squeeze(A['dirIdx_hetero'])  
		
		del A
		
		# Shuffle labels and data before decoding 
		# Not necessary for population decoding
		idx=np.random.permutation(len(homo_labels))
		homo_labels=homo_labels[idx]
		homo_data=homo_data[:,idx,:]
		
		idx=np.random.permutation(len(hetero_labels))
		hetero_labels=hetero_labels[idx]
		hetero_data=hetero_data[:,idx,:] 
		
		for pp in percent_data: 
		
			path_name=os.path.join(decoding_res_data_path,paradigm,roi,str(pp))
			create_dir(path_name)
			fname=path_name+'/Mouse'+str(p+1) + '.npy'  
			
			if not os.path.exists(fname):

				# Get the indices of the other neurons and use them for decoding
				# Only homo data is enough to sort out the neurons
				# as we always choose homotuned neurons and choose the same in the 
				# hetero condition
				if pp!=0:  # Other than tuned on both homo and hetero 
					 
					 
					# first remove the tuned tuned one from the original df
					df_homo_sorted_pp=df_homo_sorted.drop(TunedIndices) 
					df_hetero_sorted_pp=df_hetero_sorted.drop(TunedIndices)
					
					# additional neurons
					nper_cells = int(np.ceil((pp/100)*df_homo_sorted_pp.shape[0])) # get the number of cells 
					
					additional_neurons=df_homo_sorted_pp[:nper_cells].index.to_numpy()
					 
					
					# Neurons used for decoding
					NeuronIndices=np.concatenate((TunedIndices,additional_neurons))  
					
					update_df= pd.DataFrame()
					update_df['Pref.Homo']=df_homo_sorted_pp[:nper_cells]['Preference']
					update_df['Pref.Hetero']=df_hetero_sorted_pp[:nper_cells]['Preference']
					
					update_df['Neuron']=df_homo_sorted_pp[:nper_cells].index
					
					# update the  pref_df (to include the new additional neurons)
					PrefDirInfo=pd.concat([pref_df, update_df], ignore_index=True)
					
				else:
					NeuronIndices=TunedIndices
					PrefDirInfo=pd.DataFrame()
					PrefDirInfo=pref_df
				
				homo_data_p=homo_data[NeuronIndices,:,:]  # arranging homo data accroding to sorted pvalue 
				hetero_data_p=hetero_data[NeuronIndices,:,:]  # arranging hetero data accroding to sorted pvalue 
				
				print_status('Homo. shape before decoding is ' + str(homo_data_p.shape))
				print_status('Hetero. shape before decoding is ' + str(hetero_data_p.shape))   
 
				# preference direction computation from data 
				for neu_idx in range(homo_data_p.shape[0]): 
					PrefDirInfo['Pref.Homo'][neu_idx]=get_preference(homo_data_p[neu_idx,:,:],homo_labels)[0]			   
					PrefDirInfo['Pref.Hetero'][neu_idx]=get_preference(hetero_data_p[neu_idx,:,:],hetero_labels)[0]
				
				# Parallel decoding begings here
				st = time.time() 
				A=run_parallel_the_pop_decoding(homo_data_p,hetero_data_p,homo_labels,hetero_labels,PrefDirInfo,nt)  # (Homo result, Hetero. result)
				ed = time.time()
					
				elapsed_time = ed - st
				print_status('Execution time: ' + str(elapsed_time) + ' for animal ' + str(p))   
					 
				print('Shape of A is ', A.shape)
				print_status('Saving the tuning curves') 
				path_name=os.path.join(decoding_res_data_path,paradigm,roi,str(pp))
				create_dir(path_name)
				fname=path_name+'/Mouse'+str(p+1) + '.npy'
				print(fname)
							
				np.save(fname,np.squeeze(A))
				del A 
					
				print_status('Done with ' + str(p+1) + '/' + str(noa) + ' for the percentage '+ str(pp),'') 
					 
			else:
				print_status('Already done with ' + str(p+1) + '/' + str(noa) + ' for the percentage '+ str(pp),'') 

# Centering and computing slopes for passive case
 
# Slope dynamics computation and storing the results
for roi in ROIs_hetero:   # For each condition  
	for pp in percent_data: # For each percentage of data 
		
		os.chdir(os.path.join(decoding_res_data_path, paradigm))
		print_status('Computing slopes for  ROI' + roi +' for the percentage  ',str(pp)) 
	
		os.chdir(os.path.join(roi,str(pp)))

		ani_list=os.listdir()
		noa=len(ani_list)

		print_status('No. of animals in ' + roi + ' is ' + str(noa))

		st_roi = time.time()

		slopes=np.zeros((noa,nt,2))  # 2 conditions

		for p in range(len(ani_list)):   # For each animal
		
			save_dir=os.path.join(decoding_res_slopes_path,paradigm,roi,str(pp))
			create_dir(save_dir) 
			fname=os.path.join(save_dir,'slopes.npy')
			
			if not os.path.exists(fname): 

				A=np.load(ani_list[p])  # Load the tuning curve data  
				B=zero_center_the_tcs(A,shift=wrap_around)
				C=B.transpose((0,2,1))
				D=avg_across_zero_centered_tcs(C, shift=wrap_around)

				if Gaussian_smoothening_needed:
					print_status('Gaussian smoothening desired!') 
					S=Gaussian_smoothener(D,sig=sig,trun=trun ,ts=ts) 
				else:
					print_status('NO Gaussian smoothening desired!') 
					S=D  

				## Estimate the slope 
				# Homo-case
				print_status('Computing slopes..') 
				for time_pts in range(nt): 
					slopes[p,time_pts,0]=esti_slope(slope_angles,S[:,time_pts,0],intercept=True, standardise=False) 

				# Hetero-case 
				for time_pts in range(nt): 
					slopes[p,time_pts,1]=esti_slope(slope_angles,S[:,time_pts,1],intercept=True, standardise=False)   
				  
			else:
				print_status('Already done with slope computations')  
		np.save(fname,slopes)  
		
		 
 
# Plotting and montaging  (with task as solid  line and passive as dashed line) 
#plots_data_dir=decoding_res_fig_path 
#os.chdir(os.path.join(decoding_res_slopes_path,paradigm)) 
#flist=os.listdir()

for roi in ROIs_hetero:  # for each roi
	for pp in percent_data: # for each p-value percentage   
		
		paradigm='task'
		# For task (solid lines) 
		A_task=np.load(os.path.join(decoding_res_slopes_path,paradigm,roi,str(pp),'slopes.npy'))
		
		sig1=A_task[:,:,0]  # homo  # no. mouse X no. time points X (homo or hetero)
		sig2=A_task[:,:,1]  # hetero   
		
		fig, ax = plt.subplots(1,1,figsize=(7,7))	
		# plot the mean and error bar
		ax.plot(tt,np.mean(sig1,0),'r-',tt,np.mean(sig2,0),'b-',linewidth=plt_lwd) 
		ax.fill_between(tt,np.mean(sig1,0)- (np.std(sig1,0)/np.sqrt(sig1.shape[0])),  
						   np.mean(sig1,0)+ (np.std(sig1,0)/np.sqrt(sig1.shape[0])),alpha=alp,color='r')
		ax.fill_between(tt,np.mean(sig2,0)- (np.std(sig2,0)/np.sqrt(sig2.shape[0])),  
						   np.mean(sig2,0)+ (np.std(sig2,0)/np.sqrt(sig2.shape[0])),alpha=alp,color='b') 

		
		# permutation clustering for homo
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig1,
																			   tail=1,
																			   n_permutations=nperm)
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1  

		if len(clus):
			
			clus=np.concatenate(clus)
			clus=clus[clus>20]
			
			sig_tt=tt[clus]  # Significant time points
			ax.plot(sig_tt,np.repeat(first_sig_task,len(sig_tt)),'r-', linewidth=lwd) 
			slope_sig1=np.mean(A_task[:,clus,0],1)	
		else:
			print_status('No Significant clusters in ' + roi  + '  ' + str(pp) +' (homo) case')
			slope_sig1=np.zeros(sig1.shape[0])
		
		#print('Slope sig 1', slope_sig1)
		# permutation clustering for hetero
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig2,
																			   tail=1,
																		   n_permutations=nperm)
		
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1 
				
		if len(clus):
			
			clus=clus=np.concatenate(clus)
			clus=clus[clus>20]
			
			sig_tt=tt[clus]  # Significant time points
			ax.plot(sig_tt,np.repeat(second_sig_task,len(sig_tt)),'b-', linewidth=lwd) 
		
			slope_sig2=np.mean(A_task[:,clus,1],1)
		else:
			print_status('No Significant clusters in ' + roi + '  ' + str(pp) +' (hetero) case')
			slope_sig2=np.zeros(sig2.shape[0])
		

		res=np.column_stack((slope_sig1,slope_sig2))
		np.savetxt(os.path.join('/media/olive/Research/oliver/pop_slopes/',paradigm,roi+'_'+str(pp)+'.csv'),res,delimiter=',')

		# permuation clustering for homo-hetero
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig1-sig2,
																			   tail=0,
																			   n_permutations=nperm)
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1 
				
		if len(clus):
			sig_tt=tt[np.concatenate(clus)]  # Significant time points
			ax.plot(sig_tt,np.repeat(diff_sig_task,len(sig_tt)),'k-', linewidth=lwd) 
			
			
		## plotting for passive 
		paradigm='passive'
		A_passive=np.load(os.path.join(decoding_res_slopes_path,paradigm,roi,str(pp),'slopes.npy'))
		
		sig1=A_passive[:,:,0]  # homo  # no. mouse X no. time points X (homo or hetero)
		sig2=A_passive[:,:,1]  # hetero    
		 
		# plot the mean and error bar
		ax.plot(tt,np.mean(sig1,0),'r--',tt,np.mean(sig2,0),'b--',linewidth=plt_lwd) 
		ax.fill_between(tt,np.mean(sig1,0)- (np.std(sig1,0)/np.sqrt(sig1.shape[0])),  
						   np.mean(sig1,0)+ (np.std(sig1,0)/np.sqrt(sig1.shape[0])),alpha=alp,color='r')
		ax.fill_between(tt,np.mean(sig2,0)- (np.std(sig2,0)/np.sqrt(sig2.shape[0])),  
						   np.mean(sig2,0)+ (np.std(sig2,0)/np.sqrt(sig2.shape[0])),alpha=alp,color='b') 

		
		# permutation clustering for homo
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig1,
																			   tail=1,
																			   n_permutations=nperm)
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1  

		if len(clus):
			
			clus=np.concatenate(clus)
			clus=clus[clus>20]
			
			sig_tt=tt[clus]  # Significant time points
			ax.plot(sig_tt,np.repeat(first_sig_passive,len(sig_tt)),'r--', linewidth=lwd) 
			slope_sig1=np.mean(A_passive[:,clus,0],1)	
		else:
			print_status('No Significant clusters in ' + roi + '  ' + str(pp) +' (homo) case')
			slope_sig1=np.zeros(sig1.shape[0])
		
		#print('Slope sig 1', slope_sig1)
		# permutation clustering for hetero
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig2,
																			   tail=1,
																		   n_permutations=nperm)
		
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1 
				
		if len(clus):
			
			clus=clus=np.concatenate(clus)
			clus=clus[clus>20]
			
			sig_tt=tt[clus]  # Significant time points
			ax.plot(sig_tt,np.repeat(second_sig_passive,len(sig_tt)),'b--', linewidth=lwd) 
		
			slope_sig2=np.mean(A_passive[:,clus,1],1)
		else:
			print_status('No Significant clusters in ' + roi + '  ' + str(pp) +' (hetero) case')
			slope_sig2=np.zeros(sig2.shape[0])
		

		res=np.column_stack((slope_sig1,slope_sig2))
		np.savetxt(os.path.join('/media/olive/Research/oliver/pop_slopes/',paradigm,roi+'_'+str(pp)+'.csv'),res,delimiter=',')

		# permuation clustering for homo-hetero
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig1-sig2,
																			   tail=0,
																			   n_permutations=nperm)
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1 
				
		if len(clus):
			sig_tt=tt[np.concatenate(clus)]  # Significant time points
			ax.plot(sig_tt,np.repeat(diff_sig_passive,len(sig_tt)),'k--', linewidth=lwd) 
		
		
		ax.set_xticks([0, 1,2,3,4,5]) 
		ax.set_yticks(np.arange(-0.2,ymax+0.01,0.1)) 
		ax.set_xlim(xmin, xmax) 
		ax.axvline(x=0,color='k',linestyle='--',lw=1)  
		ax.axhline(y=0,color='k',linestyle='--',lw=1)  
		ax.set_ylim(round(ymin,2), round(ymax,2))  
		ax.spines[['top','right']].set_visible(False) 
		ax.spines[['bottom','left']].set_linewidth(3)
 
		#ax.text(2.5,0.01, '(N=8)',fontsize=32) 
		#ax.text(-1.3,-0.077, '$-0.5$',fontsize=24) 
		ax.tick_params(axis='both', which='major', labelsize=24) 
		
		fig.tight_layout(pad=2)   
		#plt.show() 
		save_file_name='Combined_' + roi + '_'+str(pp)+'.png'
		fig.savefig(os.path.join(decoding_res_fig_path,save_file_name),dpi=300) 
		os.chdir('..') 
		
# montaging (this will work only if your system is Linux and montage installed))
if is_montage_installed():
	os.chdir(decoding_res_fig_path)
	create_dir('montages') 
    
	fname='montages/V1_45.png' 
	status=os.system('montage Combined_V1_45_0.png Combined_V1_45_10.png Combined_V1_45_20.png Combined_V1_45_40.png Combined_V1_45_60.png  Combined_V1_45_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/V1_90.png' 
	status=os.system('montage Combined_V1_90_0.png Combined_V1_90_10.png Combined_V1_90_20.png Combined_V1_90_40.png Combined_V1_90_60.png  Combined_V1_90_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/V1_135.png' 
	status=os.system('montage Combined_V1_135_0.png Combined_V1_135_10.png Combined_V1_135_20.png Combined_V1_135_40.png Combined_V1_135_60.png  Combined_V1_135_100.png   -tile 6x1  -geometry +1+1 ' + fname) 
			
	fname='montages/PPC_45.png' 
	status=os.system('montage Combined_PPC_45_0.png Combined_PPC_45_10.png Combined_PPC_45_20.png Combined_PPC_45_40.png Combined_PPC_45_60.png  Combined_PPC_45_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/PPC_90.png' 
	status=os.system('montage Combined_PPC_90_0.png Combined_PPC_90_10.png Combined_PPC_90_20.png Combined_PPC_90_40.png Combined_PPC_90_60.png  Combined_PPC_90_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/PPC_135.png' 
	status=os.system('montage Combined_PPC_135_0.png Combined_PPC_135_10.png Combined_PPC_135_20.png Combined_PPC_135_40.png Combined_PPC_135_60.png  Combined_PPC_135_100.png   -tile 6x1  -geometry +1+1 ' + fname)
else:
	print_status('Montage NOT installed in your computer. Skipping...') 
 
	
""" 
# Plotting and montaging for one condition (as in the First submission)
## Plotting the results (task as solid  line and passive as dashed line)

plots_data_dir=decoding_res_fig_path 
os.chdir(os.path.join(decoding_res_slopes_path,paradigm)) 
flist=os.listdir()

for folder in ROIs_hetero:  # for each folder
#for folder in ROIs_hetero[1:2]:  # for checking
	os.chdir(os.path.join(decoding_res_slopes_path,paradigm))
	os.chdir(folder)
	
	for pp in percent_data: # for each p-value percentage 
		
	
		os.chdir(str(pp))
		
		A=np.load('slopes.npy')
		
		sig1=A[:,:,0]  # homo  # no. mouse X no. time points X (homo or hetero)
		sig2=A[:,:,1]  # hetero   
		
		fig, ax = plt.subplots(1,1,figsize=(7,7))	
		# plot the mean and error bar
		ax.plot(tt,np.mean(sig1,0),'r-',tt,np.mean(sig2,0),'b-',linewidth=plt_lwd) 
		ax.fill_between(tt,np.mean(sig1,0)- (np.std(sig1,0)/np.sqrt(sig1.shape[0])),  
						   np.mean(sig1,0)+ (np.std(sig1,0)/np.sqrt(sig1.shape[0])),alpha=alp,color='r')
		ax.fill_between(tt,np.mean(sig2,0)- (np.std(sig2,0)/np.sqrt(sig2.shape[0])),  
						   np.mean(sig2,0)+ (np.std(sig2,0)/np.sqrt(sig2.shape[0])),alpha=alp,color='b') 

		
		# permutation clustering for homo
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig1,
																			   tail=1,
																			   n_permutations=nperm)
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1  

		if len(clus):
			
			clus=np.concatenate(clus)
			clus=clus[clus>20]
			
			sig_tt=tt[clus]  # Significant time points
			ax.plot(sig_tt,np.repeat(first_sig,len(sig_tt)),'r-', linewidth=lwd) 
			slope_sig1=np.mean(A[:,clus,0],1)	
		else:
			print_status('No Significant clusters in ' + folder + '  ' + str(pp) +' (homo) case')
			slope_sig1=np.zeros(sig1.shape[0])
		
		#print('Slope sig 1', slope_sig1)
		# permutation clustering for hetero
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig2,
																			   tail=1,
																		   n_permutations=nperm)
		
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1 
				
		if len(clus):
			
			clus=clus=np.concatenate(clus)
			clus=clus[clus>20]
			
			sig_tt=tt[clus]  # Significant time points
			ax.plot(sig_tt,np.repeat(second_sig,len(sig_tt)),'b-', linewidth=lwd) 
		
			slope_sig2=np.mean(A[:,clus,1],1)
		else:
			print_status('No Significant clusters in ' + folder + '  ' + str(pp) +' (hetero) case')
			slope_sig2=np.zeros(sig2.shape[0])
		

		res=np.column_stack((slope_sig1,slope_sig2))
		np.savetxt(os.path.join('/media/olive/Research/oliver/pop_slopes/',paradigm,folder+'_'+str(pp)+'.csv'),res,delimiter=',')

		# permuation clustering for homo-hetero
		T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(sig1-sig2,
																			   tail=0,
																			   n_permutations=nperm)
		clus=[]
		p=0
		for k in cluster_p_values:
			if k<(cluster_alpha):
				clus.extend(clusters[p])
				p=p+1
			else:
				p=p+1 
		if len(clus):
			sig_tt=tt[np.concatenate(clus)]  # Significant time points
			ax.plot(sig_tt,np.repeat(diff_sig,len(sig_tt)),'k-', linewidth=lwd) 
		
		
		ax.set_xticks([0, 1,2,3,4,5]) 
		ax.set_yticks(np.arange(-0.2,ymax+0.01,0.1)) 
		ax.set_xlim(xmin, xmax) 
		ax.axvline(x=0,color='k',linestyle='--',lw=1)  
		ax.axhline(y=0,color='k',linestyle='--',lw=1)  
		ax.set_ylim(round(ymin,2), round(ymax,2))  
		ax.spines[['top','right']].set_visible(False) 
		ax.spines[['bottom','left']].set_linewidth(3)
 
		#ax.text(2.5,0.01, '(N=8)',fontsize=32) 
		#ax.text(-1.3,-0.077, '$-0.5$',fontsize=24) 
		ax.tick_params(axis='both', which='major', labelsize=24) 
		
		fig.tight_layout(pad=2)   
		#plt.show() 
		save_file_name=paradigm + '_' + folder + '_'+str(pp)+'.png'
		fig.savefig(os.path.join(decoding_res_fig_path,save_file_name),dpi=300) 
		os.chdir('..') 
		
# montaging (this will work only if your system is Linux and montage installed))
if is_montage_installed():
	os.chdir(decoding_res_fig_path)
	create_dir('montages')

	fname='montages/Task_V1_45.png' 
	status=os.system('montage task_V1_45_0.png task_V1_45_10.png task_V1_45_20.png task_V1_45_40.png task_V1_45_60.png  task_V1_45_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_V1_90.png' 
	status=os.system('montage task_V1_90_0.png task_V1_90_10.png task_V1_90_20.png task_V1_90_40.png task_V1_90_60.png  task_V1_90_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_V1_135.png' 
	status=os.system('montage task_V1_135_0.png task_V1_135_10.png task_V1_135_20.png task_V1_135_40.png task_V1_135_60.png  task_V1_135_100.png   -tile 6x1  -geometry +1+1 ' + fname) 
			
	fname='montages/Task_PPC_45.png' 
	status=os.system('montage task_PPC_45_0.png task_PPC_45_10.png task_PPC_45_20.png task_PPC_45_40.png task_PPC_45_60.png  task_PPC_45_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_PPC_90.png' 
	status=os.system('montage task_PPC_90_0.png task_PPC_90_10.png task_PPC_90_20.png task_PPC_90_40.png task_PPC_90_60.png  task_PPC_90_100.png   -tile 6x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_PPC_135.png' 
	status=os.system('montage task_PPC_135_0.png task_PPC_135_10.png task_PPC_135_20.png task_PPC_135_40.png task_PPC_135_60.png  task_PPC_135_100.png   -tile 6x1  -geometry +1+1 ' + fname)
else:
	print_status('Montage NOT installed in your computer. Skipping...') 

"""	 

# 1. once you receive the p-value; put them in the folder and tun the R file PrefDirection.R to get the csv files iused for decoding;
# Make sure to names are in the format as in task

# 2. Run the decoding with task off (only passive decoding)

# 3. Check the plots

# 4. Check the sslope summary plots in SlopesSummary.R