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
center_around=5  # Center the tuning curves around this angle 
  
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
ymax=0.25 
first_sig=0.24
second_sig=0.23
diff_sig=0.22

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
	B = pd.read_csv(os.path.join(pval_pref_path, paradigm,roi+'_prefer.csv'))
	 
	#C=loadmat(os.path.join(pval_path,paradigm+'_'+roi+'.mat'))
	#Pval_homo = [element[0][0][0] for i in range(C['pVal_homo'].shape[0]) if np.size(C['pVal_homo'][i][0]) != 0 for element in C['pVal_homo'][i]]
	#Pval_hetero= [element[0][0][0] for i in range(C['pVal_hetero'].shape[0]) if np.size(C['pVal_hetero'][i][0]) != 0 for element in C['pVal_hetero'][i]]
	
	if roi.startswith('V1'):
		list_for_sorting=V1_task
	else:
		list_for_sorting=PPC_task 
	
	ani_list=[[file for file in ani_list if file.lower().endswith(suffix.lower())] for suffix in list_for_sorting]
	ani_list= [ani[0] for ani in ani_list]
	
	st_roi = time.time()
	for p in range(len(ani_list)):   # For each animal
	#for p in range(1):	
		# load  the pure tuning for homo
		df_homo=B[(B['Sub']=='Animal.'+str(p+1))  & (B['Group'] =='homo')]
		df_hetero=B[(B['Sub']=='Animal.'+str(p+1))  & (B['Group'] =='hetero')] 
		
		# resetting the index is needed after subsetting the data
		df_homo.reset_index(drop=True, inplace=True)
		df_hetero.reset_index(drop=True, inplace=True)  # reset the indices
		
		# to get a data frame that is tuned in both the condition
		# finding indices that match both conditions
		final=pd.merge(df_homo[df_homo['Pvalue'] <= 0.05], df_hetero[df_hetero['Pvalue'] <= 0.05], left_index=True, right_index=True)
		#print(final.shape)
		TunedIndices=final.index.to_numpy() 
		
		pref_df = final[['Preference_x','Preference_y']]
		pref_df = pref_df.copy()  # to avoid the copy warnings
		pref_df['Neuron'] = final.index.to_list()
		pref_df.rename(columns={'Preference_x': 'Pref.Homo', 'Preference_y': 'Pref.Hetero'}, inplace=True)
		pref_df.reset_index(drop=True, inplace=True)
		
		#df_homo_final=df_homo.iloc[indices]
		#df_hetero_final=df_hetero.iloc[indices] 
		
		#df_homo_final.reset_index(drop=True, inplace=True)
		#df_hetero_final.reset_index(drop=True, inplace=True)  
		
		# This is for data sorting according to P-value
		idx_homo=np.argsort(df_homo['Pvalue'])  # homo in the ascending order
		
		df_homo_sorted=df_homo.loc[idx_homo]
		df_hetero_sorted=df_hetero.loc[idx_homo]
		
		
		
		#Ltuned=len(np.where(df_homo['Pvalue']<=pval_thresold)[0])
		#df_homo_final.shape[0]
		
		#idx_hetero=np.argsort(df_hetero['Pvalue'])  # hetero in the ascending order
		#Ltuned=len(np.where(df_hetero['Pvalue']<=pval_thresold)[0])
		
		#df_homo_final=df_homo.iloc[idx_homo[:Ltuned]]
		#df_heter_final=df_hetero.iloc[idx_homo[:Ltuned]]
		
		#df=df_homo[df_homo['Pvalue']<=pval_thresold]
		#df_homo=B[ (B['Sub']=='Animal.'+str(p+1)) & (B['Group'] =='homo') & (B['Pvalue']<= pval_thresold)]
		#df_hetero=B[ (B['Sub']=='Animal.'+str(p+1)) & (B['Group'] =='homo') & (B['Pvalue']<= pval_thresold) & ()]
		
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
		idx=np.random.permutation(len(homo_labels))
		homo_labels=homo_labels[idx]
		homo_data=homo_data[:,idx,:]
		
		idx=np.random.permutation(len(hetero_labels))
		hetero_labels=hetero_labels[idx]
		hetero_data=hetero_data[:,idx,:]
		
		print_status('Before subsetting the data')
		print_status('Homo. shape is ' + str(homo_data.shape))
		print_status('Hetero. shape is ' + str(hetero_data.shape))   
		
		# arranging according to the indices (p-value <= 0.05) (units X trials X time-pts )
		#homo_indices=np.argsort(Pval_homo[p])
		#homo_data=homo_data[NeuronIndices,:,:]  # arranging homo data accroding to sorted pvalue 
		#hetero_data=hetero_data[NeuronIndices,:,:]  # arranging hetero data accroding to sorted pvalue 
		
		#print_status('After subsetting the data')
		#print_status('Homo. shape is ' + str(homo_data.shape))
		#print_status('Hetero. shape is ' + str(hetero_data.shape))   
		#print_status('After subsetting data is used for decoding')
		
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
					 
					#hetero_data_p=hetero_data[:p_hetero,:,:] 
					# first remove the tuned tuned one from the original df
					df_homo_sorted_pp=df_homo_sorted.drop(TunedIndices) 
					df_hetero_sorted_pp=df_hetero_sorted.drop(TunedIndices)
					
					nper_cells = int(np.ceil((pp/100)*df_homo_sorted_pp.shape[0])) # get the number of cells 
					
					additional_neurons=df_homo_sorted_pp[:nper_cells].index.to_numpy()
					Ladd=len(additional_neurons)
					
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
				
				# Parallel decoding begings here
				st = time.time() 
				A=run_parallel_the_pop_decoding(homo_data_p,hetero_data_p,PrefDirInfo,nt)  # (Homo result, Hetero. result)
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
					
				ed_roi = time.time()
				elapsed_time = ed_roi - st_roi
				print_status('Execution time for the whole roi is ' + str(elapsed_time))   
		else:
			print_status('Already done with ' + str(p+1) + '/' + str(noa) + ' for the percentage '+ str(pp),'') 
