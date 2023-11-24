# Revision analysis with sorting the p-values according to the homo data

import sys
import os
 
import numpy as np 
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
pval_path='/media/olive/Research/oliver/data_down/pvals'


paradigm = 'task'   
data_path_task =  os.path.join(data_path,paradigm)  

paradigm = 'passive' 
data_path_passive =   os.path.join(data_path,paradigm)

decoding_res_path = '../decoding/'
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
cv_folds=1	   # LOO


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

ymin=-0.3
ymax=0.7


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
 
# Percentage of cells to use for decoding (it includes both tuned and untuned neurons)
percent_data=[10,20,40,60,100]	
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
	
	# load the pvalues for an animals 
	B=loadmat(os.path.join(pval_path,paradigm+'_'+roi+'.mat'))
	Pval_homo = [element[0][0][0] for i in range(B['pVal_homo'].shape[0]) if np.size(B['pVal_homo'][i][0]) != 0 for element in B['pVal_homo'][i]]
	Pval_hetero= [element[0][0][0] for i in range(B['pVal_hetero'].shape[0]) if np.size(B['pVal_hetero'][i][0]) != 0 for element in B['pVal_hetero'][i]]
	
	if roi.startswith('V1'):
		list_for_sorting=V1_task
	else:
		list_for_sorting=PPC_task
		
	
	ani_list=[[file for file in ani_list if file.lower().endswith(suffix.lower())] for suffix in list_for_sorting]
	ani_list= [ani[0] for ani in ani_list]
	
	st_roi = time.time()
	for p in range(len(ani_list)):   # For each animal
	#for p in range(1):	
		
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
		
		print_status('Homo. shape is ' + str(homo_data.shape))
		print_status('Hetero. shape is ' + str(hetero_data.shape))   
		
		# arranging according to the p-value (units X trials X time-pts )
		homo_indices=np.argsort(Pval_homo[p])
		homo_data=homo_data[homo_indices,:,:]  # arranging homo data accroding to sorted pvalue 
		hetero_data=hetero_data[homo_indices,:,:]  # arranging hetero data accroding to sorted pvalue
		
		for pp in percent_data: 
		
			path_name=os.path.join(decoding_res_data_path,paradigm,roi,str(pp))
			create_dir(path_name)
			fname=path_name+'/Mouse'+str(p+1) + '.npy'
			
			if not os.path.exists(fname): 
			
				p_homo = int(np.ceil((pp/100)*homo_data.shape[0])) 
				homo_data_p=homo_data[:p_homo,:,:]

				p_hetero = int(np.ceil((pp/100)*hetero_data.shape[0])) 
				hetero_data_p=hetero_data[:p_hetero,:,:]   

				print_status('Homo. shape before decoding is ' + str(homo_data_p.shape))
				print_status('Hetero. shape before decoding is ' + str(hetero_data_p.shape))   
			
				# Parallel decoding begings here
				st = time.time() 
				A=run_parallel_the_time(homo_data_p,hetero_data_p,homo_labels,hetero_labels,nt)  # (Homo result, Hetero. result)
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

		slopes=np.zeros((noa,nt,2))

		for p in range(len(ani_list)):   # For each animal
		
			save_dir=os.path.join(decoding_res_slopes_path,'task',roi,str(pp))
			create_dir(save_dir) 
			fname=os.path.join(save_dir,'slopes.npy')
			
			if not os.path.exists(fname): 

				A=np.load(ani_list[p])  # Load the tuning curve data 
				
				B=avg_across_zero_centered_tcs(A,shift=wrap_around)

				if Gaussian_smoothening_needed:
					print_status('Gaussian smoothening desired!') 
					S=Gaussian_smoothener(B,sig=sig,trun=trun ,ts=ts) 
				else:
					print_status('NO Gaussian smoothening desired!') 
					S=B  

				## Estimate the slope 
				# Homo-case
				print_status('Computing slopes..') 
				for time_pts in range(nt): 
					slopes[p,time_pts,0]=esti_slope(slope_angles,S[:,time_pts,0],intercept=True, standardise=False) 

				# Hetero-case 
				for time_pts in range(nt): 
					slopes[p,time_pts,1]=esti_slope(slope_angles,S[:,time_pts,1],intercept=True, standardise=False)   
				 
				np.save(fname,slopes)
			else:
				print_status('Already done with slope computations')


# Plotting, cluster computation  and storing
 
plots_data_dir=decoding_res_fig_path

os.chdir(os.path.join(decoding_res_slopes_path,paradigm))

flist=os.listdir()

for folder in ROIs_hetero:  # for each folder
#for folder in ROIs_hetero[1:2]:  # for checking
	os.chdir(os.path.join(decoding_res_slopes_path,paradigm))
	os.chdir(folder)
	
	for pp in percent_data: # for each p-value percentage
	#for pp in percent_data[4:5]: # for checking
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
			sig_tt=tt[np.concatenate(clus)]  # Significant time points
			ax.plot(sig_tt,np.repeat(first_sig,len(sig_tt)),'r-', linewidth=lwd)
	
		slope_sig1=np.mean(A[:,np.concatenate(clus),0],1)	
		
		print('Slope sig 1', slope_sig1)
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
			sig_tt=tt[np.concatenate(clus)]  # Significant time points
			ax.plot(sig_tt,np.repeat(second_sig,len(sig_tt)),'b-', linewidth=lwd) 
		
		slope_sig2=np.mean(A[:,np.concatenate(clus),1],1)
		

		res=np.column_stack((slope_sig1,slope_sig2))
		np.savetxt(os.path.join('/home/olive/Desktop/res/',paradigm,folder+'_'+str(pp)+'.csv'),res,delimiter=',')

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
		ax.set_yticks([0, 0.1,0.2]) 
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
	status=os.system('montage task_V1_45_10.png task_V1_45_20.png task_V1_45_40.png task_V1_45_60.png  task_V1_45_100.png   -tile 5x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_V1_90.png' 
	status=os.system('montage task_V1_90_10.png task_V1_90_20.png task_V1_90_40.png task_V1_90_60.png  task_V1_90_100.png   -tile 5x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_V1_135.png' 
	status=os.system('montage task_V1_135_10.png task_V1_135_20.png task_V1_135_40.png task_V1_135_60.png  task_V1_135_100.png   -tile 5x1  -geometry +1+1 ' + fname) 
			
	fname='montages/Task_PPC_45.png' 
	status=os.system('montage task_PPC_45_10.png task_PPC_45_20.png task_PPC_45_40.png task_PPC_45_60.png  task_PPC_45_100.png   -tile 5x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_PPC_90.png' 
	status=os.system('montage task_PPC_90_10.png task_PPC_90_20.png task_PPC_90_40.png task_PPC_90_60.png  task_PPC_90_100.png   -tile 5x1  -geometry +1+1 ' + fname) 

	fname='montages/Task_PPC_135.png' 
	status=os.system('montage task_PPC_135_10.png task_PPC_135_20.png task_PPC_135_40.png task_PPC_135_60.png  task_PPC_135_100.png   -tile 5x1  -geometry +1+1 ' + fname)
else:
	print_status('Montage NOT installed in your computer. Skipping...')

#########################################################################################
############################  ANALYSIS FOR PASSIVE DATA #################################
#########################################################################################

""" 
paradigm='passive'

PPC_passive=['3L.mat','3R.mat','YLbc.mat','YLcj.mat','YLck.mat','YLcl.mat','YLcm.mat']
V1_passive=['1L.mat','1R.mat','2L.mat','Ylce.mat','Ylcf.mat','Ylcg.mat']

# Decoding for passivve  conditions  
for roi in ROIs_hetero:   # For each heterogeneous condition
	os.chdir(data_path_passive)
	print_status('Dealing with ROI: ' + roi)
	
	os.chdir(roi)
	
	ani_list=os.listdir()
	print(ani_list)
	noa=len(ani_list)
	
	print_status('No. of animals in ' + roi + ' is ' + str(noa)) 
	
	# load the pvalues for all animals 
	B=loadmat(os.path.join(pval_path,paradigm+'_'+roi+'.mat'))
	Pval_homo = [element[0][0][0] for i in range(B['pVal_homo'].shape[0]) if np.size(B['pVal_homo'][i][0]) != 0 for element in B['pVal_homo'][i]]
	Pval_hetero= [element[0][0][0] for i in range(B['pVal_hetero'].shape[0]) if np.size(B['pVal_hetero'][i][0]) != 0 for element in B['pVal_hetero'][i]]
	
	if roi.startswith('V1'):
		list_for_sorting=V1_passive
	else:
		list_for_sorting=PPC_passive
		
	ani_list=[[file for file in ani_list if file.lower().endswith(suffix.lower())] for suffix in list_for_sorting]
	ani_list= [ani[0] for ani in ani_list]
	
	st_roi = time.time()
	for p in range(len(ani_list)):   # For each animal
		
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
		
		print_status('Homo. shape is ' + str(homo_data.shape))
		print_status('Hetero. shape is ' + str(hetero_data.shape))   
		
		# arranging according to the p-value (units X trials X time-pts )
		homo_indices=np.argsort(Pval_homo[p])
		homo_data=homo_data[homo_indices,:,:]  # arranging data accroding to sorted pvalue 
	  
		#hetero_indices=np.argsort(Pval_hetero[p])
		hetero_data=hetero_data[homo_indices,:,:]  # arranging hetero data accroding to sorted pvalue of homo
		
		for pp in percent_data: 
			p_homo = int(np.ceil((pp/100)*homo_data.shape[0])) 
			homo_data_p=homo_data[:p_homo,:,:]

			p_hetero = int(np.ceil((pp/100)*hetero_data.shape[0])) 
			hetero_data_p=hetero_data[:p_hetero,:,:]   

			print_status('Homo. shape before decoding is ' + str(homo_data_p.shape))
			print_status('Hetero. shape before decoding is ' + str(hetero_data_p.shape))   
		
			# Parallel decoding begings here
			st = time.time() 
			A=run_parallel_the_time(homo_data_p,hetero_data_p,homo_labels,hetero_labels,nt)  # (Homo result, Hetero. result)
			ed = time.time()
			
			elapsed_time = ed - st
			print_status('Execution time: ' + str(elapsed_time) + ' for animal ' + str(p))   
			 
			print('Shape of A is ', A.shape)
			print_status('Saving the tuning curves') 
			path_name=os.path.join(task_save_path,roi,str(pp))
			create_dir(path_name)
			fname=path_name+'/Mouse'+str(p+1) + '.npy'
			print(fname)
					
			np.save(fname,np.squeeze(A))
			del A 
			
			print_status('Done with ' + str(p+1) + '/' + str(noa) + ' for the percentage '+ str(pp),'') 
		
			print_status('Done with ' + str(p+1) + '/' + str(noa),'') 
			ed_roi = time.time()
			elapsed_time = ed_roi - st_roi
			print_status('Execution time for the whole roi is ' + str(elapsed_time))   
"""
