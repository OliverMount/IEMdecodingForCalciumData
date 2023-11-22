# Revision analysis with sorting the p=values

import sys
import os
 
import numpy as np 
import multiprocessing as mp  # For parallel processing
 
import re    
import time    

from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy.io import loadmat,savemat   

import matplotlib.pyplot as plt  

# Local modules
sys.path.append('/media/olive/Research/oliver/utils/Python/')  
from file_utils import *

from mne.stats import permutation_cluster_test,permutation_cluster_1samp_test
import mne 

from sklearn import *
from sklearn.model_selection import LeaveOneOut 
from sklearn.utils.validation import check_is_fitted

# decoder
from Decoder import InvertedEncoding
 


def print_status(msg,kind='short',line=50): 
    if kind=='short':
        print('++ '+msg)
    else:
        print('++ '+line*'-')
        print('++ '+msg)
        print('++ '+line*'-')
        
def roll_all(X,y,ref): 
	
	s=X.shape
	Ls=X.ndim  # dimension of the data X   
	
	if Ls==1:	#single-trial test set (1D) # This is typical of CV=1 in diagonal decoding 
		res=np.roll(X,ref-y)
	elif Ls==2:  # Muliple trials test set (2D); This is typical of CV=1 in CTG decoding or CV>1 in diagonal decoding
		if s[1]!=len(y):	  # Make sure the no. of cols of X and size of y are the same
			raise ValueError('-- No. of test samples and no. of ground truth labels do not match')
		else:
			res=np.zeros_like(X)
			for k in range(s[1]):  # s[1] can be trial or time axis; if time then all element of y will the same
				#res[:,k]=roll_tuning_curve(X[:,k],y[k],ref)
				res[:,k]=np.roll(X[:,k],ref-y[k])
			 
	elif Ls==3: # Multiple trials test set along time (3D); This is typical of CV>1; in CTG decoding
		if s[1]!=len(y):	  # Trial dimension of X (2nd dim) and trial dimention of y (1st dim) should be the same
			raise ValueError('-- No. of test samples and no. of ground truth labels do not match')
		else:
			res=np.zeros_like(X)
			for t in range(s[2]): # Along time
				for k in range(s[1]): # Along trials
					#res[:,k,t]=roll_tuning_curve(X[:,k,t],y[k,t],ref)
					#print("ref-" ,ref-y[k]) 
					res[:,k,t]=np.roll(X[:,k,t],ref-y[k]) # irrepective of time t and same label 
	else:  # not implemented for more than three dimensions
		raise NotImplementedError('-- roll_all is not implemented for more than 3D :-( ')
		
	return res 
        
def run_parallel_the_time(D1,D2,d1,d2,nt):
 
	n_cpu = mp.cpu_count()  # Total number of CPU
	pool = mp.Pool(n_cpu) 
	
	time_step_results = [pool.apply_async(process_time_step,args=(D1[:,:,tr],D2[:,:,tr],d1,d2)) for tr in range(nt)]   
	pool.close()
	pool.join()

	results = [r.get() for r in time_step_results]   
 
	return np.stack(results,axis=-1).transpose(0,2,1)   # no X time X (homo,hetero)


# In[22]:


def process_time_step(Xtra,Xte,ytra,yte):

	# LOO for homogeneous 
	cv=LeaveOneOut()
	cv_results = []
	
	for train, test in cv.split(Xtra.T):  # cross-validation split over trials
		cv_results.append(process_cv(Xtra[:, train], ytra[train], Xtra[:, test], ytra[test]))  
 
	final_result = np.stack(cv_results, axis=-1)   
	cv_res_final=np.mean(final_result,2)   # mean across trials of zero-centered) tuning curves (homo case)
 
	# Homo train- Hetero test 
	model=InvertedEncoding(angles, p=6, ref=center_around) 
	model.fit(Xtra,ytra,sys_type="under") 
 
	model.predict(Xte)   
	predicted=model.predicted_values

	# centering the tuning curve based on the presented orientation
	ypred=roll_all(predicted,yte,center_around)
	ypred=np.mean(ypred,1) # mean across trials of zero-centered) tuning curvesv (hetero case)   
    
	#print(cv_res_final.shape)
	#print(ypred[:,np.newaxis].shape) 
	res=np.concatenate((cv_res_final,ypred[:,np.newaxis]),axis=-1)  
	return res 
 

def process_cv(Xtrain,ytrain,Xtest,ytest): 
 
	model=InvertedEncoding(angles,p=6, ref=center_around)   
	model.fit(Xtrain, ytrain,sys_type="under")  
    
	model.predict(Xtest)    
	predicted=model.predicted_values 
 
	y_pred=roll_all(predicted,ytest,center_around) 
 
	return y_pred  


# ### Data path and parameter settings 

data_path =  '/media/olive/Research/oliver/data_down/' 

paradigm = 'task'   
data_path_task =  os.path.join(data_path,paradigm)  

paradigm = 'passive' 
data_path_passive =   os.path.join(data_path,paradigm)

decoding_res_path = '/media/olive/Research/oliver/decoding_revision/'
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

# Adjustable parameter settings for decoding
scale=1  # scaling factor for Inverted encoding model

fs=20     # sampling frequency 
ts=1/fs   # sampling time

#Data parameter settings
analyse_time=[-1,5] # sec 
time_values=np.arange(analyse_time[0],analyse_time[1],ts)  
nt=len(time_values)

angles=np.arange(22.5,360,45) # Stimulus angles
ns=len(angles)   # no. of stimulus values
center_around=5  # Center the tuning curves around this angle 
cv_folds=1       # LOO

ROIs_hetero=['V1_45','V1_90','V1_135','PPC_45','PPC_90','PPC_135']

for k in ROIs_hetero:
    create_dir(os.path.join(decoding_res_data_path,'task',k))
    create_dir(os.path.join(decoding_res_data_path,'passive',k)) 

#p-values
pval_path='/media/olive/Research/oliver/data_down/pvals'
 

percent_data=[10,20,40,60,100]   
paradigm='task'
V1_task=['1R.mat','2L.mat','newV1.mat','Ylbd.mat','Ylbe.mat','Ylbf.mat','Ylbg.mat','2.mat',	'3.mat','4.mat']  # mouse order
PPC_task=['3L.mat',	'3R.mat','YLbc.mat','YLcj.mat',	'YLck.mat','YLcl.mat','YLcm.mat']

 
#V1_passive=['1L','1R','2L','newV1	Ylbd	Ylbe	Ylbf	Ylbg	2	3	4	Ylcd	Ylce	Ylcf	Ylcg

# ### Dealing with task conditions  
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
        
        A=loadmat(ani_list[p])     # Load the data
        
        # homo and hetero data
        homo_data=A['sample_data_homo']       #orig.data:     trials X units X time-pts
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
        hetero_data=hetero_data[homo_indices,:,:]  # arranging data accroding to sorted pvalue
        
        
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