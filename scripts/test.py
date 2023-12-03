tr=80

ho=homo_data_p[:,:,tr]
he=hetero_data_p[:,:,tr]
ho_la=homo_labels 
he_la=hetero_labels 
Info=PrefDirInfo

temp=run_parallel_the_pop_decoding(homo_data_p,hetero_data_p,homo_labels,hetero_labels,PrefDirInfo,nt)
# temp shape is 8 8 4 120

def run_parallel_the_pop_decoding(ho,he,ho_la,he_la,Info,nt):
	
	n_cpu = mp.cpu_count()  # Total number of CPU
	pool = mp.Pool(n_cpu) 
	
	time_step_results = [pool.apply_async(pop_decode_at_a_single_timept,args=(ho[:,:,tr],he[:,:,tr],ho_la,he_la,Info)) for tr in range(nt)]   
	pool.close()
	pool.join()

	results = [r.get() for r in time_step_results]   
 
	return np.stack(results,axis=-1)  # neurons X presented stimuls X (mean, std) X time	

 
def pop_decode_at_a_single_timept(ho,he,ho_la,he_la,Info):
	
	"""
	Population decoding of neural data at single time point for all stimulus values
	
	Given a stimulus, what is the response of neurons that prefers different directions
	(population curve construction)
	"""
	
	# initialization for mean and standard devivation
	res_ho_mean=np.zeros((ns,ns))
	res_he_mean=np.zeros_like(res_ho_mean) 
	res_ho_std=np.zeros_like(res_ho_mean)
	res_he_std=np.zeros_like(res_ho_mean)
	
	# In a given trial a stimuls is presented  
	for k in range(1,ns+1):  # Given a stimulus (directions are coded from 1 to 8)
		
		# presented trials
		homo_trials=np.where(ho_la==k)[0]  # cannot be empty
		hetero_trials=np.where(he_la==k)[0] # cannot be empty
	
		# subset the data based on trials
		ho_subset_1=ho[:,homo_trials]
		he_subset_1=he[:,hetero_trials]
		
		# For each tuning neuron (subset the data based on neurons)
		for l in range(1,ns+1): 
			idx_homo=np.where(Info['Pref.Homo']==l)[0]
            #idx_homo=np.array(Info[Info['Pref.Homo']==l]['Neuron'])
			idx_hetero=np.where(Info['Pref.Hetero']==l)[0] 
			#idx_hetero=np.array(Info[Info['Pref.Hetero']==l]['Neuron'])
            
			# subsetted data based on the tuned neurons (if they exists)
			if len(idx_homo)!=0:
				ho_subset2=ho_subset_1[idx_homo,:]
				ho_mean=np.mean(ho_subset2.flatten())
				ho_std=np.std(ho_subset2.flatten())
			else: # if neuron group does not exist
				ho_mean=0
				ho_std=0
			if len(idx_hetero)!=0:
				he_subset2=he_subset_1[idx_hetero,:]
				he_mean=np.mean(he_subset2.flatten())
				he_std=np.std(he_subset2.flatten())
			else:
				he_mean=0
				he_std=0
				 
			res_ho_mean[l-1,k-1]=ho_mean
			res_he_mean[l-1,k-1]=he_mean

			res_ho_std[l-1,k-1]=ho_std
			res_he_std[l-1,k-1]=he_std 
			
	return np.stack((res_ho_mean,res_he_mean,res_ho_std,res_he_std),axis=-1) 
	


plt.imshow(res_ho_mean)
plt.plot(res_ho_mean)
plt.plot(res_he_mean)
plt.plot(res_he_mean[:,7])
