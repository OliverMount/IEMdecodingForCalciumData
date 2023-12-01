def run_parallel_the_pop_decoding(ho,he,ho_la,he_la,Info,nt):
	
	n_cpu = mp.cpu_count()  # Total number of CPU
	pool = mp.Pool(n_cpu) 
	
	time_step_results = [pool.apply_async(pop_decode_at_a_single_timept,args=(ho,he,ho_la,he_la,Info)) for tr in range(nt)]   
	pool.close()
	pool.join()

	results = [r.get() for r in time_step_results]   
 
	return np.stack(results,axis=-1).transpose(0,2,1)   # no X time X (homo,hetero)



def pop_decode_at_a_single_timept(ho,he,ho_la,he_la,Info):
	
	"""
	Population decoding of neural data at single time point for all stimulus values
	
	Given a stimulus, what is the response of neurons that prefers different directions
	(population curve construction)
	"""
	
	
	for k in range(1,ns+1):  # Given a stimulus (directions are coded from 1 to 8)
		pass 
