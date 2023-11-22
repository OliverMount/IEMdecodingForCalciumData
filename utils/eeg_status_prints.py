def print_status(msg,kind='short',line=50):
    
    if kind=='short':
        print('++ '+msg)
    else:
        print('++ '+line*'-')
        print('++ '+msg)
        print('++ '+line*'-')




def print_electrode_status(e,txt="Decoding"):
    
    if (len(e)!=122 or len(e)!=128): 
        if 66 in e:
            elec_type='Posterior'
        elif 97 in e:
            elec_type='Central'
        else:
            elec_type='Anterior'
    else:
        elec_type="All" 
        
    print(txt + '(' + elec_type , ' Electrodes)') 

    return elec_type
    
    
def get_electrode_group(e): 
    if (len(e)!=122 or len(e)!=128): 
        if 66 in e:
            elec_type='Posterior'
        elif 97 in e:
            elec_type='Central'
        else:
            elec_type='Anterior'
    else:
        elec_type="All" 
    return elec_type
