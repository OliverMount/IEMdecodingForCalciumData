import os 
def create_dir(p): 
    """
    Creates a directory given a path 'p'
    Examples:
    ---------
    
    create_dir('test') -> Creates a folder test in the present working directory
    create_dir('/home/user/test/) -> Creates a folder test in home/user/ directory
    """
    isExist = os.path.exists(p)  
    if not isExist:  
        os.makedirs(p)
        print('The directory ' +  p   + ' is created!')
    else:
        print('The directory ' +  p   + ' already exists. Nothing created!')
    return p


# To obtain first and last path(file or directory) 
def get_first_and_last(path):
    npath=os.path.normpath(path)
    last=os.path.basename(npath)
    first=os.path.dirname(npath)
    
    return (first,last)
