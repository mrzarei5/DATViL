import pickle

def serializeObject(object_,file_name):
    file_object = open(file_name,'wb')
    pickle.dump(object_, file_object,protocol = 2)
    file_object.close()
    return
def deserializeObject(file_name):
    file_object = open(file_name,'rb')
    object_ = pickle.load(file_object)
    file_object.close() 
    return object_