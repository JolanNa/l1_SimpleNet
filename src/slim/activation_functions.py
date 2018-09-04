import tensorflow as tf

#class identity_activation(object):

def identity_activation(drive, name=None):
    return drive


#class swish(object):
    
def swish(drive, beta, name=None):
    return drive * tf.nn.sigmoid(drive * beta, name=name)


