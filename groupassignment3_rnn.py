"""
BZAN 554 Deep Learning - Group Project #3

The goal is to predict – a couple of days before it happens - the gross margin, per unit of time shown, 
of a given product in a given show and to tune the learning procedure. Model performance is judged against
the validation data set.

Authors: Aileen Barry, Luke McConnell, Harris Taylor
"""

######################
#### INSTRUCTIONS ####
######################

#Part 1: Make the best possible model, as measured on a validation set.
# You can use anything we have covered up until and including session 17. 
# Better validation performance translates to a higher grade on the assignment.

#Part 2: Make the best possible RNN, and compare its performance with the model from part 1. 

#Use the data, inputs and outputs as specified in group assignment 2.


## IMPORT MODULES
import tensorflow as tf
import datetime as datetime
import numpy as np
import pandas as pd
import itertools as it

## SET LOCATION OF pricing.csv FILE
location_of_data = 'C:\\Users\\Aileen Barry\\Documents\\BZAN554\\pricing.csv'

##############################
#### STEP 0: prepare data ####
##############################

data = pd.read_table(location_of_data,
                     delimiter = ';',
                     dtype = {'master_id': np.int, 
                              'show_batch': np.unicode, 
                              'unit_offer_price': np.float, 
                              'quantity': np.float, 
                              'unit_cost': np.float,
                              'gross_margin_product': np.float,
                              'gross_margin': np.float,
                              'gross_margin_new_customers': np.float,
                              'gross_margin_product_new_customers': np.float,
                              'host_full_name_1_array': np.int,
                              'show_brand_label_1_array': np.int, 
                              'show_type_array': np.int,
                              'showing_start_date_time_min': np.float,
                              'showing_end_date_time_max': np.float,
                              'adjusted_duration_seconds_sum': np.float, 
                              'merch_department': np.int, 
                              'merch_class_name': np.int,
                              'country_of_origin': np.int})

# data.shape
# data.info()
# data.describe()

#For convenience, we can remove these variables that we do not need (but this is not required)

del data['show_batch']
del data['quantity']
del data['gross_margin_product']
del data['gross_margin_new_customers']
del data['gross_margin_product_new_customers']
del data['showing_end_date_time_max']

# data.info()

# Get the mean and standard deviation of numerical features

def mean_and_std(var_name):
    return (np.mean(data[var_name]) , np.std(data[var_name]))

    
unit_cost_mean, unit_cost_std = mean_and_std('unit_cost')    
showing_start_date_time_min_mean, showing_start_date_time_min_std = mean_and_std('showing_start_date_time_min')    
unit_offer_price_mean, unit_offer_price_std = mean_and_std('unit_offer_price')

#Create function that we will use to create sparse dummy variables for categorical variables
#Categoricals (are already integer encoded)

def sparse_tensor(X_cat_int, nbr_unique):
    indices = []
    X_cat_int = list(X_cat_int)
    for i in range(len(X_cat_int)):
        indices.append([i,X_cat_int[i]])
    return tf.sparse.SparseTensor(indices = indices, 
                                  values = np.ones(len(X_cat_int)), 
                                  dense_shape = [len(X_cat_int),nbr_unique])

#Compute number of categories
master_id_nbr_unique = np.max(np.unique(data['master_id'])) + 1
host_full_name_1_array_nbr_unique = np.max(np.unique(data['host_full_name_1_array']))  + 1
show_brand_label_1_array_nbr_unique = np.max(np.unique(data['show_brand_label_1_array']))  + 1
show_type_array_nbr_unique = np.max(np.unique(data['show_type_array']))  + 1
merch_department_nbr_unique = np.max(np.unique(data['merch_department']))  + 1
merch_class_name_nbr_unique = np.max(np.unique(data['merch_class_name']))  + 1
country_of_origin_nbr_unique = np.max(np.unique(data['country_of_origin']))  + 1

total_number_instances = len(data)

# order dataframe by showing_start_date_time_min
data = data.sort_values(by = ['showing_start_date_time_min'])

##################################
#### STEP 2: prepare fuction #####
##################################

#Used to specify architecture and compile model using grid

#start outer loop to cycle through the grid
    #tf code that takes the options
def prepare_data():

        #destroy any previous models
        tf.keras.backend.clear_session()        
        
       
        X_num_inputs = tf.keras.layers.Input(shape=(3,), name='X_num_inputs')
        X_master_id_sparse_inputs = tf.keras.layers.Input(shape=(master_id_nbr_unique,), sparse = False, name='X_master_id_sparse_inputs')
        X_host_full_name_1_array_sparse_inputs = tf.keras.layers.Input(shape=(host_full_name_1_array_nbr_unique,), sparse = False, name='X_host_full_name_1_array_sparse_inputs')
        X_show_brand_label_1_array_sparse_inputs = tf.keras.layers.Input(shape=(show_brand_label_1_array_nbr_unique,), sparse = False, name='X_show_brand_label_1_array_sparse_inputs')
        X_show_type_array_sparse_inputs = tf.keras.layers.Input(shape=(show_type_array_nbr_unique,), sparse = False, name='X_show_type_array_sparse_inputs')
        X_merch_department_sparse_inputs = tf.keras.layers.Input(shape=(merch_department_nbr_unique,), sparse = False, name='X_merch_department_sparse_inputs')
        X_merch_class_name_sparse_inputs = tf.keras.layers.Input(shape=(merch_class_name_nbr_unique,), sparse = False, name='X_merch_class_name_sparse_inputs')
        X_country_of_origin_sparse_inputs = tf.keras.layers.Input(shape=(country_of_origin_nbr_unique,), sparse = False, name='X_country_of_origin_sparse_inputs')

        concatenated = tf.keras.layers.concatenate([X_num_inputs,
                                                    X_master_id_sparse_inputs,
                                                    X_host_full_name_1_array_sparse_inputs,
                                                    X_show_brand_label_1_array_sparse_inputs,
                                                    X_show_type_array_sparse_inputs,
                                                    X_merch_department_sparse_inputs,
                                                    X_merch_class_name_sparse_inputs,
                                                    X_country_of_origin_sparse_inputs])        
        

##################################
###### STEP 3: train model #######
################################## 
inputs = tf.keras.layers.Input(shape = (None,1)) #no need to specify length of sequence (set first dimension to None)
rrn1 = tf.keras.layers.LSTM(100, return_sequences = True)(inputs) 
rrn2 = tf.keras.layers.LSTM(50, return_sequences = True)(rrn1) 
rrn3 = tf.keras.layers.LSTM(20, return_sequences = True)(rrn2) #NOTE return_sequences = True
outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(rrn3) 

#Create model 
model = tf.keras.Model(inputs = inputs, outputs = outputs)

#Compile model
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = 0.001))

#Fit model
model.fit(x=X_train,y=y_train, batch_size=10, epochs=10) #this can be run any number of times and it will start from the last version of the weights. To reset the weights, rerun the specification to trigger the random initialization.

y_hat = model.predict(X_test)
y_hat.shape
#get the value of the last time step only (as that is the only one we care about in 
#a prediction context).
np.mean(tf.keras.losses.mean_squared_error(y_test[:,-1,:],y_hat[:,-1,:]))
#0.012704682               
#About the same. What happens when you get more batches and do more epochs?  
    