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
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

## SET LOCATION OF pricing.csv FILE
#location_of_data = 'C:\\Users\\Aileen Barry\\Documents\\BZAN554\\pricing.csv'
location_of_data = '/Users/lukemcconnell/Desktop/class/Deep Learning/Tuning-NeuralNetwork/pricing.csv'

##############################
#### STEP 0: prepare data ####
##############################

data = pd.read_csv(location_of_data,
                     delimiter = ';',
                     dtype = {'master_id': np.int, 
                              'show_batch': np.unicode, 
                              'unit_offer_price': np.float32, 
                              'quantity': np.float32, 
                              'unit_cost': np.float32,
                              'gross_margin_product': np.float32,
                              'gross_margin': np.float32,
                              'gross_margin_new_customers': np.float32,
                              'gross_margin_product_new_customers': np.float32,
                              'host_full_name_1_array': np.int,
                              'show_brand_label_1_array': np.int, 
                              'show_type_array': np.int,
                              'showing_start_date_time_min': np.float32,
                              'showing_end_date_time_max': np.float32,
                              'adjusted_duration_seconds_sum': np.float32, 
                              'merch_department': np.int, 
                              'merch_class_name': np.int,
                              'country_of_origin': np.int})

# Sort the data by time and over-write the existing file
data = data.sort_values(by = 'showing_start_date_time_min')
data.to_csv(location_of_data)

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

del data

##############################
#### STEP 1: prepare grid ####
##############################
#STEP 1: prepare grid

hidden_layers = 3
grid = []
for hidden_layers in [1,2,3]:
    nbr_hidden_layers = [[1],[2],[3]][hidden_layers-1]
    nbr_hidden_neurons_per_layer = list(it.product([30,20,10],repeat=hidden_layers)) #Need to be a list of lists (not tuple which is immutable and we want to modify this later)
    hidden_activation_function_per_layer = list(it.product(["relu","sigmoid","tanh"],repeat=hidden_layers)) #Needs to be a list of lists
    optimizer = ['SGD','SGDwithLRscheduling','RMSprop','Adam']
    learning_rate = [0.0001,0.001,0.01]
    batch_size = [1,50,100]

    #Create cartesian product
    grid.extend(tuple(it.product(nbr_hidden_layers,
                                nbr_hidden_neurons_per_layer,
                                hidden_activation_function_per_layer,
                                optimizer,
                                learning_rate,
                                batch_size)))



#check if     
# len(grid)
# len(set(grid))
#combinations 1 layer    + combinations 2 layers      +    combinations 3 layers
# 1*3*3*4*3*3              +   1*(3**2)*(3**2)*4*3*3    +   1*(3**3)*(3**3)*4*3*3


#STEP 2: Prepare fuction to specify architecture and compile model using grid

#start outer loop to cycle through the grid
    #tf code that takes the options
def build_model(nbr_hidden_layers = 3, 
                nbr_hidden_neurons_per_layer = [5,3,2],
                hidden_activation_function_per_layer = ['relu','relu','relu'],
                optimizer = 'SGD',
                learning_rate = 0.001,
                batch_size = 1,
                timestep = 1):

        #destroy any previous models
        tf.keras.backend.clear_session()        
        
        #define architecture: layers, neurons, activation functions
        inputs = tf.keras.layers.Input(shape=(3 * timestep,), name = 'input' )
        
        X_num_inputs = tf.keras.layers.Input(shape=(3 * timestep,), name='X_num_inputs')
        X_master_id_sparse_inputs = tf.keras.layers.Input(shape=(master_id_nbr_unique * timestep,), sparse = False, name='X_master_id_sparse_inputs')
        X_host_full_name_1_array_sparse_inputs = tf.keras.layers.Input(shape=(host_full_name_1_array_nbr_unique * timestep,), sparse = False, name='X_host_full_name_1_array_sparse_inputs')
        X_show_brand_label_1_array_sparse_inputs = tf.keras.layers.Input(shape=(show_brand_label_1_array_nbr_unique * timestep,), sparse = False, name='X_show_brand_label_1_array_sparse_inputs')
        X_show_type_array_sparse_inputs = tf.keras.layers.Input(shape=(show_type_array_nbr_unique * timestep,), sparse = False, name='X_show_type_array_sparse_inputs')
        X_merch_department_sparse_inputs = tf.keras.layers.Input(shape=(merch_department_nbr_unique * timestep,), sparse = False, name='X_merch_department_sparse_inputs')
        X_merch_class_name_sparse_inputs = tf.keras.layers.Input(shape=(merch_class_name_nbr_unique * timestep,), sparse = False, name='X_merch_class_name_sparse_inputs')
        X_country_of_origin_sparse_inputs = tf.keras.layers.Input(shape=(country_of_origin_nbr_unique * timestep,), sparse = False, name='X_country_of_origin_sparse_inputs')

        concatenated = tf.keras.layers.concatenate([X_num_inputs,
                                                    X_master_id_sparse_inputs,
                                                    X_host_full_name_1_array_sparse_inputs,
                                                    X_show_brand_label_1_array_sparse_inputs,
                                                    X_show_type_array_sparse_inputs,
                                                    X_merch_department_sparse_inputs,
                                                    X_merch_class_name_sparse_inputs,
                                                    X_country_of_origin_sparse_inputs])        
        
        for layer in range(nbr_hidden_layers):
            if layer == 0:
                x = tf.keras.layers.Dense(units=nbr_hidden_neurons_per_layer[layer],
                                      activation=hidden_activation_function_per_layer[layer],
                                      name = 'hidden' + str(layer))(concatenated)
            else:
                x = tf.keras.layers.Dense(units=nbr_hidden_neurons_per_layer[layer],
                                      activation=hidden_activation_function_per_layer[layer],
                                      name = 'hidden' + str(layer))(x)
                
        outputs = tf.keras.layers.Dense(units=1,
                                      activation="linear",
                                      name = 'output')(x)
        
        model = tf.keras.Model(inputs = [X_num_inputs,
                                         X_master_id_sparse_inputs,
                                         X_host_full_name_1_array_sparse_inputs,
                                         X_show_brand_label_1_array_sparse_inputs,
                                         X_show_type_array_sparse_inputs,
                                         X_merch_department_sparse_inputs,
                                         X_merch_class_name_sparse_inputs,
                                         X_country_of_origin_sparse_inputs], 
                       outputs = outputs)        
        
        #define optimizer and learning rate
        if optimizer == 'SGD':
            opt = tf.keras.optimizers.SGD(lr = learning_rate)
        elif optimizer == 'SGDwithLRscheduling':
            initial_learning_rate = learning_rate
            decay_steps = 10000
            decay_rate = 1/10
            learning_schedule= tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, 
                decay_steps, 
                decay_rate)
            opt = tf.keras.optimizers.SGD(learning_rate=learning_schedule)
        elif optimizer == 'RMSprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, 
                                              rho=0.9, 
                                              momentum=0.0, 
                                              epsilon=1e-07)
        elif optimizer == 'Adam':            
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                           beta_1=0.9, 
                                           beta_2=0.999, 
                                           epsilon=1e-07)            
  
        model.compile(loss = 'mse', optimizer = opt)
    
        #batch size is passed on as this is only needed in the fit method
        
        #return model and batch_size
        return [model,batch_size]
        

# test = build_model(nbr_hidden_layers = 3, 
#                 nbr_hidden_neurons_per_layer = [5,3,2],
#                 hidden_activation_function_per_layer = ['relu','relu','relu'],
#                 optimizer = 'SGD',
#                 learning_rate = 0.001,
#                 batch_size = 1)        
# test[0].summary()


##STEP 3: train the model by reading data JIT and saving loss and training time

# grid_record = grid[0]


for grid_record in grid[:1]:

    out = build_model(nbr_hidden_layers = grid_record[0], 
                      nbr_hidden_neurons_per_layer = grid_record[1],
                      hidden_activation_function_per_layer = grid_record[2],
                      optimizer = grid_record[3],
                      learning_rate = grid_record[4],
                      batch_size = grid_record[5],
                      timestep = 5) 
    model = out[0]
    batch_size = out[1]    
    

    duration = []
    avg_loss_store = []
    epochs = 1
    timestep = 5
    timesteps = []
    for epoch in range(epochs):
        i = 0
        start = datetime.datetime.now()
        avg_loss = 0 
        reader = pd.read_table(location_of_data,
                     delimiter = ';',
                     dtype = {'master_id': np.int, 
                              'show_batch': np.unicode, 
                              'unit_offer_price': np.float32, 
                              'quantity': np.float32, 
                              'unit_cost': np.float32,
                              'gross_margin_product': np.float32,
                              'gross_margin': np.float32,
                              'gross_margin_new_customers': np.float32,
                              'gross_margin_product_new_customers': np.float32,
                              'host_full_name_1_array': np.int,
                              'show_brand_label_1_array': np.int, 
                              'show_type_array': np.int,
                              'showing_start_date_time_min': np.float32,
                              'showing_end_date_time_max': np.float32,
                              'adjusted_duration_seconds_sum': np.float32, 
                              'merch_department': np.int, 
                              'merch_class_name': np.int,
                              'country_of_origin': np.int},
                     chunksize = 1)
        
        counter = 0 
        for chunk in reader:
            

            counter = counter + 1
            
            if counter < timestep:
                timesteps.append(chunk)
                continue
        
            if i >= 100:
                break
                
            timesteps.append(chunk)
            timesteps = timesteps[-1 * timestep:]
            
            
            #prepare data HERE
            X_num = [np.array(c[['unit_cost', 'showing_start_date_time_min', 'unit_offer_price']], dtype=np.float32).reshape(1,3) for c in timesteps]
            X_num = [c - np.array([unit_cost_mean,showing_start_date_time_min_mean,unit_offer_price_mean]) for c in X_num]
            X_num = [c / np.array([unit_cost_std,showing_start_date_time_min_std,unit_offer_price_std]) for c in X_num]
            X_num = [tf.convert_to_tensor(c, dtype=tf.float32) for c in X_num]
            if timestep > 1:
                X_num = tf.keras.layers.concatenate(X_num) 
            
            #prepare one sparse tensor for each categorical state variable
            X_master_id_sparse = [tf.cast(sparse_tensor(c['master_id'],master_id_nbr_unique), dtype=tf.float32) for c in timesteps]
            X_host_full_name_1_array_sparse = [tf.cast(sparse_tensor(c['host_full_name_1_array'],host_full_name_1_array_nbr_unique), dtype=tf.float32) for c in timesteps]
            X_show_brand_label_1_array_sparse = [tf.cast(sparse_tensor(c['show_brand_label_1_array'],show_brand_label_1_array_nbr_unique), dtype=tf.float32) for c in timesteps]
            X_show_type_array_sparse = [tf.cast(sparse_tensor(c['show_type_array'],show_type_array_nbr_unique), dtype=tf.float32) for c in timesteps]
            X_merch_department_sparse = [tf.cast(sparse_tensor(c['merch_department'],merch_department_nbr_unique), dtype=tf.float32) for c in timesteps]
            X_merch_class_name_sparse = [tf.cast(sparse_tensor(c['merch_class_name'],merch_class_name_nbr_unique), dtype=tf.float32) for c in timesteps]
            X_country_of_origin_sparse = [tf.cast(sparse_tensor(c['country_of_origin'],country_of_origin_nbr_unique), dtype=tf.float32) for c in timesteps]
            
            #concatonate timesteps
            if timestep > 1:
                X_master_id_sparse = tf.keras.layers.concatenate(X_master_id_sparse) 
                X_host_full_name_1_array_sparse = tf.keras.layers.concatenate(X_host_full_name_1_array_sparse) 
                X_show_brand_label_1_array_sparse = tf.keras.layers.concatenate(X_show_brand_label_1_array_sparse) 
                X_show_type_array_sparse = tf.keras.layers.concatenate(X_show_type_array_sparse)
                X_merch_department_sparse = tf.keras.layers.concatenate(X_merch_department_sparse) 
                X_merch_class_name_sparse = tf.keras.layers.concatenate(X_merch_class_name_sparse) 
                X_country_of_origin_sparse = tf.keras.layers.concatenate(X_country_of_origin_sparse) 
            
            #y for last timestep
            y = timesteps[-1]['gross_margin'] / timesteps[-1]['adjusted_duration_seconds_sum']
            y = tf.convert_to_tensor(y)
            
            modinfo=model.fit(x=[X_num,
                                 X_master_id_sparse,
                                 X_host_full_name_1_array_sparse,
                                 X_show_brand_label_1_array_sparse,
                                 X_show_type_array_sparse,
                                 X_merch_department_sparse,
                                 X_merch_class_name_sparse,
                                 X_country_of_origin_sparse], 
                      y=y, 
                      batch_size = batch_size,
                      epochs = 1,
                      verbose = 0)
            loss = modinfo.history['loss'][0]
            avg_loss = avg_loss + (1/(i+1))*(loss - avg_loss)   
            i += 1

        #store the loss and training time per epoch in two arrays
        avg_loss_store.append(avg_loss)
        duration.append(datetime.datetime.now() - start)
        
    #store the loss and training time in RAM
    gridresults = [grid_record, avg_loss_store, duration]
    #and also on disk:
    file = open("gridresults.txt", "a")  # append mode 
    file.write(str(gridresults) + "\n")     
    file.close()    

