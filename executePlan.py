import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from multiprocessing import Pool
from keras import backend as K
#from scipy.interpolate import make_interp_spline, BSpline
from itertools import repeat
from functools import partial
pd.options.mode.chained_assignment = None #ignores warning when going from sort_by_index to sort_by_name
#import matplotlib.pyplot as plt

# this serves a as queue for start multiple experimental plan, they are executed consecutively
expPlanFileStrArray = ['expParameters_actual']  # define inputs and hyperparameters
dataFileStrArray = ['InputData/Input_actual']  # define Input File
expResultsStrArray = ['final_model']  # define output csv name

# these fuels won't be predicted, they help to train the model. Define before modeling
fixed_list = ['Simulated_Ethanol', 'iso-octane']

seedTrainTest = 1111 # seed to split between train and test set 
train_fraction = 1  # =1 --> no test set is empty

savePredictionsBool = True# True to save the detailed predictions
plotTrainingProgress = False # True to display training progress while finding optimal Epoch


def load_data(dataFileStrLocal, expPlanFileStrLocal):
    expPlan = False
    if expPlanFileStrLocal != False:
        expPlan = pd.read_csv(expPlanFileStrLocal + '.csv', header=0)  # load experimental plan from csv
    dataWhole = pd.read_csv(dataFileStrLocal + '.csv', header=0)  # load from csv

    dataWhole = dataWhole.set_index('Name')

    return expPlan, dataWhole

def cut_data(dataWholeLocal, inputColumns, fuelClasses):
    inputColumns = inputColumns.split()
    data = dataWholeLocal.copy()
        
    fuelClasses = fuelClasses.split() # list of relevant fuel classes
    label = inputColumns[-1] # Ron / OS / or etc.

    # saves the position of the fuels belonging to different fuel classes
    fuels_by_classes = [[] for i in repeat(None, len(fuelClasses))]
    exclude_fuels = [False] * len(data.index)
     
    data_classes = data['FuelClass']
    data_label = data[label]
    
    # exclude fuels where label is nan
    for idx, label_value in enumerate(data_label):
        if np.isnan(label_value):
            exclude_fuels[idx] = True

    # exclude fuels belonging to fuel classes not defined in expPlan
    for idx, classes in enumerate(data_classes): 
        if classes in fuelClasses and exclude_fuels[idx] == False:
            fuels_by_classes[fuelClasses.index(classes)].append(data_classes.index.values[idx])
        else:
            exclude_fuels[idx] = True

    exclude_fuels = pd.Series(np.array(exclude_fuels), index = data.index)
    data = data[exclude_fuels == False]

       # select features as defined in expPlan
    data = data[inputColumns]

    return data, fuels_by_classes

def arrange_data(seed, data, train_fraction_local,yName):
    # splitting up the dataset into train and test set while shuffling, save complete train data after shuffling
    x_train = data.sample(frac=train_fraction_local, random_state=seed)
    x_test = data.drop(x_train.index)

    # Extract the label from the features dataframe.
    y_train = x_train.pop(yName)
    y_test = x_test.pop(yName)

    return (x_train, y_train), (x_test, y_test)


def shuffle_train_data(seed, x_train_unshuff, y_train_unshuff): 
    numpy64 = False # turns true when only one y-value per fuel (for SDP Validation)

    # unique fuel names of features and labels
    x_unique = x_train_unshuff.loc[~x_train_unshuff.index.duplicated(keep='first')]
    y_unique = y_train_unshuff.loc[~y_train_unshuff.index.duplicated(keep='first')]

    # complete repeated fuel names of features and labels 
    x_complete = x_train_unshuff
    y_complete = y_train_unshuff

    # shuffle x/y_unique 
    x_shuffled = x_unique.sample(frac=1, random_state=seed)
    y_shuffled = y_unique.sample(frac=1, random_state=seed)

    appended_x = []
    appended_y = []  

    # add repeated fuel names to new order of unique names
    for i, row in x_shuffled.iterrows():
        t_dfx = x_complete.loc[i]
        t_dfy = y_complete.loc[i]
	
        if isinstance(t_dfx, pd.Series):
            t_dfx = t_dfx.to_frame().T
            t_dfy = pd.Series(t_dfy, index=[i], name='LBV').transpose()
            numpy64 = True 

        appended_x.append(t_dfx)
        appended_y.append(t_dfy)
 
    x_train = pd.concat(appended_x)
    y_train = pd.concat(appended_y)

    shuffledFuelOrder = x_train['HelpIdx']
    shuffledFuelOrder = shuffledFuelOrder.to_frame()
    x_train = x_train.drop(columns=['HelpIdx']) # HelpIdx will not be trained


    return x_train, y_train, shuffledFuelOrder, numpy64

def normalizeDataStd(train_data, train_labels, test_data, test_labels):
    # xdata gets normalized with mean value and std
    concatHelpIdx = False 

    if 'HelpIdx' in train_data.columns: # activates for validation data containing HelpIdx, for final model no HelpIdx should be in InputData
        concatHelpIdx = True
        helpIdx = train_data['HelpIdx'] #dont normalize HelpIdx
        train_data = train_data.drop(columns=['HelpIdx'])

    mean = [train_data.mean(axis=0)]
    std = [train_data.std(axis=0)]   

    train_data = (train_data - mean[0]) / std[0]
    test_data = (test_data - mean[0]) / std[0]

    # labels get normalized from 0 (lowerst) to 1 (highest)
    mean_label = train_labels.mean(axis=0)
    std_label = train_labels.std(axis=0)

    train_labels = (train_labels - mean_label) / std_label
    test_labels = (test_labels - mean_label) / std_label
    
    if concatHelpIdx:
        train_data = pd.concat([train_data, helpIdx], axis=1)

    return train_data, train_labels, test_data, test_labels, mean_label, std_label


def denormLabels(labels, mean_label, std_label):
    # Linear Normalizing (min to max)
    labels = mean_label + labels * std_label
    return labels

def buildModel(modelHyperparameters, inputshape):
    #hyperparameters and the number of inputs are required
    (n1, n2, n3, d1, d2, d3, activation, alpha, beta1, beta2) = modelHyperparameters

    #constraint_val = 1
    #constraint = keras.constraints.max_norm(max_value=constraint_val,axis=0) # nax-norm regularization
    constraint = None

    # builds a model with 1, 2 or 3 layers
    if n2 == 0:
        model = keras.Sequential()
        model.add(keras.layers.Dense(n1, activation=activation, kernel_constraint=constraint, input_shape=(inputshape,)))
        model.add(keras.layers.Dropout(d1))
        model.add(keras.layers.Dense(1))

    elif n3 == 0:
        model = keras.Sequential()
        model.add(keras.layers.Dense(n1, activation=activation, kernel_constraint=constraint,  input_shape=(inputshape,)))
        model.add(keras.layers.Dropout(d1))
        model.add(keras.layers.Dense(n2, activation=activation, kernel_constraint=constraint))
        model.add(keras.layers.Dropout(d2))
        model.add(keras.layers.Dense(1))

    else:
        model = keras.Sequential()
        model.add(keras.layers.Dense(n1, activation=activation, kernel_constraint=constraint, input_shape=(inputshape,)))
        model.add(keras.layers.Dropout(d1))
        model.add(keras.layers.Dense(n2, activation=activation, kernel_constraint=constraint))
        model.add(keras.layers.Dropout(d2))
        model.add(keras.layers.Dense(n3, activation=activation, kernel_constraint=constraint))
        model.add(keras.layers.Dropout(d3))
        model.add(keras.layers.Dense(1))

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1, beta2=beta2)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])

    return model

def sort_by_name_kf(train_data, y_data):

    unique_names = train_data.index.unique() # creates a vector with unique names
    train_data["label"] = y_data.values	 # adds y-values to input train_data, in the right order 
	
    values_names =  []
    y_data_names = []

    # Sort fuels by name as an array
    for unique_name in unique_names:
        values_name = train_data.loc[unique_name] #stores rows of current repeated fuel name

        if isinstance(values_name, pd.Series):
            values_name = values_name.to_frame().T

        y_data_name = values_name["label"].values # stores the y-values of current repeated fuel name 
        values_name = values_name.drop("label", axis=1) # train_data matrix shouldn't contain at the end train_labels

        # transforms stored dataFrame values into a np array 
        values_names.append(values_name.values) 
        y_data_names.append(y_data_name)
	
    #Array of arrays  	
    train_data = np.array(values_names)
    y_data = np.array(y_data_names)

    return train_data, y_data


def split_by_index(x_data, y_data, fixed_list):

    x_list = []
    y_list = []

    #fixes a fuel in training set, fixed_value must be changed manually before modeling
    for name in fixed_list:

        mask = (x_data.index == name)
        x_to_be_fixed = x_data.loc[mask]
        x_to_be_fixed = x_to_be_fixed.drop(columns=['HelpIdx'])
        x_data = x_data.loc[~mask]
        y_to_be_fixed = y_data.loc[mask]
        y_data = y_data.loc[~mask]
 
        x_list.append(x_to_be_fixed)
        y_list.append(y_to_be_fixed) 
 
    x_fixed = pd.concat(x_list)
    y_fixed = pd.concat(y_list)

    return x_fixed, x_data, y_fixed, y_data


def kFoldTrain(train_data, train_labels, numberKF, modelHyperparameters, x_fixed, y_fixed, optEpoch):
    #  K fold cross validation
    kf = KFold(numberKF)  # does not shuffle
    foldData = []
    foldLabel = []
    foldPrediction = []

    # Only data without "Fuels to be fixed" will be splitted into folds
    train_data, train_labels = sort_by_name_kf(train_data, train_labels)

    #  trains model for each fold, predicts ON and stores Variables with OPTIMAL Epochs
    fold = 0

    for foldTrain, foldVal in kf.split(train_data):
        # np.concatenate so to transform "array of arrays" into np matrix
        x_foldTrain = np.concatenate(train_data[foldTrain], axis=0)
        y_foldTrain = np.concatenate(train_labels[foldTrain], axis=0)
        x_foldVal = np.concatenate(train_data[foldVal], axis=0)
        y_foldVal = np.concatenate(train_labels[foldVal], axis=0)
       
        ### add fixed_values to the training set
        x_foldTrain = np.concatenate([x_foldTrain, x_fixed.values])
        y_foldTrain = np.concatenate([y_foldTrain, y_fixed.values])

        model = buildModel(modelHyperparameters, x_foldTrain.shape[1])


        # Train model with max number of epochs and store training stats
        history = model.fit(x_foldTrain, y_foldTrain, epochs=optEpoch, validation_data=(x_foldVal, y_foldVal), verbose=0)
        #history saves various information about the training progress

        # store data of Validation set and its prediction
        foldLabel.append(y_foldVal)

        foldPrediction.append(model.predict(x_foldVal).flatten())

        fold += 1
    # Convert storage matrices into vectors
    foldPrediction = np.concatenate(foldPrediction)
    foldLabel = np.concatenate(foldLabel)
    #clear some memory
    tf.keras.backend.clear_session()
    K.clear_session()
    return foldPrediction, foldLabel

def findOptEpochs(train_data_list, train_labels_list, numberKF, modelHyperparameters, x_fixed, y_fixed, EPOCHS):
    
    #return 50
    """
    This trains the 20 models with random initilization and random train/validation set
    With cross-validation, it finds the oest validation mean squared error over all epochs.
    The validation mse is avaraged for the 20 models for each epoch count, then the Epoch count with minimal average mse is returned
    """
    optEpoch = 1
    test_runs = 20 # total runs will be len(seed_list) * numberKf
    filter = 10 # an area of +- 10 epochs is used to evaluate the optimal epoch

    fold = 0
    kf = KFold(numberKF)  # does not shuffle
    valError = []
    #
    for (train_data, train_labels) in zip(train_data_list, train_labels_list):
        #This was from RON/MON model
        train_data = train_data.values
        train_labels = train_labels.values
  
        for foldTrain, foldVal in kf.split(train_data):
            
            if fold <= max(test_runs/len(train_data_list),1):
                
                x_foldTrain = train_data[foldTrain]
                y_foldTrain = train_labels[foldTrain]
                x_foldVal = train_data[foldVal]
                y_foldVal = train_labels[foldVal]

                ### add fixed_values to the training set
                x_foldTrain = np.concatenate([x_foldTrain, x_fixed.values])
                y_foldTrain = np.concatenate([y_foldTrain, y_fixed.values])
                
                # build model
                model = buildModel(modelHyperparameters, train_data.shape[1])
                
                # Train model with max number of epochs and store training stats
                history = model.fit(x_foldTrain, y_foldTrain, epochs=EPOCHS, validation_data=(x_foldVal, y_foldVal), verbose=0)

                # monitor training and validation loss
                if plotTrainingProgress:
                    #smoothing
                    loss_red = []
                    error_red = []
                    loss = history.history['loss']
                    error = history.history['val_loss']
                    loss = list(loss)

                    error = list(error)
                    xnew = np.arange(0, EPOCHS, 10)
                    # plot every 10th entry
                    for x in xnew:
                        loss_red.append(loss[x])
                        error_red.append(error[x])

                    plt.rcParams.update({'font.size': 26})

                    plt.plot(xnew, loss_red, linewidth=3)
                    plt.plot(xnew, error_red, linewidth=3)
                    plt.xticks([0,2000,4000,6000,8000,10000])

                    #plt.plot(history.history['loss'], linewidth=1)
                    #plt.plot(history.history['val_loss'], linewidth=1)
                    plt.title('Sigmoid')
                    plt.ylabel('normed MSE')
                    plt.xlabel('Training epoch')
                    plt.legend(['Loss', 'Prediction error'], loc='upper right')
                    axes = plt.gca()
                    axes.set_xlim([-50, EPOCHS])
                    axes.set_ylim([-0.01, 0.5])
                    plt.show()

                fold += 1

                valError.append(np.array(history.history['val_mean_squared_error']))

    # mean Error
    sumError = np.mean(np.square(valError), axis = 0)

    for i,element_i in enumerate(sumError):
        if i >= filter-1 and i <= len(sumError)-1-filter:
            sum_error_filtered = np.mean(sumError[i-filter+1:i+filter-1])

            if i == filter-1:
                min_sum_error_filtered = sum_error_filtered

            elif sum_error_filtered < min_sum_error_filtered:
                min_sum_error_filtered = sum_error_filtered
                optEpoch = i
    
    return optEpoch

def calculateRow(row, dataWhole):
    foldPrediction = []  # saves predictions on the 5 folds
    shuffledFuelOrder = []  # saves the shuffled Fuel Order, to unshuffle the results later
    global yName

    expStr = row[0]
    fuelClasses = row[1]
    inputFeatures = row[2]
    testSeedList = row[3]
    activationStr = row[4]
    numberKF = row[5]
    n1 = row[6]
    n2 = row[7]
    n3 = row[8]
    d1 = row[9]
    d2 = row[10]
    d3 = row[11]
    alpha = row[12]
    beta1 = row[13]
    beta2 = row[14]
    maxEpoch = row[15]
    adjustment = row[16] # indicates the feature adjustment made for this row
    iterationStep = row[17]

    numberOfFeatures = len(inputFeatures.split())

    # sets neuron activation according to the exp plan, defaults to sigmoid
    if activationStr == 'relu' or activationStr == 'ReLU':
        activation = tf.nn.relu
    elif activationStr == 'tanh'or activationStr == 'Tanh':
        activation = tf.nn.tanh
    elif activationStr == 'leaky'or activationStr == 'leaky ReLU' or activationStr == 'Leaky':
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.sigmoid

    modelHyperparameters = (n1, n2, n3, d1, d2, d3, activation, alpha, beta1, beta2)

    # cut relevant data from raw data, save the used input Columns to name the output file
    data, fuels_by_classes = cut_data(dataWholeLocal=dataWhole, inputColumns=inputFeatures, fuelClasses=fuelClasses)
    yName = list(data.columns.values)[-1]

        # arrange data and split into train and test set
    # train fraction should be 1 when using cross validation, therefore the test set is empty
    (train_data, train_labels), (test_data, test_labels) = arrange_data(seed=seedTrainTest, data=data,
                                                                        train_fraction_local=train_fraction,
                                                                        yName=yName)

    # normalize data
    train_data, train_labels, test_data, test_labels, label_mean, label_std = normalizeDataStd(train_data, train_labels,
                                                                                    test_data, test_labels)

    # separates between fixed data only for training folds and data for training and validation folds
    x_fixed, train_data, y_fixed, train_labels = split_by_index(train_data, train_labels, fixed_list)
    
    # prepare iteration on testSeeds
    if type(testSeedList) == int:
        testSeedList = [testSeedList]
        nseed = 1

    else:
        testSeedList = testSeedList.split()
        testSeedList = list(map(int, testSeedList))
        nseed = len(testSeedList)

    # storage variables for different seeds, different seeds shuffle the input data rows
    train_data_list = [0] * nseed
    train_labels_list = [0] * nseed
    shuffledFuelOrder_list = []
    #foldPrediction_list = [0] * nseed
    foldLabel_list = [0] * nseed
    r2_list = [0] * nseed
    mae_list = [0] * nseed

    foldPrediction_list = []
    seedCounter = 0

    for seed in testSeedList:
        # shuffle train_data, labels and the fuel order

        tdata, tlabel, shuffledFuelOrd, numpy64 = shuffle_train_data(seed, train_data, train_labels)
        train_data_list[seedCounter] = tdata
        train_labels_list[seedCounter] = tlabel
        shuffledFuelOrder_list.append(shuffledFuelOrd)
        seedCounter += 1
    
    seedCounter = 0
    # Find optimal Epoch based on first seed
    
    optEpoch = findOptEpochs(train_data_list, train_labels_list, numberKF, modelHyperparameters, x_fixed, y_fixed, EPOCHS=maxEpoch)
    # iterate over the different seeds of fuel orders
    for train_data in train_data_list:
 	
        train_labels = train_labels_list[seedCounter]
        shuffledFuelOrder = shuffledFuelOrder_list[seedCounter]

        # Train Model (kfold), optimize for epoch number
        foldPrediction, foldLabel = kFoldTrain(train_data, train_labels, numberKF, modelHyperparameters, x_fixed, y_fixed,
                                               optEpoch=optEpoch)

        foldPrediction = pd.DataFrame(foldPrediction, columns=['Predicted ' + yName])
        foldLabel= pd.DataFrame(foldLabel, columns=['Predicted ' + yName])
       
        # denorm
        foldPrediction = denormLabels(foldPrediction, label_mean, label_std)
        foldLabel = denormLabels(foldLabel, label_mean, label_std)
        
        # Calculate correlation coefficient between measured and predicted RON RA
        r2 = r2_score(foldLabel, foldPrediction)
        r2_list[seedCounter] = r2

        # calculate MEA for different fuel species
        mae = np.mean(np.abs(foldLabel-foldPrediction))
        mae_list[seedCounter] = mae
        
        # reorder fuels with HelpIdx    
           
        #foldPrediction['foldLabel'] = foldLabel # activate for checking wheter foldlabel goes to the right foldPrediction
        shuffledFuelOrder = shuffledFuelOrder.reset_index()
        foldPrediction  = pd.concat([foldPrediction, shuffledFuelOrder], axis=1)
        foldPrediction = foldPrediction.set_index('HelpIdx')

        if not numpy64: # for SDP validation
        	foldPrediction = foldPrediction.drop(columns=['Name'])

        foldPrediction_list.append(foldPrediction)

        print("Iteration Step: {}".format(iterationStep)
              +", inputs = {}".format(inputFeatures)
              + " r2 = {}".format(r2_list[seedCounter]))

        seedCounter += 1


    foldPrediction_list = pd.concat(foldPrediction_list, axis=1)
    mean_pred = foldPrediction_list.mean(axis=1)
    foldPrediction_list['std'] = foldPrediction_list.std(axis=1)
    foldPrediction_list['mean'] = mean_pred


    # save data file
    data = data.reset_index()
    data = data.set_index('HelpIdx')
    foldPrediction_list = foldPrediction_list.sort_index()
    saveFile = pd.concat([data, foldPrediction_list], axis=1, sort=False).reindex(data.index)
    saveFile = saveFile.reset_index()
    saveFile = saveFile.set_index('Name') 

    #abs_error
    saveFile['abs_error'] = abs(saveFile['mean']-saveFile[inputFeatures.split()[-1]])


    # calculate mean correlation with standard deviation, since it's calculated between foldprediciton and foldlabel there's no math errordue to NaN 
    r2_mean = np.mean(r2_list)
    r2_std = np.std(r2_list)
    mae_mean = np.mean(mae_list)


    result_row = [expStr,
                 fuelClasses,
                 inputFeatures,
                 testSeedList,
                 activationStr,
                 numberKF,
                 n1, n2, n3, d1, d2, d3, alpha, beta1, beta2,
                 maxEpoch,
                 adjustment,
                 iterationStep,
                 numberOfFeatures,
                 optEpoch,
                 r2_mean,
                 r2_std, mae_mean]
    


    # calculate MAE for each fuel class
    mean_absolute_error_classes = []
    mean_absolute_errors = []
    
    for idx, fuel_Class in enumerate(fuelClasses.split()):
        mean_absolute_errors.append(np.mean(saveFile.loc[fuels_by_classes[idx], 'abs_error']))
        mean_absolute_error_classes.append(fuel_Class)


    result_row.append(mean_absolute_error_classes)
    result_row.append(mean_absolute_errors)
  
    # mean error with sign (Bias)
    unique_names2 = saveFile.index.unique()
    fuel_unique = []
    err_sign = []

    for name in unique_names2:

        current_fuel = (saveFile.index == name)
        phi_interval = saveFile.loc[current_fuel]
        error_with_sign = np.mean(phi_interval['mean']-phi_interval[inputFeatures.split()[-1]])
        fuel_unique.append(name)
        err_sign.append(error_with_sign)

    result_row.append(fuel_unique)
    result_row.append(err_sign)    


    # save save_file to csv
    if savePredictionsBool:
        saveFile = saveFile.reset_index()        
        saveFile.to_csv(expStr + '_Predictions.csv', index=False)
        print('Predictions saved: ' + expStr + '_Predictions.csv')

        # clear some memory
        K.clear_session()

    return result_row

def start_main(argv):
    assert len(argv) == 1
    for idx,expPlanFileStr in enumerate(expPlanFileStrArray):
        dataFileStr=dataFileStrArray[idx]
        expResultsStr=expResultsStrArray[idx]
        main(expPlanFileStr, dataFileStr, expResultsStr)

def main(expPlanFileStr, dataFileStr, expResultsStr):
    global yName

    # Load dataset and exp Plan from csv
    expPlan, dataWhole = load_data(dataFileStr, expPlanFileStr)

    threadCalc = partial(calculateRow,dataWhole=dataWhole) #creates partial function with only 1 variable
    p = Pool(3) # allows multiple threads

    # split the information from expPlan into rows
    rows = expPlan.itertuples(index=False, name=None)
    rows=list(rows) 

    rowList = []
    for row in rows:
      rowList.append(list(row))
     
	# all model training and saving takes place here
    expResults=p.map(threadCalc,rowList)

    p.close()
    p.join()
    # wait for all threads to finish

    expResults = pd.DataFrame(expResults, columns=['Exp',
                                                   'fuelClasses',
                                                   'inputFeatures',
                                                   'testSeed',
                                                   'activation',
                                                   '#folds',
                                                   'n1','n2','n3','d1','d2','d3','alpha','beta1','beta2',
                                                   'maxEpoch',
                                                   'adjustment',
                                                   'iterationStep',
                                                   'number of features',
                                                   'OptEpoch',
                                                   'mean corrSq',
                                                   'std','mean mae','Fuel classes','MAE of fuel class','Phi Interval','Bias'])

    print('EXPRESULTS:')
    print(expResults)
    expResults.to_csv(expResultsStr + '.csv', index= False)
    print('Experimental Results saved: '+ expResultsStr + '.csv')

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    #tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.app.run(main=start_main)

	start_main([1])

