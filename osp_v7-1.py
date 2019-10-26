
"""
Sam Hughes
20/10/2019

This code presents a systm to predict ovrall patient survival given the
MICCAI BraTS 2018 image and survival data. The structure is as follows.
All parameters are set at or near the top, including file paths etc.
Statistical functions are defined. Thenteh feature extraction code begins.
This will extract spatial and grey level features from the MRI images and
output these to a .csv file. This takes approximately 2 hours. 
In the next phase, a feature seletion algorithm fits the entire feature set
to a gradient boosted tree model and ranks the features baed on importance,
selecting the best 10.
The next phase optimises the regressor hyperparameters using a grid search
algorithm. This section takes approximately 5 hours with the paraeter range
currently set. 
The optimised parameters are passed to the final model for assessment of
system performance by cross validation

REFERENCES
"MICCAI BRATS - The Multimodal Brain Tumour Segmentation Challenge",
Braintumoursegmentation.org, 2019. [Online]. Available:
http://braintumoursegmentation.org. [Accessed: 17- Oct- 2019].

M. Brett, NiBabel. MIT, 2019.

XGBoost. https://pypi.org/project/xgboost/: xgboost developers, 2019.


T. Chen and C. Guestrin, "XGBoost", Proceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining - KDD '16,
2016. Available: 10.1145/2939672.2939785 [Accessed 26 October 2019].

M. Alam et al., "Deep Learning and Radiomics for Glioblastoma Survival
Prediction", in 2018 International MICCAI BraTS Challenge, Granada, Spain,
2018, pp. 11-18

V. Fonov, A. Evans, K. Botteron, C. Almli, R. McKinstry and D. Collins, "Unbiased
average age-appropriate atlases for pediatric studies", NeuroImage, vol. 54, no. 1,
pp. 313-327, 2011. Available: 10.1016/j.neuroimage.2010.07.033 [Accessed 25 October 2019].

V. Fonov, A. Evans, R. McKinstry, C. Almli and D. Collins, "Unbiased nonlinear average
age-appropriate brain templates from birth to adulthood", 2019. .

D. Collins, A. Zijdenbos, W. Baaré and A. Evans, "ANIMAL+INSECT: Improved Cortical
Structure Segmentation", 2019. .

Lowekamp BC, Chen DT, Ibáñez L and Blezek D (2013) The Design of SimpleITK. Front.
Neuroinform. 7:45. doi: 10.3389/fninf.2013.00045

Z. Yaniv, B. C. Lowekamp, H. J. Johnson, R. Beare, "SimpleITK Image-Analysis Notebooks:
a Collaborative Environment for Education and Reproducible Research", J Digit Imaging.,
https://doi.org/10.1007/s10278-017-0037-8, 2017

van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V.,
Beets-Tan, R. G. H., Fillon-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017).
Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research,
77(21), e104–e107. `https://doi.org/10.1158/0008-5472.CAN-17-0339 <https://doi.org/
10.1158/0008-5472.CAN-17-0339>`_



"""
import os
import numpy as np
from radiomics import featureextractor, getTestCase
import yaml
import csv
from sklearn.decomposition import PCA
import random
from collections import OrderedDict, Counter
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold


#____________________________ File Paths ______________________________#
#set working directory
WDir = 'work_directory'
#os.chdir(WDir)

#directory of OS Prediction files
fileDir = 'OSPrediction'

#data names
datalist = 'survival_data.csv'

df_file = 'features_df.pkl'

csv_file = 'osp_features.csv'

reference_img = '/Users/Sam/Uni/FYP/MNITemplate-master/inst/extdata/MNI152_T1_1mm_Brain.nii.gz'

#_________________ Feature Extraction Parameter Setup __________________#

EndSample = 260 #last item of interest in sample list
numchan = 4 #Number of channels

#feature extraction parameter file names
#Pyradiomics uses the yaml file format for receiving parameter lists
GFeatParamName = 'OSPFeatParamGrey.txt'
GFeatYamlName = 'GreyFeatParamsYaml.yaml'
SFeatParamName = 'OSPFeatParamShape.txt'
SFeatYamlName = 'ShapeFeatParamsYaml.yaml'

# prepare pyradiomics parameter files
Gfeatparamstxt = open(os.path.join(WDir,fileDir,GFeatParamName),'r')
GfeatparamsYaml = open(os.path.join(WDir,fileDir,GFeatYamlName),'w')
yaml.dump(yaml.safe_load(Gfeatparamstxt.read()),GfeatparamsYaml)

Sfeatparamstxt = open(os.path.join(WDir,fileDir,SFeatParamName),'r')
SfeatparamsYaml = open(os.path.join(WDir,fileDir,SFeatYamlName),'w')
yaml.dump(yaml.safe_load(Sfeatparamstxt.read()),SfeatparamsYaml)

#___________________Grid Search Parameter Setup_________________________#

report = False #toggle the training reports, for debugging

use_hpfeat = False #Toggle whether to use hand picked or generated features

split = 0.1 #train/test split

k = 10 # number of k-folds

feat_num = 10 #number of best features used

#neutral parameter set for the feature selection model
params1 = {'colsample_bytree': 1,
           'gamma': 0,
           'lambda': 1,
           'alpha': 0,
           'eta': 0.05,
           'max_depth': 3,
           'min_child_weight': 1.5,
           'n_estimators': 1000,
           'reg_alpha': 0.75,
           'reg_lambda': 0.45,
           'subsample': 1,
           'max_delta_step': 0,
           'tree_method': 'exact',
           'objective': 'reg:squarederror',
           'base_score': 0.5,
           'eval_metric': 'rmse',
           'seed': 66}

#Parameter set derived from the grid search, with some manual tuning
params_cv1 = {
           'gamma': 0.12,
           'lambda': 4,
           'alpha': 0.15,
           'eta': 0.2,
           'max_depth': 2,
           'min_child_weight': 6,
           'colsample_bytree': 0.4,
           'subsample': 0.2,
           'objective': 'reg:squarederror',
           'base_score': 0.5,
           'eval_metric': 'rmse'}

#Define the swept values in the grid earch
params_gs = {
           'gamma': [0, 0.04, 0.08, 0.12],
           'lambda': [1, 1.5, 2, 3],
           'alpha': [0, 0.5, 1],
           'eta': [0.05, 0.1, 0.2, 0.3],
           'max_depth': [2, 3, 5, 8, 13],
           'min_child_weight': [1, 2, 3, 5, 7],
           'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1],
           'subsample': [0.2, 0.4, 0.6, 0.8, 1],
           'objective': 'reg:squarederror',
           'base_score': 0.5,
           'eval_metric': 'rmse'}

params_file = 'best_parameters.csv'

#__Hand Picked Features___#

hp_feats = ['age','original_shape_SurfaceArea', 'original_shape_Maximum3DDiameter', \
            'WT_centroid_x', 'WT_centroid_y', 'WT_centroid_z',\
            'original_firstorder_90Percentile', \
            'NCR_norm_vol', 'WT_norm_vol']


#_____________________________________________________________________#


#____________________Define Statistical Functions_____________________#

#______________________RMSE___________________#

def rmse(preds, targets):
    return np.sqrt(((preds - targets) ** 2).mean())
#_____________________________________________#


#_______________Categorise output_____________#

def days_to_category(y):
    a = 300
    b = 450
    j = 0
    categorised = []
    for i in y:
        if i < a:
            categorised.append(0)
        elif a <= i and i < b:
            categorised.append(1)
        elif b < i:
            categorised.append(2)
        j += 1
    return categorised

#____________________________________________#


#___________________Acuracy__________________#
def accuracy_test(y_test, y_train):
    hit = 0
    for i in range(len(y_test)):
        if y_test[i] - y_train[i] == 0:
            hit += 1 
    return (hit/len(y_test))*100
#___________________________________________#

#_______________Z-Sore Norm_________________#

def z_norm(data):
    return (data - sum(data)/len(data)) / np.std(data)

def reverse_znorm(new_data, o_data):
    return new_data * np.std(o_data) + sum(o_data)/len(o_data)

#__________________________________________#

#______________max Min Norm________________#
def mm_norm(data):
    return (data - min(data))/(max(data) - min(data))

def reverse_mmnorm(new_data, o_data):
    return new_data * (max(o_data) - min(o_data)) + min(o_data)


#_________________Begin Feature Extraction_____________________________#


#_________________ Feature Extraction initialisation __________________#



#initialise extractors. Grey extractor generates the first order
#grey-level features. Shape extractor fits a mesh to the tumor mask to
#calculate surface area, and fits an ellipsoid to the mask to guage it's
#geometry. 
GreyExtractor = featureextractor.RadiomicsFeatureExtractor(GFeatYamlName)
ShapeExtractor = featureextractor.RadiomicsFeatureExtractor(SFeatYamlName)

#channel type lookup
channel = ['flair', 't1', 't1ce', 't2', 'seg']

#Open survival data
with open(os.path.join(WDir,fileDir, datalist), 'r') as f:
            reader = csv.reader(f)
            svl_data = np.asarray(list(reader))



#___________________define centroid function____________________________#


def centroid_features(moving_img, reference_img, mask):

    #The mask is first registered to the reference image, such as MNI152.
    #A grey data image is needed to find the transform matrix. This is the
    #moving image.
    #Only rotation, translation and cropping is permitted in this transformation,
    #as we are only adjsuting for patient head alignment. 
    
    moving_img = sitk.ReadImage(moving_img, sitk.sitkFloat32)
    reference_img = sitk.ReadImage(reference_img, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask)

    #set up the types of transformations to use. This is where Euler transform is
    #specified, and the reference and "moving" iage are input.
    #NOTE the "moving image" is the patient T1 image, but this will not actually
    #be transformed. We are only interested in retreiving the trainsform
    #matrix so that it can be applied to the corresponding tumor mask. 
    initial_transform = sitk.CenteredTransformInitializer(reference_img,
                                                          moving_img,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Optimizer settings. These have been left at default. 
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, 
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        
    # Similarity metric settings. These have also been left at default. 
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    #This line yeilds the transform matrix
    final_transform = registration_method.Execute(reference_img, moving_img)

    #this line yeilds the transformed mask
    mask_resampled = sitk.Resample(mask, reference_img, final_transform, sitk.sitkLinear, 0.0, \
                                     mask.GetPixelID())

    #convert the mask image to numpy array to measure centroids
    data = sitk.GetArrayFromImage(mask_resampled)
    dims = data.shape

    WT_x_sum = 0
    WT_y_sum = 0
    WT_z_sum = 0
    WT_sum = 0

    NCR_x_sum = 0
    NCR_y_sum = 0
    NCR_z_sum = 0
    NCR_sum = 0

    ED_x_sum = 0
    ED_y_sum = 0
    ED_z_sum = 0
    ED_sum = 0

    ET_x_sum = 0
    ET_y_sum = 0
    ET_z_sum = 0
    ET_sum = 0


    #the following set of loops test each voxel and update the corresponding
    #mass sums for the respective tumor region, weighted by distance from the origin. 
    for x in range(0, dims[2]):
        for y in range(0, dims[1]):
            for z in range(0,dims[0]):

                if data[z,y,x]  != 0:
                    
                    WT_x_sum += x
                    WT_y_sum += y
                    WT_z_sum += z
                    WT_sum += 1

                    if data[z,y,x] == 1:
                        NCR_x_sum += x
                        NCR_y_sum += y
                        NCR_z_sum += z
                        NCR_sum += 1

                    elif data[z,y,x] == 2:
                        ED_x_sum += x
                        ED_y_sum += y
                        ED_z_sum += z
                        ED_sum += 1

                    elif data[z,y,x] == 4:
                        ET_x_sum += x
                        ET_y_sum += y
                        ET_z_sum += z
                        ET_sum += 1

    mask_v = dims[0] * dims[1] * dims[2]

    #Centroid location is found by dividing the weighted mass
    #sums by the region total mass
    
    WT_X = WT_x_sum/WT_sum
    WT_Y = WT_y_sum/WT_sum
    WT_Z = WT_z_sum/WT_sum
    WT_norm = WT_sum / (mask_v)

    NCR_X = NCR_x_sum/NCR_sum
    NCR_Y = NCR_y_sum/NCR_sum
    NCR_Z = NCR_z_sum/NCR_sum
    NCR_norm = NCR_sum / (mask_v)

    ED_X = ED_x_sum/ED_sum
    ED_Y = ED_y_sum/ED_sum
    ED_Z = ED_z_sum/ED_sum
    ED_norm = ED_sum / (mask_v)

    ET_X = ET_x_sum/ET_sum
    ET_Y = ET_y_sum/ET_sum
    ET_Z = ET_z_sum/ET_sum
    ET_norm = ET_sum / (mask_v)


    features = OrderedDict({'WT_centroid_x': WT_X,\
                            'WT_centroid_y': WT_Y,\
                            'WT_centroid_z': WT_Z,\
                            'WT_norm_vol': WT_norm,\
                            'NCR_centroid_x': NCR_X,\
                            'NCR_centroid_y': NCR_Y,\
                            'NCR_centroid_z': NCR_Z,\
                            'NCR_norm_vol': NCR_norm,\
                            'ED_centroid_x': ED_X,\
                            'ED_centroid_y': ED_Y,\
                            'ED_centroid_z': ED_Z,\
                            'ED_norm_vol': ED_norm,\
                            'ET_centroid_x': ET_X,\
                            'ET_centroid_y': ET_Y,\
                            'ET_centroid_z': ET_Z,\
                            'ET_norm_vol': ET_norm})

    return features


#_____________________ Extract features for Training ____________________#


j = 0 #0 corresponds to flair. 
DictList = []

#The feature extraction proces starts here. It loops over each patient, tests
#that the file exists, and then applies the greylevel and spatial featureextractors
#initialised earlier
for i in train_set[0:len(train_set)]:

    featDict = OrderedDict({})

    
    print('Getting features for patient ', i,', ', svl_data[i,0],'\n')

    #test that the age and survival data is present for this patient
    try:
        floattest = svl_data[i,1].astype(np.float)
        floattest = svl_data[i,2].astype(np.float)
    except:
        print('Patient ', i, ' was skipped. String in place of float. \n')
        continue

    featDict.update(OrderedDict({'Patient': svl_data[i,0]}))
    
    
    imageName = os.path.join(WDir, dataDir2, svl_data[i,0], svl_data[i,0] + \
                            '_' + channel[j] + '.nii.gz')
    
    maskName = os.path.join(WDir, dataDir2, svl_data[i,0], svl_data[i,0] + \
                        '_' + channel[4] + '.nii.gz')

    #test for file existence, if so proceed with extraction
    try: 
        shapefeat = ShapeExtractor.execute(imageName, maskName)
    except:
        print('Patient ', i, ' shape feature extraction failed. Patient Skipped.\n')
        continue

    featDict.update(shapefeat)

    #extract features for sample i, channel j
    try: 
        greyfeat = GreyExtractor.execute(imageName, maskName)
    except:
        print('Patient ', i, ' grey level feature extraction failed. Patient Skipped.\n')
        continue

    featDict.update(greyfeat)
        
    #Append line with centroid data
    T1img = os.path.join(WDir, dataDir2, svl_data[i,0], svl_data[i,0] + \
                            '_' + channel[1] + '.nii.gz')
    c_feat = centroid_features(T1img, reference_img, maskName)

    featDict.update(c_feat)
        
    #patient age from csv
    agefeat = OrderedDict({'age': svl_data[i,1]})

    featDict.update(agefeat)

    #GTR feature
    gtrfeat = OrderedDict({'GTR': (1 if svl_data[i,3] == 'GTR' else 0)})
    featDict.update(gtrfeat)
    
    #Survivalin days 
    survival = OrderedDict({'survival_days':svl_data[i,2]})

    featDict.update(survival)

    DictList.append(featDict)

df = pd.DataFrame(DictList)

df.to_pickle(df_file)

df.to_csv(csv_file)

#____________________End Feature Extraction_______________________________#

#____________________Begin Feature Selection______________________________#


#THe data is categorised by survival days, normalised and stratified,
#then split in to training and validation sets

csv_feats = 'osp_features.csv'

csv_ordered_feats = 'osp_feat_importance.csv'

dataset = pd.read_csv(csv_feats, header = 0)

dataset['cat_days'] = days_to_category(dataset['survival_days'])

feature_names = dataset.columns[25:119]

dataset_znorm = dataset.iloc[:, 25:121]

for k in dataset.columns[25:120]:
dataset_znorm[k] = z_norm(dataset_znorm[k])
dataset_0 = dataset_znorm[dataset_znorm['cat_days'] == 0]
dataset_1 = dataset_znorm[dataset_znorm['cat_days'] == 1]
dataset_2 = dataset_znorm[dataset_znorm['cat_days'] == 2]

acy = 0
rmse_av = 0
for t in range(0, reps):

    train_data_0, test_data_0 = train_test_split(dataset_0, test_size = split)
    train_data_1, test_data_1 = train_test_split(dataset_1, test_size = split)
    train_data_2, test_data_2 = train_test_split(dataset_2, test_size = split)

    train_data = pd.concat([train_data_0, train_data_1, train_data_2])
    test_data = pd.concat([test_data_0, test_data_1, test_data_2])

    train_feat_norm = train_data[feature_names]
    test_feat_norm = test_data[feature_names]
    train_y = train_data['survival_days']
    test_y = test_data['survival_days']


    #This is the general model that is fit with the full data set.
    #It is not a useful predictor. Instead it is used to rank feature importance
    #by occurrence in trees. 
    model = xgb.XGBRegressor(colsample_bytree = 0.4, gamma = 0,
                             learning_rate = 0.07, max_depth = 3,
                             min_child_weight = 1.5, n_estimators = 1000,
                             reg_alpha = 0.75, reg_lambda = 0.45,
                             seed = 42,
                             objective = 'reg:squarederror')

    model.fit(train_feat_norm, train_y)

    ordered_feats = pd.DataFrame(sorted(model.get_booster().get_fscore().items(), \
                                       key = lambda t: t[1], reverse = True))

    #save the feature ranknings and select the best. 
    ordered_feats.to_csv(csv_ordered_feats)

    best_feats = list(ordered_feats.iloc[[0, feat_num],0])

#____________________Begin Parameter Grid Search__________________________#

#______________________________stratify data______________________________#

#The cross validation XGB model needs to be passed lists of stratified
#data. If this step is skipped, the shuffled data may conceal representitive
#results due to the target class imbalance. 

ind_0 = train_data.index[train_data['cat_days'] == 0].tolist()
ind_1 = train_data.index[train_data['cat_days'] == 1].tolist()
ind_2 = train_data.index[train_data['cat_days'] == 2].tolist()

folds = []
for i in range(0, k):
    out_0 = ind_0[i::k]
    out_1 = ind_1[i::k]
    out_2 = ind_2[i::k]
    in_0 = [u for v, u in enumerate(ind_0) if u not in out_0]
    in_1 = [u for v, u in enumerate(ind_1) if u not in out_1]
    in_2 = [u for v, u in enumerate(ind_2) if u not in out_2]
    fold = (in_0 + in_1 + in_2, out_0 + out_1 + out_2)
    folds.append(fold)

#folds specifies the train and test sets allowed for each cross-validation


#_______________________________________________________________________#


#_______________________________CV Grid Search__________________________#

#the grid search is a brute force optimisation approach. Each parameter is
#stepped individually. The best performing set is saved. 

#initialise the data matrices for XGB
dtrain = xgb.DMatrix(train_feat_norm, train_y)
dtest = xgb.DMatrix(test_feat_norm, test_y)

best_result = 1000
start = time.time()

#loop over each parameter
for gam in params_gs['gamma']:
    for lam in params_gs['lambda']:
        for alph in params_gs['alpha']:
            for eta in params_gs['eta']:
                for md in params_gs['max_depth']:
                    for mcw in params_gs['min_child_weight']:
                        for cols in params_gs['colsample_bytree']:
                            for subs in params_gs['subsample']:
                                parameters = {'gamma': gam,
                                              'lambda': lam,
                                              'alpha': alph,
                                              'eta': eta,
                                              'max_depth': md,
                                              'min_child_weight': mcw,
                                              'colsample_bytree': cols,
                                              'subsample': subs,
                                              'objective': 'reg:squarederror',
                                              'base_score': 0.5,
                                              'eval_metric': 'rmse'}

                                cv_results = xgb.cv(params = parameters,
                                                    dtrain = dtrain,
                                                    folds = folds,
                                                    num_boost_round = 1000,
                                                    nfold = k,
                                                    early_stopping_rounds = 20)

                                if cv_results['test-rmse-mean'].min() < best_result:
                                    best_result = cv_results['test-rmse-mean'].min()
                                    best_params = {'gamma': gam,
                                                   'lambda': lam,
                                                   'alpha': alph,
                                                   'eta': eta,
                                                   'max_depth': md,
                                                   'min_child_weight': mcw,
                                                   'colsample_bytree': cols,
                                                   'subsample': subs,
                                                   'objective': 'reg:squarederror',
                                                   'base_score': 0.5,
                                                   'eval_metric': 'rmse'}

end = time.time()

print('Best rmse: ', best_result, '\n With Parameters \n', best_params, '\n')
print('Execution time: ', end - start, 's\n')

best_params_df = pd.DataFrame(best_params, index = [0])

best_params_df.to_csv(params_file)

#_______________________End Parameter Grid Search______________________________#

#____________________________Train final Model_________________________________#

csv_feats = 'osp_features.csv'
csv_ordered_feats = 'osp_feat_importance.csv'
ordered_feats = pd.read_csv(csv_ordered_feats, header = 0)
dataset = pd.read_csv(csv_feats, header = 0)

dataset['cat_days'] = days_to_category(dataset['survival_days'])
dataset['cat_days'] = dataset['cat_days']

if use_hpfeat == True:
    feature_names = hp_feats
else:
    feature_names = list(ordered_feats.iloc[[0, feat_num],0])

dataset_znorm = dataset.iloc[:, 25:121]

for i in dataset.columns[25:120]:
    dataset_znorm[i] = mm_norm(dataset_znorm[i])


#_______________________________Train final model______________________________#

dtrain = xgb.DMatrix(train_feat_norm, train_y)
dtest = xgb.DMatrix(test_feat_norm, test_y)

best_params = pd.read_csv(params_file)

best_params = best_params.to_dict()

cv_results = xgb.cv(params = best_params, dtrain = dtrain, folds = folds,
                    num_boost_round = 80, nfold = k,
                    early_stopping_rounds = 80, verbose_eval = 10)

#______________________________________________________________________________#

#__________________________Process and export results__________________________#

cv_results['train-rmse-mean'] = reverse_mmnorm(cv_results['train-rmse-mean'],
                                               dataset['survival_days'])
cv_results['train-rmse-std'] = reverse_mmnorm(cv_results['train-rmse-std'],
                                               dataset['survival_days'])
cv_results['test-rmse-mean'] = reverse_mmnorm(cv_results['test-rmse-mean'],
                                               dataset['survival_days'])
cv_results['test-rmse-std'] = reverse_mmnorm(cv_results['test-rmse-std'],
                                               dataset['survival_days'])

results_name = 'cv_results.csv'



cv_results.to_csv(results_name)



