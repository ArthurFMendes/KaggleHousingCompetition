#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:50:44 2018

@author: arthurmendes

directory:
    /Users/arthurmendes/Desktop/PythonCourse
Purpose:
    This is a attempt to create a predictive analysis to price houses in the
    city of Aimes/IO.

Bugs know:

"""


import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # more data visualization
import statsmodels.formula.api as smf # regression modeling


 # Set ipython's max row display
pd.set_option('display.max_row', 1000)

# Set iPython's max column width to 500
pd.set_option('display.max_columns', 500)

file = open('housing_train.csv')
housing = pd.read_csv(file)


# Viewing the first 5 rows of the dataset
housing.head()


# Viewing the last 5 rows of the dataset
housing.tail()

# Checkin the data types
housing.dtypes


###############################################################################
# Summarizing our dataset
###############################################################################

housing.shape


housing.info()


housing.describe()


housing.describe().round(2)


###############################################################################
# Misssing values
###############################################################################

# Check which observations are not missing
housing.count()

# Check which column has missing observations
print(housing.isnull().any())

# Counting the number of missing observations
print(housing.isnull().sum())
null_val = housing.isnull().sum()
#Copy the data frame (in case I mess up)
copy_housing  = pd.DataFrame.copy(housing)


"""

NOTE:
    This are the values that are nan are actually not a missing value, but are
    as bellow:
       THIS HAS NOT BEEN ADDED TO THE CODE YET

Alley:
NA - No alley access

BsmtCond:
NA - no basement

BsmtQual:
    NA - no basement

BsmtExposure:
NA - no basement

BsmtFinType1:
NA - no basement

BsmtFinType2:
NA - no basement

FireplaceQu:
NA - no fireplace

GarageType:
NA - no garage

GarageYrBlt:
    NA - no Garage

GarageFinish:
NA - no garage

GarageQual:
NA - no garage

GarageCond:
NA - no garage

PoolQC:
NA - no pool

Fence:
NA - no fence

MiscFeature:
NA - none

"""

# Alley access
copy_housing['Alley'] = copy_housing['Alley'].fillna('No_Acess')

# Basement condition
copy_housing['BsmtCond'] = copy_housing['BsmtCond'].fillna('NB')

# Basement Quality
copy_housing['BsmtQual'] = copy_housing['BsmtQual'].fillna('NB')

# Basement Exposure
copy_housing['BsmtExposure'] = copy_housing['BsmtExposure'].fillna('NB')

# Basement Exposure
copy_housing['BsmtFinType1'] = copy_housing['BsmtFinType1'].fillna('NB')

# Basement Exposure
copy_housing['BsmtFinType2'] = copy_housing['BsmtFinType2'].fillna('NB')

# Fireplace
copy_housing['FireplaceQu'] = copy_housing['FireplaceQu'].fillna('NF')

# Garage Type
copy_housing['GarageType'] = copy_housing['GarageType'].fillna('NG')

# Garage Type
copy_housing['GarageYrBlt'] = copy_housing['GarageYrBlt'].fillna('NG')

# GarageFinish
copy_housing['GarageFinish'] = copy_housing['GarageFinish'].fillna('NF')

# Garage Quality
copy_housing['GarageQual'] = copy_housing['GarageQual'].fillna('NF')

# Garage Condition
copy_housing['GarageCond'] = copy_housing['GarageCond'].fillna('NF')

# Pool QC
copy_housing['PoolQC'] = copy_housing['PoolQC'].fillna('NP')

# Fences
copy_housing['Fence'] = copy_housing['Fence'].fillna('NP')

# MiscFeatures
copy_housing['MiscFeature'] = copy_housing['MiscFeature'].fillna('NM')



# Counting the number of missing observations
print(copy_housing.isnull().sum())
null_val_2 = copy_housing.isnull().sum()

"""
LotFrontage is still missing 259 variables (DROP IT?)
Electrical is missing one variable (Flag it)

"""


###############################################################################
# Misssing values
###############################################################################


# Counting the number of missing observations
print(copy_housing.isnull().sum())

# Number of missing values
null_val_2 = copy_housing.isnull().sum()


# Flagging Missing values

for col in copy_housing:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if copy_housing[col].isnull().any():
        copy_housing['m_'+col] = copy_housing[col].isnull().astype(int)


###############################################################################
# Analyzing the missing values
###############################################################################


# Copy the housing data again, this time for droping observations

copy_housing_2  = pd.DataFrame.copy(copy_housing)




# Drop columns with more than 200 missing values  (MAYBE)
"""
for col in copy_housing_2:

    """ "Delete columns with over 200 missing values" """

    if copy_housing_2[col].isnull().sum() > 200:
        copy_housing_2 = copy_housing_2.drop(col, axis = 1)

null_val_2 = copy_housing_2.isnull().sum()

print(copy_housing_2.isnull().sum())

"""
# Check how would I want to fill
copy_housing_2['LotFrontage'].describe().round(2)

### Histogram
copy_housing_2['LotFrontage'].hist(bins = 'fd')

plt.xlabel("Lot Frontage")
plt.show()


# Creating tbe mean for LotFrontage inputation
LotFrt_mean = copy_housing_2['LotFrontage'].mean()

copy_housing_2['LotFrontage'] = (copy_housing_2['LotFrontage']
                            .fillna(LotFrt_mean)
                            .round(2))

# Electrical

copy_housing_2['Electrical'].nunique()

copy_housing_2['Electrical'].describe()

# The most common electrical system would be SBrkr

copy_housing_2['Electrical'] = (copy_housing_2['Electrical']
                            .fillna('SBrkr'))

# SOME ERRORS
"""
I want to change the values that are != 0 in MasVnrArea when MasVnrType is
equal to None

if copy_housing_2['MasVnrType'] == 'None':
    copy_housing_2['MasVnrArea'] = copy_housing_2['MasVnrArea'](0)

MasVnrType       1452 non-null category
MasVnrArea       1452 non-null float64
"""

copy_housing_2.loc[:, 'MasVnrArea'][copy_housing_2['MasVnrType'] == 'None'] = 0

###############################################################################
# Change variable types: categories would help me analyze the data further
###############################################################################


for col in copy_housing_2:

    """Change the variable type, from object to categories to facilitate
    calculatinons"""

    if copy_housing_2[col].dtypes == 'object':
        copy_housing_2[col] = copy_housing_2[col].astype('category')


housing_corr = copy_housing_2.corr().round(2)

#copy_housing_2.dtype

copy_housing_2.info()




###############################################################################
# Regressions & Predictions
###############################################################################



lm_full = smf.ols(formula = """
                  SalePrice  ~
                  C(MSSubClass) +
                  C(MSSubClass) +
                  C(MSZoning) +
                  C(Alley) +
                  LotArea +
                  C(Street) +
                  C(LotShape) +
                  C(LandContour) +
                  C(Utilities) +
                  C(LotConfig) +
                  C(LandSlope) +
                  C(Neighborhood) +
                  C(Condition1) +
                  C(Condition2) +
                  C(BldgType) +
                  C(HouseStyle) +
                  C(OverallQual) +
                  C(OverallCond) +
                  C(RoofMatl) +
                  C(YearBuilt) +
                  C(YearRemodAdd) +
                  C(RoofStyle ) +
                  C(Exterior1st) +
                  C(Exterior2nd) +
                  C(MasVnrType) +
                  MasVnrArea +
                  C(ExterQual) +
                  C(ExterCond) +
                  C(Foundation) +
                  C(BsmtQual) +
                  C(BsmtCond) +
                  C(BsmtExposure) +
                  C(BsmtFinType1) +
                  BsmtFinSF1 +
                  C(BsmtFinType2) +
                  BsmtFinSF2 +
                  BsmtUnfSF +
                  TotalBsmtSF +
                  C(PoolQC) +
                  C(Heating) +
                  C(HeatingQC) +
                  C(CentralAir) +
                  C(Electrical) +
                  C(Fence) +
                  LowQualFinSF +
                  GrLivArea +
                  BsmtFullBath +
                  BsmtHalfBath +
                  FullBath +
                  HalfBath +
                  BedroomAbvGr +
                  KitchenAbvGr +
                  C(KitchenQual) +
                  TotRmsAbvGrd +
                  C(Functional) +
                  C(MiscFeature) +
                  MiscVal +
                  Fireplaces +
                  C(GarageType) +
                  GarageYrBlt +
                  C(GarageFinish) +
                  GarageCars +
                  GarageArea +
                  C(GarageQual) +
                  C(GarageCond) +
                  C(PavedDrive) +
                  WoodDeckSF +
                  OpenPorchSF +
                  EnclosedPorch +
                  m_LotFrontage  +
                  m_MasVnrType +
                  m_MasVnrArea +
                  ScreenPorch +
                  PoolArea +
                  m_Electrical+
                  C(MoSold) +
                  C(YrSold) +
                  C(YrSold) +
                  C(SaleType) +
                  C(SaleCondition)
                  """ , data = copy_housing_2)


results = lm_full.fit()

print(results.summary())

print(results.rsquared)

###############################################################################
# Part 2
###############################################################################
file = open('housing_test.csv')
housing_test = pd.read_csv(file)


# Alley access
housing_test['Alley'] = housing_test['Alley'].fillna('No_Acess')

# Basement condition
housing_test['BsmtCond'] = housing_test['BsmtCond'].fillna('NB')

# Basement Quality
housing_test['BsmtQual'] = housing_test['BsmtQual'].fillna('NB')

# Basement Exposure
housing_test['BsmtExposure'] = housing_test['BsmtExposure'].fillna('NB')

# Basement Exposure
housing_test['BsmtFinType1'] = housing_test['BsmtFinType1'].fillna('NB')

# Basement Exposure
housing_test['BsmtFinType2'] = housing_test['BsmtFinType2'].fillna('NB')

# Fireplace
housing_test['FireplaceQu'] = housing_test['FireplaceQu'].fillna('NF')

# Garage Type
housing_test['GarageType'] = housing_test['GarageType'].fillna('NG')

# Garage Type
housing_test['GarageYrBlt'] = housing_test['GarageYrBlt'].fillna('NG')

# GarageFinish
housing_test['GarageFinish'] = housing_test['GarageFinish'].fillna('NF')

# Garage Quality
housing_test['GarageQual'] = housing_test['GarageQual'].fillna('NF')

# Garage Condition
housing_test['GarageCond'] = housing_test['GarageCond'].fillna('NF')

# Pool QC
housing_test['PoolQC'] = housing_test['PoolQC'].fillna('NP')

# Fences
housing_test['Fence'] = housing_test['Fence'].fillna('NP')

# MiscFeatures
housing_test['MiscFeature'] = housing_test['MiscFeature'].fillna('NM')

#ADD SALE PRICE
housing_test['SalePrice'] = 0


print(housing_test.isnull().sum())

# Lot frontage
LotFrt_mean_test = housing_test['LotFrontage'].mean()

housing_test['LotFrontage'] = (housing_test['LotFrontage']
                            .fillna(LotFrt_mean_test)
                            .round(2))

## MSZoning

housing_test['MSZoning'].describe()
housing_test['MSZoning'] = (housing_test['MSZoning']
                            .fillna('RL'))

# Utilities
housing_test['Utilities'].describe()
housing_test['Utilities'] = (housing_test['Utilities']
                            .fillna('AllPub'))

# Exterior1st
housing_test['Exterior1st'].describe()
housing_test['Exterior1st'] = (housing_test['Exterior1st']
                            .fillna('VinylSd'))

# Exterior2nd
housing_test['Exterior2nd'].describe()
housing_test['Exterior2nd'] = (housing_test['Exterior2nd']
                            .fillna('VinylSd'))


# MasVnrType
housing_test['MasVnrType'].describe()
housing_test['MasVnrType'] = (housing_test['MasVnrType']
                            .fillna('None'))

# MasVnrArea
housing_test['MasVnrArea'] = (housing_test['MasVnrArea'].fillna(0))
                            
# BsmtFinSF1
Bsm_sqf_mena = housing_test['BsmtFinSF1'].mean()

housing_test['BsmtFinSF1'] = (housing_test['BsmtFinSF1']
                            .fillna(Bsm_sqf_mena)
                            .round(2))
# BsmtFinSF2
housing_test['BsmtFinSF2'].describe()
housing_test['BsmtFinSF2'] = (housing_test['BsmtFinSF2']
                            .fillna(0))
                           
# BsmtUnfSF
housing_test['BsmtUnfSF'].describe()
Bsm_sqf_mean = housing_test['BsmtUnfSF'].mean()

housing_test['BsmtUnfSF'] = (housing_test['BsmtUnfSF']
                            .fillna(Bsm_sqf_mean)
                            .round(2))
# TotalBsmtSF

housing_test['TotalBsmtSF'].describe()
ttl_Bsm_sqf_mean = housing_test['TotalBsmtSF'].mean()

housing_test['TotalBsmtSF'] = (housing_test['TotalBsmtSF']
                            .fillna(Bsm_sqf_mean)
                            .round(2))
# BsmtFullBath
housing_test['BsmtFullBath'].describe()
housing_test['BsmtFullBath'] = (housing_test['BsmtFullBath']
                            .fillna(0))

# BsmtHalfBath
housing_test['BsmtHalfBath'].describe()
housing_test['BsmtHalfBath'] = (housing_test['BsmtHalfBath']
                            .fillna(0))

# KitchenQual
housing_test['KitchenQual'].describe()
housing_test['KitchenQual'] = (housing_test['KitchenQual']
                            .fillna('TA'))

# Functional
housing_test['Functional'].describe()
housing_test['Functional'] = (housing_test['Functional']
                            .fillna('Typ'))

# GarageCars
housing_test['GarageCars'].describe()
housing_test['GarageCars'] = (housing_test['GarageCars']
                            .fillna(2))

# GarageArea
housing_test['GarageArea'].describe()
grn_area = housing_test['GarageArea'].mean()

housing_test['GarageArea'] = (housing_test['GarageArea']
                            .fillna(grn_area)
                            .round(2))

# SaleType
housing_test['SaleType'].describe()
housing_test['SaleType'] = (housing_test['SaleType']
                            .fillna('WD'))

"""
Id                 0
MSSubClass         0
MSZoning           4
LotFrontage      227
LotArea            0
Street             0
Alley              0
LotShape           0
LandContour        0
Utilities          2
LotConfig          0
LandSlope          0
Neighborhood       0
Condition1         0
Condition2         0
BldgType           0
HouseStyle         0
OverallQual        0
OverallCond        0
YearBuilt          0
YearRemodAdd       0
RoofStyle          0
RoofMatl           0
Exterior1st        1
Exterior2nd        1
MasVnrType        16
MasVnrArea        15
ExterQual          0
ExterCond          0
Foundation         0
BsmtQual           0
BsmtCond           0
BsmtExposure       0
BsmtFinType1       0
BsmtFinSF1         1
BsmtFinType2       0
BsmtFinSF2         1
BsmtUnfSF          1
TotalBsmtSF        1
Heating            0
HeatingQC          0
CentralAir         0
Electrical         0
1stFlrSF           0
2ndFlrSF           0
LowQualFinSF       0
GrLivArea          0
BsmtFullBath       2
BsmtHalfBath       2
FullBath           0
HalfBath           0
BedroomAbvGr       0
KitchenAbvGr       0
KitchenQual        1
TotRmsAbvGrd       0
Functional         2
Fireplaces         0
FireplaceQu        0
GarageType         0
GarageYrBlt        0
GarageFinish       0
GarageCars         1
GarageArea         1
GarageQual         0
GarageCond         0
PavedDrive         0
WoodDeckSF         0
OpenPorchSF        0
EnclosedPorch      0
3SsnPorch          0
ScreenPorch        0
PoolArea           0
PoolQC             0
Fence              0
MiscFeature        0
MiscVal            0
MoSold             0
YrSold             0
SaleType           1
SaleCondition      0
SalePrice          0
"""





for col in housing_test:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if housing_test[col].isnull().any():
        housing_test['m_'+col] = housing_test[col].isnull().astype(int)
        

###
        
housing_test.loc[:, 'MasVnrArea'][housing_test['MasVnrType'] == 'None'] = 0

print(housing_test.isnull().sum())


for col in housing_test:
    
    """Change the variable type, from object to categories to facilitate 
    calculatinons"""
    
    if housing_test[col].dtypes == 'object':
        housing_test[col] = housing_test[col].astype('category')
    

lm_full.fit('copy_housing_2','housing_test')

housing_test.info()

predict = results.predict(housing_test)

print(predict.fittedvalues)

print(results.fittedvalues)

## Change category

housing_test['Id'] = housing_test['Id'].astype('category')

housing_test['MSSubClass'] = housing_test['MSSubClass'].astype('category')

housing_test['OverallQual'] = housing_test['OverallQual'].astype('int')

housing_test['OverallCond'] = housing_test['OverallCond'].astype('int')

housing_test['YearBuilt'] = housing_test['YearBuilt'].astype('category')

housing_test['YearRemodAdd'] = housing_test['YearRemodAdd'].astype('category')

housing_test['MoSold'] = housing_test['MoSold'].astype('category')

housing_test['YrSold'] = housing_test['YrSold'].astype('category')



"""
Id               1459 non-null int64
MSSubClass       1459 non-null int64
OverallQual      1459 non-null int64
OverallCond      1459 non-null int64
YearBuilt        1459 non-null int64
YearRemodAdd     1459 non-null int64
MoSold           1459 non-null int64
YrSold           1459 non-null int64
SalePrice        1459 non-null int64
"""


