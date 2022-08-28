{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1c325d5d",
   "metadata": {},
   "source": [
    "# Real estate - price predictor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7977345c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "fb0a3b42",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing=pd.read_csv(\"data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a3ae1eb0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.00632</td>\n",
       "      <td>18.0</td>\n",
       "      <td>2.31</td>\n",
       "      <td>0</td>\n",
       "      <td>0.538</td>\n",
       "      <td>6.575</td>\n",
       "      <td>65.2</td>\n",
       "      <td>4.0900</td>\n",
       "      <td>1</td>\n",
       "      <td>296</td>\n",
       "      <td>15.3</td>\n",
       "      <td>396.90</td>\n",
       "      <td>4.98</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.02731</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>6.421</td>\n",
       "      <td>78.9</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2</td>\n",
       "      <td>242</td>\n",
       "      <td>17.8</td>\n",
       "      <td>396.90</td>\n",
       "      <td>9.14</td>\n",
       "      <td>21.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.02729</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.07</td>\n",
       "      <td>0</td>\n",
       "      <td>0.469</td>\n",
       "      <td>7.185</td>\n",
       "      <td>61.1</td>\n",
       "      <td>4.9671</td>\n",
       "      <td>2</td>\n",
       "      <td>242</td>\n",
       "      <td>17.8</td>\n",
       "      <td>392.83</td>\n",
       "      <td>4.03</td>\n",
       "      <td>34.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.03237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>6.998</td>\n",
       "      <td>45.8</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3</td>\n",
       "      <td>222</td>\n",
       "      <td>18.7</td>\n",
       "      <td>394.63</td>\n",
       "      <td>2.94</td>\n",
       "      <td>33.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.06905</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.18</td>\n",
       "      <td>0</td>\n",
       "      <td>0.458</td>\n",
       "      <td>7.147</td>\n",
       "      <td>54.2</td>\n",
       "      <td>6.0622</td>\n",
       "      <td>3</td>\n",
       "      <td>222</td>\n",
       "      <td>18.7</td>\n",
       "      <td>396.90</td>\n",
       "      <td>5.33</td>\n",
       "      <td>36.2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO  \\\n",
       "0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3   \n",
       "1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8   \n",
       "2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8   \n",
       "3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7   \n",
       "4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7   \n",
       "\n",
       "        B  LSTAT  MEDV  \n",
       "0  396.90   4.98  24.0  \n",
       "1  396.90   9.14  21.6  \n",
       "2  392.83   4.03  34.7  \n",
       "3  394.63   2.94  33.4  \n",
       "4  396.90   5.33  36.2  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4698de78",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 506 entries, 0 to 505\n",
      "Data columns (total 14 columns):\n",
      " #   Column   Non-Null Count  Dtype  \n",
      "---  ------   --------------  -----  \n",
      " 0   CRIM     506 non-null    float64\n",
      " 1   ZN       506 non-null    float64\n",
      " 2   INDUS    506 non-null    float64\n",
      " 3   CHAS     506 non-null    int64  \n",
      " 4   NOX      506 non-null    float64\n",
      " 5   RM       506 non-null    float64\n",
      " 6   AGE      506 non-null    float64\n",
      " 7   DIS      506 non-null    float64\n",
      " 8   RAD      506 non-null    int64  \n",
      " 9   TAX      506 non-null    int64  \n",
      " 10  PTRATIO  506 non-null    float64\n",
      " 11  B        506 non-null    float64\n",
      " 12  LSTAT    506 non-null    float64\n",
      " 13  MEDV     506 non-null    float64\n",
      "dtypes: float64(11), int64(3)\n",
      "memory usage: 55.5 KB\n"
     ]
    }
   ],
   "source": [
    "housing.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5ca2cc77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    471\n",
       "1     35\n",
       "Name: CHAS, dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing[\"CHAS\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5b12ed63",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "      <td>506.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>3.613524</td>\n",
       "      <td>11.363636</td>\n",
       "      <td>11.136779</td>\n",
       "      <td>0.069170</td>\n",
       "      <td>0.554695</td>\n",
       "      <td>6.284634</td>\n",
       "      <td>68.574901</td>\n",
       "      <td>3.795043</td>\n",
       "      <td>9.549407</td>\n",
       "      <td>408.237154</td>\n",
       "      <td>18.455534</td>\n",
       "      <td>356.674032</td>\n",
       "      <td>12.653063</td>\n",
       "      <td>22.532806</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>8.601545</td>\n",
       "      <td>23.322453</td>\n",
       "      <td>6.860353</td>\n",
       "      <td>0.253994</td>\n",
       "      <td>0.115878</td>\n",
       "      <td>0.702617</td>\n",
       "      <td>28.148861</td>\n",
       "      <td>2.105710</td>\n",
       "      <td>8.707259</td>\n",
       "      <td>168.537116</td>\n",
       "      <td>2.164946</td>\n",
       "      <td>91.294864</td>\n",
       "      <td>7.141062</td>\n",
       "      <td>9.197104</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.006320</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.460000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.385000</td>\n",
       "      <td>3.561000</td>\n",
       "      <td>2.900000</td>\n",
       "      <td>1.129600</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>187.000000</td>\n",
       "      <td>12.600000</td>\n",
       "      <td>0.320000</td>\n",
       "      <td>1.730000</td>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.082045</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.190000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.449000</td>\n",
       "      <td>5.885500</td>\n",
       "      <td>45.025000</td>\n",
       "      <td>2.100175</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>279.000000</td>\n",
       "      <td>17.400000</td>\n",
       "      <td>375.377500</td>\n",
       "      <td>6.950000</td>\n",
       "      <td>17.025000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.256510</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.690000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.538000</td>\n",
       "      <td>6.208500</td>\n",
       "      <td>77.500000</td>\n",
       "      <td>3.207450</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>330.000000</td>\n",
       "      <td>19.050000</td>\n",
       "      <td>391.440000</td>\n",
       "      <td>11.360000</td>\n",
       "      <td>21.200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>3.677083</td>\n",
       "      <td>12.500000</td>\n",
       "      <td>18.100000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.624000</td>\n",
       "      <td>6.623500</td>\n",
       "      <td>94.075000</td>\n",
       "      <td>5.188425</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>666.000000</td>\n",
       "      <td>20.200000</td>\n",
       "      <td>396.225000</td>\n",
       "      <td>16.955000</td>\n",
       "      <td>25.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>88.976200</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>27.740000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.871000</td>\n",
       "      <td>8.780000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>12.126500</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>711.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>396.900000</td>\n",
       "      <td>37.970000</td>\n",
       "      <td>50.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             CRIM          ZN       INDUS        CHAS         NOX          RM  \\\n",
       "count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   \n",
       "mean     3.613524   11.363636   11.136779    0.069170    0.554695    6.284634   \n",
       "std      8.601545   23.322453    6.860353    0.253994    0.115878    0.702617   \n",
       "min      0.006320    0.000000    0.460000    0.000000    0.385000    3.561000   \n",
       "25%      0.082045    0.000000    5.190000    0.000000    0.449000    5.885500   \n",
       "50%      0.256510    0.000000    9.690000    0.000000    0.538000    6.208500   \n",
       "75%      3.677083   12.500000   18.100000    0.000000    0.624000    6.623500   \n",
       "max     88.976200  100.000000   27.740000    1.000000    0.871000    8.780000   \n",
       "\n",
       "              AGE         DIS         RAD         TAX     PTRATIO           B  \\\n",
       "count  506.000000  506.000000  506.000000  506.000000  506.000000  506.000000   \n",
       "mean    68.574901    3.795043    9.549407  408.237154   18.455534  356.674032   \n",
       "std     28.148861    2.105710    8.707259  168.537116    2.164946   91.294864   \n",
       "min      2.900000    1.129600    1.000000  187.000000   12.600000    0.320000   \n",
       "25%     45.025000    2.100175    4.000000  279.000000   17.400000  375.377500   \n",
       "50%     77.500000    3.207450    5.000000  330.000000   19.050000  391.440000   \n",
       "75%     94.075000    5.188425   24.000000  666.000000   20.200000  396.225000   \n",
       "max    100.000000   12.126500   24.000000  711.000000   22.000000  396.900000   \n",
       "\n",
       "            LSTAT        MEDV  \n",
       "count  506.000000  506.000000  \n",
       "mean    12.653063   22.532806  \n",
       "std      7.141062    9.197104  \n",
       "min      1.730000    5.000000  \n",
       "25%      6.950000   17.025000  \n",
       "50%     11.360000   21.200000  \n",
       "75%     16.955000   25.000000  \n",
       "max     37.970000   50.000000  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d33bb45a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0d7625fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[<AxesSubplot:title={'center':'CRIM'}>,\n",
       "        <AxesSubplot:title={'center':'ZN'}>,\n",
       "        <AxesSubplot:title={'center':'INDUS'}>,\n",
       "        <AxesSubplot:title={'center':'CHAS'}>],\n",
       "       [<AxesSubplot:title={'center':'NOX'}>,\n",
       "        <AxesSubplot:title={'center':'RM'}>,\n",
       "        <AxesSubplot:title={'center':'AGE'}>,\n",
       "        <AxesSubplot:title={'center':'DIS'}>],\n",
       "       [<AxesSubplot:title={'center':'RAD'}>,\n",
       "        <AxesSubplot:title={'center':'TAX'}>,\n",
       "        <AxesSubplot:title={'center':'PTRATIO'}>,\n",
       "        <AxesSubplot:title={'center':'B'}>],\n",
       "       [<AxesSubplot:title={'center':'LSTAT'}>,\n",
       "        <AxesSubplot:title={'center':'MEDV'}>, <AxesSubplot:>,\n",
       "        <AxesSubplot:>]], dtype=object)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x1080 with 16 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "housing.hist(bins=50,figsize=(20,15))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "682db26c",
   "metadata": {},
   "source": [
    "# TRAIN - TEST SPLITTING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d17abe4a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "dae4b3e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "#def split_train_test(data,test_ratio):\n",
    "#    np.random.seed(42)\n",
    "#    shuffled=np.random.permutation(len(data))\n",
    "#   print(shuffled)\n",
    "#   test_set_size=int(len(data)*test_ratio)\n",
    "#   test_indices= shuffled[:test_set_size]\n",
    "#   train_indices= shuffled[test_set_size:]\n",
    "#   return data.iloc[train_indices], data.iloc[test_indices]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4a88a388",
   "metadata": {},
   "outputs": [],
   "source": [
    "#from sklearn.model_selection import train_test_split\n",
    "\n",
    "#train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e66f1e93",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedShuffleSplit\n",
    "a=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)\n",
    "for train_index,test_index in a.split(housing,housing['CHAS']):\n",
    "    strat_train_set=housing.loc[train_index]\n",
    "    strat_test_set=housing.loc[test_index]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "63aa418a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    95\n",
       "1     7\n",
       "Name: CHAS, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "strat_test_set['CHAS'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7e277bab",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing=strat_train_set.copy()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9767c252",
   "metadata": {},
   "source": [
    "# Looking for correlation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "abd51a36",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MEDV       1.000000\n",
      "RM         0.679894\n",
      "B          0.361761\n",
      "ZN         0.339741\n",
      "DIS        0.240451\n",
      "CHAS       0.205066\n",
      "AGE       -0.364596\n",
      "RAD       -0.374693\n",
      "CRIM      -0.393715\n",
      "NOX       -0.422873\n",
      "TAX       -0.456657\n",
      "INDUS     -0.473516\n",
      "PTRATIO   -0.493534\n",
      "LSTAT     -0.740494\n",
      "Name: MEDV, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "corr_matrix=housing.corr()\n",
    "print(corr_matrix['MEDV'].sort_values(ascending=False))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8473e0d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='RM', ylabel='MEDV'>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from pandas.plotting import scatter_matrix\n",
    "housing.plot(kind=\"scatter\",x=\"RM\",y=\"MEDV\",alpha=0.8)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6941434",
   "metadata": {},
   "source": [
    "##Trying out attributes combination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "24b837a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>MEDV</th>\n",
       "      <th>TAXRM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>254</th>\n",
       "      <td>0.04819</td>\n",
       "      <td>80.0</td>\n",
       "      <td>3.64</td>\n",
       "      <td>0</td>\n",
       "      <td>0.392</td>\n",
       "      <td>6.108</td>\n",
       "      <td>32.0</td>\n",
       "      <td>9.2203</td>\n",
       "      <td>1</td>\n",
       "      <td>315</td>\n",
       "      <td>16.4</td>\n",
       "      <td>392.89</td>\n",
       "      <td>6.57</td>\n",
       "      <td>21.9</td>\n",
       "      <td>51.571709</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>348</th>\n",
       "      <td>0.01501</td>\n",
       "      <td>80.0</td>\n",
       "      <td>2.01</td>\n",
       "      <td>0</td>\n",
       "      <td>0.435</td>\n",
       "      <td>6.635</td>\n",
       "      <td>29.7</td>\n",
       "      <td>8.3440</td>\n",
       "      <td>4</td>\n",
       "      <td>280</td>\n",
       "      <td>17.0</td>\n",
       "      <td>390.94</td>\n",
       "      <td>5.99</td>\n",
       "      <td>24.5</td>\n",
       "      <td>42.200452</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>476</th>\n",
       "      <td>4.87141</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.614</td>\n",
       "      <td>6.484</td>\n",
       "      <td>93.6</td>\n",
       "      <td>2.3053</td>\n",
       "      <td>24</td>\n",
       "      <td>666</td>\n",
       "      <td>20.2</td>\n",
       "      <td>396.21</td>\n",
       "      <td>18.68</td>\n",
       "      <td>16.7</td>\n",
       "      <td>102.714374</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>321</th>\n",
       "      <td>0.18159</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.38</td>\n",
       "      <td>0</td>\n",
       "      <td>0.493</td>\n",
       "      <td>6.376</td>\n",
       "      <td>54.3</td>\n",
       "      <td>4.5404</td>\n",
       "      <td>5</td>\n",
       "      <td>287</td>\n",
       "      <td>19.6</td>\n",
       "      <td>396.90</td>\n",
       "      <td>6.87</td>\n",
       "      <td>23.1</td>\n",
       "      <td>45.012547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>326</th>\n",
       "      <td>0.30347</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.38</td>\n",
       "      <td>0</td>\n",
       "      <td>0.493</td>\n",
       "      <td>6.312</td>\n",
       "      <td>28.9</td>\n",
       "      <td>5.4159</td>\n",
       "      <td>5</td>\n",
       "      <td>287</td>\n",
       "      <td>19.6</td>\n",
       "      <td>396.90</td>\n",
       "      <td>6.15</td>\n",
       "      <td>23.0</td>\n",
       "      <td>45.468948</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  \\\n",
       "254  0.04819  80.0   3.64     0  0.392  6.108  32.0  9.2203    1  315   \n",
       "348  0.01501  80.0   2.01     0  0.435  6.635  29.7  8.3440    4  280   \n",
       "476  4.87141   0.0  18.10     0  0.614  6.484  93.6  2.3053   24  666   \n",
       "321  0.18159   0.0   7.38     0  0.493  6.376  54.3  4.5404    5  287   \n",
       "326  0.30347   0.0   7.38     0  0.493  6.312  28.9  5.4159    5  287   \n",
       "\n",
       "     PTRATIO       B  LSTAT  MEDV       TAXRM  \n",
       "254     16.4  392.89   6.57  21.9   51.571709  \n",
       "348     17.0  390.94   5.99  24.5   42.200452  \n",
       "476     20.2  396.21  18.68  16.7  102.714374  \n",
       "321     19.6  396.90   6.87  23.1   45.012547  \n",
       "326     19.6  396.90   6.15  23.0   45.468948  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing[\"TAXRM\"]=housing[\"TAX\"]/housing[\"RM\"]\n",
    "housing.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82042d5b",
   "metadata": {},
   "source": [
    "# Missing attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "56983416",
   "metadata": {},
   "outputs": [],
   "source": [
    "# housing.dropna(subset=['RM']).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b9752d71",
   "metadata": {},
   "outputs": [],
   "source": [
    "#housing.drop('RM',axis=1).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "3d2db633",
   "metadata": {},
   "outputs": [],
   "source": [
    "#median=housing[\"RM\"].median()\n",
    "#housing[\"RM\"].fillna(median)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b014f4d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(404, 15)"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "84127e4b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CRIM</th>\n",
       "      <th>ZN</th>\n",
       "      <th>INDUS</th>\n",
       "      <th>CHAS</th>\n",
       "      <th>NOX</th>\n",
       "      <th>RM</th>\n",
       "      <th>AGE</th>\n",
       "      <th>DIS</th>\n",
       "      <th>RAD</th>\n",
       "      <th>TAX</th>\n",
       "      <th>PTRATIO</th>\n",
       "      <th>B</th>\n",
       "      <th>LSTAT</th>\n",
       "      <th>TAXRM</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>254</th>\n",
       "      <td>0.04819</td>\n",
       "      <td>80.0</td>\n",
       "      <td>3.64</td>\n",
       "      <td>0</td>\n",
       "      <td>0.392</td>\n",
       "      <td>6.108</td>\n",
       "      <td>32.0</td>\n",
       "      <td>9.2203</td>\n",
       "      <td>1</td>\n",
       "      <td>315</td>\n",
       "      <td>16.4</td>\n",
       "      <td>392.89</td>\n",
       "      <td>6.57</td>\n",
       "      <td>51.571709</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>348</th>\n",
       "      <td>0.01501</td>\n",
       "      <td>80.0</td>\n",
       "      <td>2.01</td>\n",
       "      <td>0</td>\n",
       "      <td>0.435</td>\n",
       "      <td>6.635</td>\n",
       "      <td>29.7</td>\n",
       "      <td>8.3440</td>\n",
       "      <td>4</td>\n",
       "      <td>280</td>\n",
       "      <td>17.0</td>\n",
       "      <td>390.94</td>\n",
       "      <td>5.99</td>\n",
       "      <td>42.200452</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>476</th>\n",
       "      <td>4.87141</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.614</td>\n",
       "      <td>6.484</td>\n",
       "      <td>93.6</td>\n",
       "      <td>2.3053</td>\n",
       "      <td>24</td>\n",
       "      <td>666</td>\n",
       "      <td>20.2</td>\n",
       "      <td>396.21</td>\n",
       "      <td>18.68</td>\n",
       "      <td>102.714374</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>321</th>\n",
       "      <td>0.18159</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.38</td>\n",
       "      <td>0</td>\n",
       "      <td>0.493</td>\n",
       "      <td>6.376</td>\n",
       "      <td>54.3</td>\n",
       "      <td>4.5404</td>\n",
       "      <td>5</td>\n",
       "      <td>287</td>\n",
       "      <td>19.6</td>\n",
       "      <td>396.90</td>\n",
       "      <td>6.87</td>\n",
       "      <td>45.012547</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>326</th>\n",
       "      <td>0.30347</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.38</td>\n",
       "      <td>0</td>\n",
       "      <td>0.493</td>\n",
       "      <td>6.312</td>\n",
       "      <td>28.9</td>\n",
       "      <td>5.4159</td>\n",
       "      <td>5</td>\n",
       "      <td>287</td>\n",
       "      <td>19.6</td>\n",
       "      <td>396.90</td>\n",
       "      <td>6.15</td>\n",
       "      <td>45.468948</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>155</th>\n",
       "      <td>3.53501</td>\n",
       "      <td>0.0</td>\n",
       "      <td>19.58</td>\n",
       "      <td>1</td>\n",
       "      <td>0.871</td>\n",
       "      <td>6.152</td>\n",
       "      <td>82.6</td>\n",
       "      <td>1.7455</td>\n",
       "      <td>5</td>\n",
       "      <td>403</td>\n",
       "      <td>14.7</td>\n",
       "      <td>88.01</td>\n",
       "      <td>15.02</td>\n",
       "      <td>65.507152</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>423</th>\n",
       "      <td>7.05042</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.614</td>\n",
       "      <td>6.103</td>\n",
       "      <td>85.1</td>\n",
       "      <td>2.0218</td>\n",
       "      <td>24</td>\n",
       "      <td>666</td>\n",
       "      <td>20.2</td>\n",
       "      <td>2.52</td>\n",
       "      <td>23.29</td>\n",
       "      <td>109.126659</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>98</th>\n",
       "      <td>0.08187</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.89</td>\n",
       "      <td>0</td>\n",
       "      <td>0.445</td>\n",
       "      <td>7.820</td>\n",
       "      <td>36.9</td>\n",
       "      <td>3.4952</td>\n",
       "      <td>2</td>\n",
       "      <td>276</td>\n",
       "      <td>18.0</td>\n",
       "      <td>393.53</td>\n",
       "      <td>3.57</td>\n",
       "      <td>35.294118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>455</th>\n",
       "      <td>4.75237</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.10</td>\n",
       "      <td>0</td>\n",
       "      <td>0.713</td>\n",
       "      <td>6.525</td>\n",
       "      <td>86.5</td>\n",
       "      <td>2.4358</td>\n",
       "      <td>24</td>\n",
       "      <td>666</td>\n",
       "      <td>20.2</td>\n",
       "      <td>50.92</td>\n",
       "      <td>18.13</td>\n",
       "      <td>102.068966</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>216</th>\n",
       "      <td>0.04560</td>\n",
       "      <td>0.0</td>\n",
       "      <td>13.89</td>\n",
       "      <td>1</td>\n",
       "      <td>0.550</td>\n",
       "      <td>5.888</td>\n",
       "      <td>56.0</td>\n",
       "      <td>3.1121</td>\n",
       "      <td>5</td>\n",
       "      <td>276</td>\n",
       "      <td>16.4</td>\n",
       "      <td>392.80</td>\n",
       "      <td>13.51</td>\n",
       "      <td>46.875000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>404 rows × 14 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  \\\n",
       "254  0.04819  80.0   3.64     0  0.392  6.108  32.0  9.2203    1  315   \n",
       "348  0.01501  80.0   2.01     0  0.435  6.635  29.7  8.3440    4  280   \n",
       "476  4.87141   0.0  18.10     0  0.614  6.484  93.6  2.3053   24  666   \n",
       "321  0.18159   0.0   7.38     0  0.493  6.376  54.3  4.5404    5  287   \n",
       "326  0.30347   0.0   7.38     0  0.493  6.312  28.9  5.4159    5  287   \n",
       "..       ...   ...    ...   ...    ...    ...   ...     ...  ...  ...   \n",
       "155  3.53501   0.0  19.58     1  0.871  6.152  82.6  1.7455    5  403   \n",
       "423  7.05042   0.0  18.10     0  0.614  6.103  85.1  2.0218   24  666   \n",
       "98   0.08187   0.0   2.89     0  0.445  7.820  36.9  3.4952    2  276   \n",
       "455  4.75237   0.0  18.10     0  0.713  6.525  86.5  2.4358   24  666   \n",
       "216  0.04560   0.0  13.89     1  0.550  5.888  56.0  3.1121    5  276   \n",
       "\n",
       "     PTRATIO       B  LSTAT       TAXRM  \n",
       "254     16.4  392.89   6.57   51.571709  \n",
       "348     17.0  390.94   5.99   42.200452  \n",
       "476     20.2  396.21  18.68  102.714374  \n",
       "321     19.6  396.90   6.87   45.012547  \n",
       "326     19.6  396.90   6.15   45.468948  \n",
       "..       ...     ...    ...         ...  \n",
       "155     14.7   88.01  15.02   65.507152  \n",
       "423     20.2    2.52  23.29  109.126659  \n",
       "98      18.0  393.53   3.57   35.294118  \n",
       "455     20.2   50.92  18.13  102.068966  \n",
       "216     16.4  392.80  13.51   46.875000  \n",
       "\n",
       "[404 rows x 14 columns]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing_features=housing.drop(\"MEDV\",axis=1)\n",
    "housing_labels= housing['MEDV'].copy()\n",
    "housing_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "d7a98820",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SimpleImputer(strategy='median')"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy=\"median\")\n",
    "imputer.fit(housing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "30776017",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2.86735000e-01, 0.00000000e+00, 9.90000000e+00, 0.00000000e+00,\n",
       "       5.38000000e-01, 6.21000000e+00, 7.82000000e+01, 3.12220000e+00,\n",
       "       5.00000000e+00, 3.37000000e+02, 1.90000000e+01, 3.90955000e+02,\n",
       "       1.15700000e+01, 2.11500000e+01, 5.39474541e+01])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "imputer.statistics_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "f0694f41",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=imputer.transform(housing)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "9e6b03d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "housing_tr=pd.DataFrame(x, columns=housing.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d18f03d1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 404 entries, 0 to 403\n",
      "Data columns (total 15 columns):\n",
      " #   Column   Non-Null Count  Dtype  \n",
      "---  ------   --------------  -----  \n",
      " 0   CRIM     404 non-null    float64\n",
      " 1   ZN       404 non-null    float64\n",
      " 2   INDUS    404 non-null    float64\n",
      " 3   CHAS     404 non-null    float64\n",
      " 4   NOX      404 non-null    float64\n",
      " 5   RM       404 non-null    float64\n",
      " 6   AGE      404 non-null    float64\n",
      " 7   DIS      404 non-null    float64\n",
      " 8   RAD      404 non-null    float64\n",
      " 9   TAX      404 non-null    float64\n",
      " 10  PTRATIO  404 non-null    float64\n",
      " 11  B        404 non-null    float64\n",
      " 12  LSTAT    404 non-null    float64\n",
      " 13  MEDV     404 non-null    float64\n",
      " 14  TAXRM    404 non-null    float64\n",
      "dtypes: float64(15)\n",
      "memory usage: 47.5 KB\n"
     ]
    }
   ],
   "source": [
    "housing_tr.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9911fa22",
   "metadata": {},
   "source": [
    "## Creating a pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7e293a88",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler \n",
    "my_pipeline=Pipeline(\n",
    "[\n",
    "    (\"imputer\",SimpleImputer(strategy=\"median\")),\n",
    "    (\"std_scaler\",StandardScaler()),\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "cf2a1a1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.43942006,  3.12628155, -1.12165014, ...,  0.41164221,\n",
       "        -0.86091034, -0.50635717],\n",
       "       [-0.44352175,  3.12628155, -1.35893781, ...,  0.39131918,\n",
       "        -0.94116739, -0.80462611],\n",
       "       [ 0.15682292, -0.4898311 ,  0.98336806, ...,  0.44624347,\n",
       "         0.81480158,  1.12141466],\n",
       "       ...,\n",
       "       [-0.43525657, -0.4898311 , -1.23083158, ...,  0.41831233,\n",
       "        -1.27603303, -1.02444134],\n",
       "       [ 0.14210728, -0.4898311 ,  0.98336806, ..., -3.15239177,\n",
       "         0.73869575,  1.10087256],\n",
       "       [-0.43974024, -0.4898311 ,  0.37049623, ...,  0.41070422,\n",
       "         0.09940681, -0.65584432]])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "housing_tr=my_pipeline.fit_transform(housing_features)\n",
    "housing_tr"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8323be10",
   "metadata": {},
   "source": [
    "## Selecting a desired model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "0d8684c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor()"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "#model=LinearRegression()\n",
    "#model=DecisionTreeRegressor()\n",
    "model=RandomForestRegressor()\n",
    "model.fit(housing_tr,housing_labels)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "62c82f57",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([22.313, 25.537, 16.575, 23.447, 23.423])"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_data=housing_features.iloc[:5]\n",
    "prepared_data=my_pipeline.transform(some_data)\n",
    "some_labels=housing_labels.iloc[:5]\n",
    "model.predict(prepared_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b8df8ded",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "254    21.9\n",
       "348    24.5\n",
       "476    16.7\n",
       "321    23.1\n",
       "326    23.0\n",
       "Name: MEDV, dtype: float64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "some_labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "3a73a268",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.5112525717821772"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "x=model.predict(housing_tr)\n",
    "y=mean_squared_error(housing_labels,x)\n",
    "y"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0eeebf72",
   "metadata": {},
   "source": [
    "# Using better evaluation technique - cross validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "88268d2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "score=cross_val_score(model,housing_tr,housing_labels,scoring=\"neg_mean_squared_error\")\n",
    "rmse_score=np.sqrt(-score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "21e6c10f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def print_score(a):\n",
    "    print(a)\n",
    "    print(\"mean : \",a.mean())\n",
    "    print(\"Standard deviation : \",a.std())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "3b711e3d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2.93973664 3.5892711  3.12194738 3.89347011 3.25272663]\n",
      "mean :  3.3594303705055255\n",
      "Standard deviation :  0.3411603429679328\n"
     ]
    }
   ],
   "source": [
    "print_score(rmse_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "5026e245",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Real_estate.joblib']"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from joblib import dump, load\n",
    "dump(model,\"Real_estate.joblib\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "372a762a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'strat_test_set' (DataFrame)\n"
     ]
    }
   ],
   "source": [
    "%store strat_test_set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "d27a4411",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'my_pipeline' (Pipeline)\n"
     ]
    }
   ],
   "source": [
    "%store my_pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "f9cc04ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Stored 'model' (RandomForestRegressor)\n"
     ]
    }
   ],
   "source": [
    "%store model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dcaf9edc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4bfbc7a8",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
