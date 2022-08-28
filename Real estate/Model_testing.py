{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "15320078",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(102, 15)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "%store -r strat_test_set\n",
    "%store -r my_pipeline\n",
    "\n",
    "from joblib import load\n",
    "model=load(\"Real_estate.joblib\")\n",
    "\n",
    "strat_test_set[\"TAXRM\"]=strat_test_set[\"TAX\"]/strat_test_set[\"RM\"]\n",
    "strat_test_set.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "46ab8a56",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_features=strat_test_set.drop(\"MEDV\",axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "788a9331",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_labels=strat_test_set[\"MEDV\"].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "17d52acf",
   "metadata": {},
   "outputs": [],
   "source": [
    "prd_test_features=my_pipeline.fit_transform(test_features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "81cb20e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.425007586938784"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predicted=model.predict(prd_test_features)\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import numpy as np\n",
    "err=mean_squared_error(test_labels,predicted)\n",
    "rt_err=np.sqrt(err)\n",
    "rt_err"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f831ba92",
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
