{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "executionInfo": {
     "elapsed": 1704,
     "status": "ok",
     "timestamp": 1606731504328,
     "user": {
      "displayName": "Richard C",
      "photoUrl": "",
      "userId": "08040814671867660929"
     },
     "user_tz": -180
    },
    "id": "sy5EcZ65lzy-"
   },
   "outputs": [],
   "source": [
    "import pandas as pd      \n",
    "import numpy as np \n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "executionInfo": {
     "elapsed": 825,
     "status": "ok",
     "timestamp": 1606731524112,
     "user": {
      "displayName": "Richard C",
      "photoUrl": "",
      "userId": "08040814671867660929"
     },
     "user_tz": -180
    },
    "id": "-AJL3lU-lzzA"
   },
   "outputs": [],
   "source": [
    "def eval_metrics(actual, pred):\n",
    "    rmse = np.sqrt(mean_squared_error(actual, pred))\n",
    "    mae = mean_absolute_error(actual, pred)\n",
    "    mse = mean_squared_error(actual, pred)\n",
    "    score = r2_score(actual, pred)\n",
    "    return print(\" r2_score:\", score, \"\\n\",\"mae:\", mae, \"\\n\",\"mse:\",mse, \"\\n\",\"rmse:\",rmse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
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
       "      <th>make_model</th>\n",
       "      <th>body_type</th>\n",
       "      <th>price</th>\n",
       "      <th>vat</th>\n",
       "      <th>km</th>\n",
       "      <th>Type</th>\n",
       "      <th>Fuel</th>\n",
       "      <th>Gears</th>\n",
       "      <th>Comfort_Convenience</th>\n",
       "      <th>Entertainment_Media</th>\n",
       "      <th>...</th>\n",
       "      <th>Previous_Owners</th>\n",
       "      <th>hp_kW</th>\n",
       "      <th>Inspection_new</th>\n",
       "      <th>Paint_Type</th>\n",
       "      <th>Upholstery_type</th>\n",
       "      <th>Gearing_Type</th>\n",
       "      <th>Displacement_cc</th>\n",
       "      <th>Weight_kg</th>\n",
       "      <th>Drive_chain</th>\n",
       "      <th>cons_comb</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Audi A1</td>\n",
       "      <td>Sedans</td>\n",
       "      <td>15770</td>\n",
       "      <td>VAT deductible</td>\n",
       "      <td>56013.0</td>\n",
       "      <td>Used</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>7.0</td>\n",
       "      <td>Air conditioning,Armrest,Automatic climate con...</td>\n",
       "      <td>Bluetooth,Hands-free equipment,On-board comput...</td>\n",
       "      <td>...</td>\n",
       "      <td>2.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Metallic</td>\n",
       "      <td>Cloth</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>1220.0</td>\n",
       "      <td>front</td>\n",
       "      <td>3.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Audi A1</td>\n",
       "      <td>Sedans</td>\n",
       "      <td>14500</td>\n",
       "      <td>Price negotiable</td>\n",
       "      <td>80000.0</td>\n",
       "      <td>Used</td>\n",
       "      <td>Benzine</td>\n",
       "      <td>7.0</td>\n",
       "      <td>Air conditioning,Automatic climate control,Hil...</td>\n",
       "      <td>Bluetooth,Hands-free equipment,On-board comput...</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>0</td>\n",
       "      <td>Metallic</td>\n",
       "      <td>Cloth</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>1798.0</td>\n",
       "      <td>1255.0</td>\n",
       "      <td>front</td>\n",
       "      <td>5.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Audi A1</td>\n",
       "      <td>Sedans</td>\n",
       "      <td>14640</td>\n",
       "      <td>VAT deductible</td>\n",
       "      <td>83450.0</td>\n",
       "      <td>Used</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>7.0</td>\n",
       "      <td>Air conditioning,Cruise control,Electrical sid...</td>\n",
       "      <td>MP3,On-board computer</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>85.0</td>\n",
       "      <td>0</td>\n",
       "      <td>Metallic</td>\n",
       "      <td>Cloth</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>1598.0</td>\n",
       "      <td>1135.0</td>\n",
       "      <td>front</td>\n",
       "      <td>3.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Audi A1</td>\n",
       "      <td>Sedans</td>\n",
       "      <td>14500</td>\n",
       "      <td>VAT deductible</td>\n",
       "      <td>73000.0</td>\n",
       "      <td>Used</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>6.0</td>\n",
       "      <td>Air suspension,Armrest,Auxiliary heating,Elect...</td>\n",
       "      <td>Bluetooth,CD player,Hands-free equipment,MP3,O...</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>0</td>\n",
       "      <td>Metallic</td>\n",
       "      <td>Cloth</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>1195.0</td>\n",
       "      <td>front</td>\n",
       "      <td>3.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Audi A1</td>\n",
       "      <td>Sedans</td>\n",
       "      <td>16790</td>\n",
       "      <td>VAT deductible</td>\n",
       "      <td>16200.0</td>\n",
       "      <td>Used</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>7.0</td>\n",
       "      <td>Air conditioning,Armrest,Automatic climate con...</td>\n",
       "      <td>Bluetooth,CD player,Hands-free equipment,MP3,O...</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Metallic</td>\n",
       "      <td>Cloth</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>1422.0</td>\n",
       "      <td>1135.0</td>\n",
       "      <td>front</td>\n",
       "      <td>4.1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 23 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "  make_model body_type  price               vat       km  Type     Fuel  \\\n",
       "0    Audi A1    Sedans  15770    VAT deductible  56013.0  Used   Diesel   \n",
       "1    Audi A1    Sedans  14500  Price negotiable  80000.0  Used  Benzine   \n",
       "2    Audi A1    Sedans  14640    VAT deductible  83450.0  Used   Diesel   \n",
       "3    Audi A1    Sedans  14500    VAT deductible  73000.0  Used   Diesel   \n",
       "4    Audi A1    Sedans  16790    VAT deductible  16200.0  Used   Diesel   \n",
       "\n",
       "   Gears                                Comfort_Convenience  \\\n",
       "0    7.0  Air conditioning,Armrest,Automatic climate con...   \n",
       "1    7.0  Air conditioning,Automatic climate control,Hil...   \n",
       "2    7.0  Air conditioning,Cruise control,Electrical sid...   \n",
       "3    6.0  Air suspension,Armrest,Auxiliary heating,Elect...   \n",
       "4    7.0  Air conditioning,Armrest,Automatic climate con...   \n",
       "\n",
       "                                 Entertainment_Media  ... Previous_Owners  \\\n",
       "0  Bluetooth,Hands-free equipment,On-board comput...  ...             2.0   \n",
       "1  Bluetooth,Hands-free equipment,On-board comput...  ...             1.0   \n",
       "2                              MP3,On-board computer  ...             1.0   \n",
       "3  Bluetooth,CD player,Hands-free equipment,MP3,O...  ...             1.0   \n",
       "4  Bluetooth,CD player,Hands-free equipment,MP3,O...  ...             1.0   \n",
       "\n",
       "   hp_kW  Inspection_new  Paint_Type  Upholstery_type  Gearing_Type  \\\n",
       "0   66.0               1    Metallic            Cloth     Automatic   \n",
       "1  141.0               0    Metallic            Cloth     Automatic   \n",
       "2   85.0               0    Metallic            Cloth     Automatic   \n",
       "3   66.0               0    Metallic            Cloth     Automatic   \n",
       "4   66.0               1    Metallic            Cloth     Automatic   \n",
       "\n",
       "  Displacement_cc Weight_kg Drive_chain  cons_comb  \n",
       "0          1422.0    1220.0       front        3.8  \n",
       "1          1798.0    1255.0       front        5.6  \n",
       "2          1598.0    1135.0       front        3.8  \n",
       "3          1422.0    1195.0       front        3.8  \n",
       "4          1422.0    1135.0       front        4.1  \n",
       "\n",
       "[5 rows x 23 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"EDA_scout_car_not_dummy.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 15915 entries, 0 to 15914\n",
      "Data columns (total 23 columns):\n",
      " #   Column               Non-Null Count  Dtype  \n",
      "---  ------               --------------  -----  \n",
      " 0   make_model           15915 non-null  object \n",
      " 1   body_type            15915 non-null  object \n",
      " 2   price                15915 non-null  int64  \n",
      " 3   vat                  15915 non-null  object \n",
      " 4   km                   15915 non-null  float64\n",
      " 5   Type                 15915 non-null  object \n",
      " 6   Fuel                 15915 non-null  object \n",
      " 7   Gears                15915 non-null  float64\n",
      " 8   Comfort_Convenience  15915 non-null  object \n",
      " 9   Entertainment_Media  15915 non-null  object \n",
      " 10  Extras               15915 non-null  object \n",
      " 11  Safety_Security      15915 non-null  object \n",
      " 12  age                  15915 non-null  float64\n",
      " 13  Previous_Owners      15915 non-null  float64\n",
      " 14  hp_kW                15915 non-null  float64\n",
      " 15  Inspection_new       15915 non-null  int64  \n",
      " 16  Paint_Type           15915 non-null  object \n",
      " 17  Upholstery_type      15915 non-null  object \n",
      " 18  Gearing_Type         15915 non-null  object \n",
      " 19  Displacement_cc      15915 non-null  float64\n",
      " 20  Weight_kg            15915 non-null  float64\n",
      " 21  Drive_chain          15915 non-null  object \n",
      " 22  cons_comb            15915 non-null  float64\n",
      "dtypes: float64(8), int64(2), object(13)\n",
      "memory usage: 2.8+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "ojUll31ZlzzD"
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'Opel Insignia'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_2916\\2395761037.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mX_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_test\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtrain_test_split\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtest_size\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0.3\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrandom_state\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m42\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mmodel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mLinearRegression\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 5\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      6\u001b[0m \u001b[0mfilename\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m'streamed_model'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      7\u001b[0m \u001b[0mpickle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdump\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilename\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'wb'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\linear_model\\_base.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight)\u001b[0m\n\u001b[0;32m    661\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    662\u001b[0m         X, y = self._validate_data(\n\u001b[1;32m--> 663\u001b[1;33m             \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0maccept_sparse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_numeric\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmulti_output\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    664\u001b[0m         )\n\u001b[0;32m    665\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[1;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[0;32m    579\u001b[0m                 \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mcheck_y_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    580\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 581\u001b[1;33m                 \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_X_y\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    582\u001b[0m             \u001b[0mout\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    583\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_X_y\u001b[1;34m(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)\u001b[0m\n\u001b[0;32m    974\u001b[0m         \u001b[0mensure_min_samples\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mensure_min_samples\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    975\u001b[0m         \u001b[0mensure_min_features\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mensure_min_features\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 976\u001b[1;33m         \u001b[0mestimator\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mestimator\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    977\u001b[0m     )\n\u001b[0;32m    978\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[0;32m    744\u001b[0m                     \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0marray\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcasting\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"unsafe\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    745\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 746\u001b[1;33m                     \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morder\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    747\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mcomplex_warning\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    748\u001b[0m                 raise ValueError(\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m__array__\u001b[1;34m(self, dtype)\u001b[0m\n\u001b[0;32m   1991\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1992\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__array__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mNpDtype\u001b[0m \u001b[1;33m|\u001b[0m \u001b[1;32mNone\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m->\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndarray\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1993\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_values\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1994\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1995\u001b[0m     def __array_wrap__(\n",
      "\u001b[1;31mValueError\u001b[0m: could not convert string to float: 'Opel Insignia'"
     ]
    }
   ],
   "source": [
    "X= df.drop(\"price\", axis=1)\n",
    "y= df[\"price\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "filename = 'streamed_model'\n",
    "pickle.dump(model, open(filename, 'wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
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
       "      <th>price</th>\n",
       "      <th>km</th>\n",
       "      <th>Gears</th>\n",
       "      <th>age</th>\n",
       "      <th>Previous_Owners</th>\n",
       "      <th>hp_kW</th>\n",
       "      <th>Inspection_new</th>\n",
       "      <th>Displacement_cc</th>\n",
       "      <th>Weight_kg</th>\n",
       "      <th>cons_comb</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "      <td>15915.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>18024.380584</td>\n",
       "      <td>32089.995708</td>\n",
       "      <td>5.937355</td>\n",
       "      <td>1.389695</td>\n",
       "      <td>1.042853</td>\n",
       "      <td>88.499340</td>\n",
       "      <td>0.247063</td>\n",
       "      <td>1428.661891</td>\n",
       "      <td>1337.700534</td>\n",
       "      <td>4.832124</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>7381.679318</td>\n",
       "      <td>36977.214964</td>\n",
       "      <td>0.704772</td>\n",
       "      <td>1.121306</td>\n",
       "      <td>0.339178</td>\n",
       "      <td>26.674341</td>\n",
       "      <td>0.431317</td>\n",
       "      <td>275.804272</td>\n",
       "      <td>199.682385</td>\n",
       "      <td>0.867530</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>4950.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>890.000000</td>\n",
       "      <td>840.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>12850.000000</td>\n",
       "      <td>1920.500000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>66.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1229.000000</td>\n",
       "      <td>1165.000000</td>\n",
       "      <td>4.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>16900.000000</td>\n",
       "      <td>20413.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>85.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1461.000000</td>\n",
       "      <td>1295.000000</td>\n",
       "      <td>4.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>21900.000000</td>\n",
       "      <td>46900.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>103.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1598.000000</td>\n",
       "      <td>1472.000000</td>\n",
       "      <td>5.400000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>74600.000000</td>\n",
       "      <td>317000.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>294.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2967.000000</td>\n",
       "      <td>2471.000000</td>\n",
       "      <td>9.100000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              price             km         Gears           age  \\\n",
       "count  15915.000000   15915.000000  15915.000000  15915.000000   \n",
       "mean   18024.380584   32089.995708      5.937355      1.389695   \n",
       "std     7381.679318   36977.214964      0.704772      1.121306   \n",
       "min     4950.000000       0.000000      5.000000      0.000000   \n",
       "25%    12850.000000    1920.500000      5.000000      0.000000   \n",
       "50%    16900.000000   20413.000000      6.000000      1.000000   \n",
       "75%    21900.000000   46900.000000      6.000000      2.000000   \n",
       "max    74600.000000  317000.000000      8.000000      3.000000   \n",
       "\n",
       "       Previous_Owners         hp_kW  Inspection_new  Displacement_cc  \\\n",
       "count     15915.000000  15915.000000    15915.000000     15915.000000   \n",
       "mean          1.042853     88.499340        0.247063      1428.661891   \n",
       "std           0.339178     26.674341        0.431317       275.804272   \n",
       "min           0.000000     40.000000        0.000000       890.000000   \n",
       "25%           1.000000     66.000000        0.000000      1229.000000   \n",
       "50%           1.000000     85.000000        0.000000      1461.000000   \n",
       "75%           1.000000    103.000000        0.000000      1598.000000   \n",
       "max           4.000000    294.000000        1.000000      2967.000000   \n",
       "\n",
       "          Weight_kg     cons_comb  \n",
       "count  15915.000000  15915.000000  \n",
       "mean    1337.700534      4.832124  \n",
       "std      199.682385      0.867530  \n",
       "min      840.000000      3.000000  \n",
       "25%     1165.000000      4.100000  \n",
       "50%     1295.000000      4.800000  \n",
       "75%     1472.000000      5.400000  \n",
       "max     2471.000000      9.100000  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "U7FTTc6XlzzE",
    "outputId": "d69eaee2-231f-4ba0-f8a2-0a03f0fd9ed9"
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'Opel Insignia'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_2916\\2533574158.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0my_pred\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX_test\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0meval_metrics\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my_test\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0my_pred\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\linear_model\\_base.py\u001b[0m in \u001b[0;36mpredict\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    360\u001b[0m             \u001b[0mReturns\u001b[0m \u001b[0mpredicted\u001b[0m \u001b[0mvalues\u001b[0m\u001b[1;33m.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    361\u001b[0m         \"\"\"\n\u001b[1;32m--> 362\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_decision_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    363\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    364\u001b[0m     \u001b[0m_preprocess_data\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mstaticmethod\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m_preprocess_data\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\linear_model\\_base.py\u001b[0m in \u001b[0;36m_decision_function\u001b[1;34m(self, X)\u001b[0m\n\u001b[0;32m    343\u001b[0m         \u001b[0mcheck_is_fitted\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    344\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 345\u001b[1;33m         \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_validate_data\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"csr\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"csc\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"coo\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mreset\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    346\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0msafe_sparse_dot\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcoef_\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mT\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdense_output\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m+\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mintercept_\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    347\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\base.py\u001b[0m in \u001b[0;36m_validate_data\u001b[1;34m(self, X, y, reset, validate_separately, **check_params)\u001b[0m\n\u001b[0;32m    564\u001b[0m             \u001b[1;32mraise\u001b[0m \u001b[0mValueError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Validation should be done on X, y or both.\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    565\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0mno_val_y\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 566\u001b[1;33m             \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mcheck_params\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    567\u001b[0m             \u001b[0mout\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mX\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    568\u001b[0m         \u001b[1;32melif\u001b[0m \u001b[0mno_val_X\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0mno_val_y\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator)\u001b[0m\n\u001b[0;32m    744\u001b[0m                     \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0marray\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mastype\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcasting\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"unsafe\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    745\u001b[0m                 \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 746\u001b[1;33m                     \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morder\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    747\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mcomplex_warning\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    748\u001b[0m                 raise ValueError(\n",
      "\u001b[1;32mc:\\Users\\TOSHIBA\\anaconda3\\envs\\streamlit2\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m__array__\u001b[1;34m(self, dtype)\u001b[0m\n\u001b[0;32m   1991\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1992\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m__array__\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mNpDtype\u001b[0m \u001b[1;33m|\u001b[0m \u001b[1;32mNone\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m->\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mndarray\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1993\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_values\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1994\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1995\u001b[0m     def __array_wrap__(\n",
      "\u001b[1;31mValueError\u001b[0m: could not convert string to float: 'Opel Insignia'"
     ]
    }
   ],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "eval_metrics(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**deployment**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'streamed_model'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp\\ipykernel_2916\\2383039677.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mrichard_model\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpickle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mload\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'streamed_model'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'rb'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'streamed_model'"
     ]
    }
   ],
   "source": [
    "richard_model = pickle.load(open('streamed_model', 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['TV', 'radio', 'newspaper']"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "columns=list(X.columns)\n",
    "columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "my_dict = {\n",
    "    \"TV\": 150,\n",
    "    \"radio\": 25,\n",
    "    \"newspaper\": 30,\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.DataFrame.from_dict([my_dict])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
       "      <th>TV</th>\n",
       "      <th>radio</th>\n",
       "      <th>newspaper</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>150</td>\n",
       "      <td>25</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    TV  radio  newspaper\n",
       "0  150     25         30"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[14.50650223]\n"
     ]
    }
   ],
   "source": [
    "prediction = richard_model.predict(df)\n",
    "print(prediction)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The estimated value of sales is 14. \n"
     ]
    }
   ],
   "source": [
    "print(\"The estimated value of sales is {}. \".format(int(prediction[0])))"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "modeling_auto_scout.ipynb",
   "provenance": []
  },
  "interpreter": {
   "hash": "5a9a31fe17a2cbd26ce1c8f66b51dac5f19c80dd2e6ae518f36253c8acd414f6"
  },
  "kernelspec": {
   "display_name": "Python 3.7.13 ('streamlit2')",
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
   "version": "3.7.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
