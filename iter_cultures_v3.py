
import numpy as np
import pandas as pd 
import sklearn
from sklearn.preprocessing import MinMaxScaler
import warnings
import geopandas as gpd
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error
import nbimporter
import tensorflow as tf
import keras
from keras.layers import Conv1D,Dense, Dropout, Input, Concatenate, MaxPooling1D, BatchNormalization
import tensorflow.keras
import h5py
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Conv1D,Dense, Dropout, Input
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.layers import Dense, Activation, Concatenate
from keras.callbacks import ModelCheckpoint
import h5py
from keras.models import load_model
from scipy.optimize import curve_fit
import json
import matplotlib
import matplotlib.pyplot as plt 
import os
import math
import pprint
from pymongo import MongoClient
import scipy.interpolate as interpolate
from scipy.interpolate import BSpline, splrep, splev



# THs function extracts the polygon or multipolygon data from a geojson file and return latitude/longitude coordinates for the specific feature type

def replace_substring(array):
    
    return [string.replace("'", ' ') for string in array]

def remove_regions(Regions,regs_remove):
    
    for reg in regs_remove :
        
        remove_idx_region=np.where(Regions['Region Name']==reg)[0]
        Regions=Regions.drop(remove_idx_region,axis=0)
        # Regions=Regions.reset_index(drop=True)
        Regions=Regions.sort_values('Region Name',ascending=True).reset_index(drop=True)
        
    reg_names=np.asarray(Regions['Region Name'])
    code_list=np.asarray(Regions['code'])
    return Regions, reg_names , code_list



    
 #This function scales a given 1D array between 0 and 1   
def scaler(array):
    scaler = MinMaxScaler()
    array=array.reshape(-1,1)
    scaler.fit(array)
    array=scaler.transform(array)
    return array


#Returns a data variable array meaned over a day timeframe

def period_means(arr,year_size_data,choice,nb_measurements):
    
    if choice=='day':
    
        day_means=[]
        _iter=0
        while _iter < year_size_data :
            day_actual=[]
            for hour in range(nb_measurements):
                day_actual.append(arr[_iter])
                _iter+=1
            
            
            day_means.append(np.mean(day_actual))
        
        return day_means 


def find_indices(array1, array2):
    indices = []
    for i, element in enumerate(array1):
        if element in array2:
            indices.append(i)
    return indices
   

#Creates a new geojson file with chosen regions
def geojson_extract(geojson_file,geojson_files_choice,name_attribute,geometry_attribute,code_attribute):
    region_names=[]
    geometries=[]
    code_list=[]
    for choice in geojson_files_choice:
        region_names.append(geojson_file[name_attribute][choice])
        geometries.append(geojson_file[geometry_attribute][choice])   
        code_list.append(geojson_file[code_attribute][choice])
    # Regions.append({'Region Name':region_names,'Geometry':geometries })
    return region_names,geometries,code_list


# Calculates the weights of each class 
def weights(labels,method):
    if method=='binary':
        #Calculating initial weights for each class
        zero_class_nb=0
        one_class_nb=0
        for i in range(len(labels)):
            if labels[i]==0:
                zero_class_nb+=1
            else:
                one_class_nb+=1
        
        # MODIF LOUIS: Division par zéro (car zero_class_nb = 0)
        if zero_class_nb == 0:
            weight_for_0 = 0
        else:
            weight_for_0 = (1 / zero_class_nb) * (len(labels) / 2.0)
        if one_class_nb == 0:
            weight_for_1 = 0
        else:
            weight_for_1 = (1 / one_class_nb) * (len(labels) / 2.0)
        
        class_weight = {
            0: weight_for_0,
            1: weight_for_1
        }
        
        return class_weight,zero_class_nb,one_class_nb

    else:
        print('Incorrect argument')



def gen_labels_portion_4(years_data,production,regions,portion):
    
    #Méthode classification binaire
    binary_labels=[]
    scaler = MinMaxScaler(feature_range=(0, 1))
    first_column = production['Dept']
    remaining_columns = production.iloc[:, 1:]

    # Scale the remaining columns
    scaled_data = scaler.fit_transform(remaining_columns.values.flatten().reshape(-1, 1))
    scaled_data_2d = scaled_data.reshape(remaining_columns.shape)

    # Create a new DataFrame with the scaled values and original structure
    scaled_df = pd.DataFrame(scaled_data_2d, index=remaining_columns.index, columns=remaining_columns.columns)

    # Merge the scaled data with the first column
    production_scaled= pd.concat([first_column, scaled_df], axis=1)

    for region in regions:
        
        prod_region=np.asarray(production_scaled[region])

        #Valeur de référence pour la définition des classes
        valeur_olympique=np.mean(prod_region)*portion 
        
        # valeur_olympique=portion  

        # valeur_olympique=0.5*portion  
        
        for year in years_data:
            
            year_ref=np.where(production['Dept']==year)[0]
            if prod_region[year_ref]<valeur_olympique:
                binary_labels.append(int(1))
            if prod_region[year_ref]>=valeur_olympique:
                binary_labels.append(int(0))
            
    # Méthode multi classification
                             
    return binary_labels 

def gen_std_prod(years_data,production,regions):
    
    #Méthode classification binaire
    max_region_general=[]
    min_region_general=[]
    val_olymp_general=[]
    val_olymp_abs=[]
    mu_regions=[]
    for region in regions:
        
        prod_region=production[region]
        mean_region=np.mean(prod_region)
        max_region=max(prod_region)
        min_region=min(prod_region)
        olymp_vals=[]
        for val in range(5):
            olymp_vals.append(np.asarray(prod_region)[-val])
        
        olymp_vals=np.delete(olymp_vals,np.argmax(olymp_vals))
        olymp_vals=np.delete(olymp_vals,np.argmin(olymp_vals))
        
        max_region_general.append(np.round((max_region/mean_region),2))
        min_region_general.append(np.round((min_region/mean_region),2))
        val_olymp_general.append(np.mean(olymp_vals)/mean_region)
        val_olymp_abs.append(np.mean(olymp_vals))
        mu_regions.append(np.mean(prod_region))
    return   max_region_general,min_region_general,np.std(min_region_general),np.std(max_region_general),val_olymp_general,val_olymp_abs,mu_regions

def region_year_filter(region_choice,year_choice,nb_years_tot,data_matrix,nb_measurements,feature_choice):
    
    nb_years=len(year_choice)
    
    nb_examples=(len(region_choice)*(nb_years))
    data_matrix=data_matrix.T[feature_choice]
    data_matrix=data_matrix.T
    new_data_matrix=np.zeros((nb_examples,365,len(feature_choice)))
    
    new_labels = []                            

    region_idx_start=0
    region_idx_end=nb_years-1
    region_iter=0
    
    for region in region_choice:
        
        idx_start=(region*nb_years_tot)
        idx_end=(region*nb_years_tot)+nb_years-1
        new_data_matrix[region_idx_start:region_idx_end+1]=data_matrix[idx_start:idx_end+1]
        
        region_iter+=1
        region_idx_start+=nb_years
        region_idx_end+=nb_years

    return new_data_matrix


def rms_prod_model(predictions,rendement,year):
    rms=[]
    rms_noabs=[]
    prod_region=np.asarray(rendement)[np.where(rendement['Dept']==year)[0]]
    prod_region=np.delete(prod_region,0)
    for i in range(len(prod_region)):
        rms.append(abs(prod_region[i]-predictions[i]))
        rms_noabs.append(prod_region[i]-predictions[i])
    return np.mean(rms),np.std(rms),min(rms),max(rms),rms,rms_noabs

def accuracy_prod_assur(predictions,rendement,year):
    rms=[]
    rms_noabs=[]
    prod_region=np.asarray(rendement)[np.where(rendement['Dept']==year)[0]]
    val_olymps=[]
        
    prod_region=np.delete(prod_region,0)
    
    for reg in range(len(prod_region)):
        
        val_olymp_year=[]
        for i in range(5):
            iter_year=year-i
            val_olymp_year.append(np.asarray(rendement)[np.where(rendement['Dept']==iter_year)[0]][0][reg+1])
        val_olymp_year=np.delete(val_olymp_year,np.argmax(val_olymp_year))
        val_olymp_year=np.delete(val_olymp_year,np.argmin(val_olymp_year))
        val_olymps.append(np.mean(val_olymp_year))

    accuracy_assur70=[]
    accuracy_assur80=[]
    for i in range(len(prod_region)):
        
        if predictions[i]> 0.7*val_olymps[i] :
    
            if prod_region[i]<0.7*val_olymps[i]:
                accuracy_assur70.append(i)
                
        if predictions[i]<0.7*val_olymps[i] :
    
            if prod_region[i]>0.7*val_olymps[i]:
                accuracy_assur70.append(i)
                
        if predictions[i]>0.8*val_olymps[i] :

            if prod_region[i]<0.8*val_olymps[i]:
                accuracy_assur80.append(i)   
                
        if predictions[i]<0.8*val_olymps[i] :

            if prod_region[i]>0.8*val_olymps[i]:
                accuracy_assur80.append(i)      

    score70=len(accuracy_assur70)/len(prod_region)
    score80=len(accuracy_assur80)/len(prod_region)
    
    return score70,score80

# REQUEST 2: Retourne la liste des rendements pour une culture et un département donné

def get_rendements(my_cult, my_area):

    pipeline = [
        {
            "$match": {
                "cult": my_cult,
                "area": my_area
            }
        },
        {
            "$project": {
                "_id": 0,
                "yield_array": {
                    "$map": {
                        "input": "$years",
                        "as": "year",
                        "in": "$$year.yield"
                    }
                }
            }
        }
    ]

    # Execute the aggregation
    results = list(collection_p_france.aggregate(pipeline))

    if len(results) and 'yield_array' in results[0]:
        # "res" est la liste des rendements
        res = results[0]['yield_array']

        #print('Il y a', len(res), 'rendements:')

        # Afficher chaque valeur
        #for value in res:
        #    pprint.pprint(value)
    
        return res
    return []

# REQUEST 1: Retourne la liste des departements dont tous les années sont au-dessus de 0 de production

def get_departements_complets(my_cult):

    pipeline = [
        {
            "$match": {
                "cult": my_cult
            }
        },
        {
            "$project": {
                "area": 1,
                "initial_years_length": {"$size": "$years"},
                "years": {
                    "$filter": {
                        "input": "$years",
                        "as": "year",
                        "cond": {"$gt": ["$$year.prod", 0]}
                    }
                },
                "list_years": {
                    "$map": {
                        "input": "$years",
                        "as": "year",
                        "in": "$$year.year"
                    }
                }
            }
        },
        {
            "$match": {
                "years.0": {"$exists": True},
                "$expr": {"$eq": ["$initial_years_length", {"$size": "$years"}]}
            }
        },
        {
            "$group": {
                "_id": None,
                "areas": {"$addToSet": "$area"},
                "list_years": { "$addToSet": "$list_years" }
            }
        },
        {
            "$project": {
                "_id": 0,
                "areas": 1,
                "list_years": 1,
            }
        }
    ]

    results = list(collection_p_france.aggregate(pipeline))

    if len(results) and 'areas' in results[0]:
        # "res" est la liste des départements complets
        res = results[0]['areas']
        years_array = results[0]['list_years'][0]
        
        #print(years_array)

        print('Il y a', len(res), 'departements complets:')

        #for document in res:
        #    print(document)
    
        return res, years_array
    return [], 0

nb_years_tot=23
rand_state=np.random.randint(50)

path_departments ='Geojson_files/contour-des-departements.geojson'
French_departments = gpd.read_file(path_departments)
regions_names,geometries,code_list=geojson_extract(French_departments,np.arange(len(French_departments)),'nom','geometry','code')
geojson_columns= {'Region Name': regions_names, 'Geometry':geometries,'code':code_list}
Regions=gpd.GeoDataFrame(geojson_columns)
regs_to_remove=['Paris','Haute-Corse','Corse-du-Sud']

Regions, reg_names, code_list = remove_regions(Regions,regs_to_remove)
reg_names = replace_substring(reg_names)



data_matrix_load = np.loadtxt("data_matrix_departments.txt")

data_features=['u10', 'v10', 't2m', 'evabs', 'evatc', 'evavt','src', 'stl1', 'sp', 'e', 'tp', 'swvl1']
Nb_examples = len(Regions) * nb_years_tot
data_matrix_init= data_matrix_load.reshape(
    data_matrix_load.shape[0], data_matrix_load.shape[1] //  len(data_features), len(data_features))
data_matrix_init=data_matrix_init.reshape((Nb_examples,365,len(data_features)))





cultures = {}
with open('lexique_cultures.json') as file:
    cultures = json.load(file)


# Create a MongoClient to the running mongod instance
# PROD:
#client = MongoClient('mongodb+srv://admin:xQOh4UX0qNDkig18CZe9@mongodb-65953a22-o6c2cee0f.database.cloud.ovh.net/admin?replicaSet=replicaset&tls=true')
# TEST:
client = MongoClient('mongodb+srv://admin:vR3Yxfrmq2G4t5AhM1kd@mongodb-45a72fc7-o6c2cee0f.database.cloud.ovh.net/admin?replicaSet=replicaset&tls=true')


# # Remove all documents from the 'france' collection
# collection.delete_many({})

# Choix de la culture :
#culture_num=100


culture_keys=[99,152,143]

for culture_num in culture_keys:
    culture_name=cultures[str(culture_num)]

    print('Culture en cours : ',culture_name)

        # Connect to your 'production' database
    db_production = client['production']

    # Connect to your 'forecast_loss' database
    collection_p_france = db_production['france']


    year_choices=[2020]
    


    #This will give us a score for test_size  if True

    # Mettre False rendra les évaluations déterministes i.e même données d'entraînement, même données de test (utiles pour test de score avec d'autres paramètres), 
    # Mettre True rendra les évaluations non déterministes

    random = True
    # [True,False]


    feature_choice=[0,2,10]
    # feature_choice=[0,1,2,3,4,5,6,7,8,9,10,11]
    # data_features=['u10', 'v10', 't2m', 'evabs', 'evatc', 'evavt','src', 'stl1', 'sp', 'e', 'tp', 'swvl1']


    #Nombre d'itérations d'entraînement
    nb_epochs=30
    # [20 to 35]

    #Départ de l'enregistrement de loss minimal trouvé
    start_loss_checkpoint=12
    # [8 to 15 ]


    # Paramètres de l'optimiseur
    optimizer_centering=False
    #True,False
    Verbose=0

    portions=np.arange(0.1,2.2,0.02)


    #Données production, année début / année fin
    year_start=2000
    year_finish=2020

    #Ensemble des années représentées en données de production
    years_data=np.arange(year_start,year_finish+1,1)
    #Lecture du fichier excel de production d'une culture


    #Nombres d'Années représentées dans données climatiques 
    nb_years_tot=23



    #_______________________________________
    #CHARGER DONNEES MANGODB LOUIS

    departements_complets, years_array = get_departements_complets(str(culture_num))
    interval = 1
    # Create an empty dataframe
    df = pd.DataFrame()


    # Set the years_array as the index of the dataframe
    series = pd.Series(years_array, name="Dept")

    # Append the series as a column to the dataframe
    df = pd.concat([df, series], axis=1)

    for dep in departements_complets:
        #print('Exploring cult:', culture, 'dep:', dep)
        rendements = get_rendements(str(culture_num), dep)
        #print(rendements)

        # Create a series with rendements as data and dep as the column name
        series = pd.Series(rendements, name=dep)

        # Append the series as a column to the dataframe
        df = pd.concat([df, series], axis=1)

    rendement=df

    db_forecast_test = client['forecast_loss']

    collection_ft_france = db_forecast_test['france']


    # (Solution plus clean)
    # Set the years_array as the index of the dataframe
    #df.index = years_array
    # rendement = pd.read_excel('Production data/cultures_2000_2020/xlsx/culture_'+str(culture_num)+'Rendement'+'.xlsx')


    
    #__________________________________________________________


    #Removing column name
    reg_rep=np.delete(np.asarray(rendement.columns),0)

    x_max_region,x_min_region,std_xmax,std_xmin,val_olymp_region,olymp_abs,mean_regions=gen_std_prod(years_data,rendement,reg_rep)

    reg_filt = find_indices(code_list, reg_rep)

    portions_min=np.round(np.mean(x_min_region),2)
    portions_max=np.round(np.mean(x_max_region),2)

    year_data_start=2000
    drop_out_rate=0.1
    outlayer_neurons=128

    loss_histories=[]

    # year_pick=int(year_choice-year_data_start)
    # pick_years=[year_pick]
    # for reg in range(1,len(reg_filt)):
    #     pick_years.append(year_pick+len(years_data)*reg)

    tf.keras.optimizers.RMSprop(
        learning_rate=0.001,
        centered=optimizer_centering,
        use_ema=False,
        name="RMSprop",
    )
    portions_off=[]
    a=0
    for portion in portions : 
        
        if random == True:
            rand_state=np.random.randint(50)
        #Construction et définition du réseau de neurones 
        labels=gen_labels_portion_4(years_data,rendement,reg_rep,portion)
        data_matrix=region_year_filter(reg_filt,years_data,nb_years_tot,data_matrix_init,24,feature_choice)
        Class_weights,zero_class_nb,one_class_nb=weights(labels,'binary')
        if one_class_nb ==0 or zero_class_nb ==0:
            portions_off.append(a)
            continue
        a+=1
#         if year_pick<len(years_data):

#             data_matrix=np.delete(data_matrix,pick_years,axis=0)
#             labels=np.delete(labels,pick_years,axis=0)

        nb_regions=len(Regions) 



        model = Sequential([

              layers.Conv1D(256, 14, padding='same', activation='relu', input_shape=(365,len(feature_choice))),
              layers.MaxPooling1D(pool_size=8),
              layers.Dropout(drop_out_rate),
              layers.Conv1D(128, 4, padding='same', activation='relu'),
              layers.MaxPooling1D(pool_size=6),
              layers.Dropout(drop_out_rate),
              layers.Conv1D(64, 2, padding='same', activation='relu'),
              layers.MaxPooling1D(pool_size=4),
              layers.Dropout(drop_out_rate),
              layers.Flatten(),
              layers.Dense(outlayer_neurons, activation='relu'),
              layers.Dense(2,activation='softmax')

        ])


        #Compilation du modèle
        model.compile(optimizer='RMSProp',loss='CosineSimilarity',metrics='binary_accuracy' )

        #Séparation des données test et entraînement
        x_train, x_test, y_train, y_test = train_test_split(data_matrix,to_categorical(labels), test_size=0.01,random_state=rand_state)

        mc = ModelCheckpoint('Best_weights/'+str(culture_num)+'_best_model_'+str(portion*100)+'.h5', monitor='val_loss', mode='min',verbose=0,save_best_only=True,start_from_epoch=start_loss_checkpoint)

        history=model.fit(x_train, 
                          y_train,
                          epochs=nb_epochs,
                          batch_size=4,
                          verbose=Verbose,
                          class_weight=Class_weights,
                          validation_split=0.25,
                          callbacks=[mc])


        saved_model = load_model('Best_weights/'+ str(culture_num)+'_best_model_'+str(portion*100)+'.h5')

        min_loss_history=min(history.history['val_loss'])
        loss_histories.append(min_loss_history)



    years_tot=np.arange(2000,2023,1)

    data_matrix2=region_year_filter(reg_filt,years_tot,nb_years_tot,data_matrix_init,24,feature_choice)
    portions=np.delete(portions,portions_off)
    for year_choice in year_choices:
        
        year_pick=int(year_choice-year_data_start)


        pick_years=[year_pick]
        for reg in range(1,len(reg_filt)):
            pick_years.append(year_pick+nb_years_tot*reg)

        test_year_data=data_matrix2[pick_years]


        #Génération des prédictions
        prob_csv=[]
        for portion in portions : 

                saved_model = load_model('Best_weights/'+str(culture_num)+'_best_model_'+str(portion*100)+'.h5')

                predictions=np.round(np.asarray(saved_model.predict(test_year_data))[:,1],2)
                prob_csv.append(predictions)

        Columns=np.asarray(reg_rep)
        prob_csv=pd.DataFrame(prob_csv,columns=Columns,index=portions)


        prod_france_year_fit=[]
        N=10000

        for choice in range(len(np.asarray(prob_csv).T)):

            most_prob=[] 

            #####

            prob_region=np.asarray(prob_csv).T[choice] 

            t, c, k = interpolate.splrep(portions, prob_region, s=0, k=4)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)
            xmin, xmax =portions.min(), portions.max()
            xx = np.linspace(xmin, xmax, N)

            n_interior_knots = 5

            qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
            knots = np.quantile(xx, qs)
            tck = splrep(xx, spline(xx), t=knots, k=2)
            ys_smooth = splev(xx, tck)

            prediction_rendement=np.nanmean(xx[np.where(np.round(ys_smooth,2)==0.5)[0]])*mean_regions[choice]

            prod_france_year_fit.append(prediction_rendement)

            # MODIF LOUIS: Pour ne pas avoir de problèmes d'approximation des float (ex: 1.4999 = 1.4)
            #Précédemment: steps=np.arange(0.2,1.5,0.05)
            steps_olymp = np.arange(0.3, 1.0, 2) / 100

            new_document = {
                "id": str(int(np.asarray(reg_rep)[choice])), 
                "key": np.asarray(reg_rep)[choice], 
                "cult": str(culture_num), 
                "annee": year_pick,
                "valeur_ol": olymp_abs[choice], # You might want to fill this value
                'pred_rend': prediction_rendement,

            }

            val_probs=[]
            for step in steps_olymp:

                idx_pred_olymp=np.where(np.round(xx,2)==np.round(val_olymp_region[choice]*step,2))[0]
                val_probs.append(np.round(np.mean(ys_smooth[idx_pred_olymp])*100,2))

            
            risks = {}

            for i in range(len(val_probs)):
                risks['r_' + str(int(steps_olymp[i]*100))] = val_probs[i]

            new_document['risks'] = risks

            collection_ft_france.insert_one(new_document)

    #             mean_rms,std_rms,min_rms,max_rms,rms,rms_noabs=rms_prod_model(prod_france_year_fit,rendement,year_choice)
    #             score70,score80=accuracy_prod_assur(prod_france_year_fit,rendement,year_choice)

