
# THs function extracts the polygon or multipolygon data from a geojson file and return latitude/longitude coordinates for the specific feature type
def coord_lister(geom):
    
    if geom.geom_type=='MultiPolygon':
        coords = [list(x.exterior.coords) for x in geom.geoms]
    else:
        coords = list(geom.exterior.coords)        
    return (coords)
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

#This function allows picking the timeframe of our choice from a path directory for our soon-to-be processed data
def data_period_pick(path,year_start,year_finish,month_start,month_finish):
    
    delta_year=year_finish-year_start
    delta_month=month_finish-month_start
    data_files=[]
    year_size_data=[]
    
    for i in range(year_start,year_finish+1):
        month_size_data=[]
        for j in range(month_start,month_finish+1):
            year_actual=str(i)
            month_actual=str(j)
            if j <= 9 :
                data_files.append(path+'-'+year_actual+'-'+'0'+month_actual+'.nc')
            else:
                 data_files.append(path+'-'+year_actual+'-'+month_actual+'.nc')
                
            data=netCDF4.Dataset(data_files[-1])
            month_size_data.append(len(data.variables['t2m']))
            
        year_size_data.append(np.sum(month_size_data))
    return data_files,delta_year,delta_month,year_size_data

#This function separates times series per year for the variable of a region  amongst all avialbable data
def data_year_sep(data,year_size_data):
    
    data_matrix_year=[]
    _iter=0
    
    for i in range(len(year_size_data)):
                                   
        size_year_actual = year_size_data[i]
        _iter_end=_iter+size_year_actual
        data_matrix_year.append(data[int(_iter):int(_iter_end)])                          
        _iter+=year_size_data[i]
            
    return data_matrix_year

#This function generates latitude/longitude coordinates for a region 
def coordinates_region(geojson_file,region_choice):
    
        if geojson_file[region_choice].geom_type=='MultiPolygon':

            coordinates = coord_lister(geojson_file[region_choice])
            nb_polygons=len(coordinates)
            
            data_coords=[]
            for j in range(nb_polygons):

                coords = coordinates[j]
                coords=np.round(coords,1)
                min_lat=min(coords[:,1])
                max_lat=max(coords[:,1])
                lat_list_iter=np.arange(min_lat,max_lat,0.1)
            
                for latitude in lat_list_iter:

                    idx_long=np.where(coords[:,1]==np.round(latitude,1))[0]
                    long_values_at_iter_lat=coords[:,0][idx_long]
                    long_list_iter=np.arange(min(long_values_at_iter_lat),max(long_values_at_iter_lat),0.1)
                    
                    for longitude in long_list_iter:
                        
                        data_coords.append([latitude,longitude])

            return data_coords

        else:    
            
            data_coords=[]
            coordinates = coord_lister(geojson_file[region_choice])
            coordinates=np.round(coordinates,1)
            min_lat=min(coordinates[:,1])
            max_lat=max(coordinates[:,1])
            lat_list=np.arange(min_lat,max_lat,0.1)
            
            for latitude in lat_list:

                idx_long=np.where(coordinates[:,1]==np.round(latitude,1))[0]
                long_values_at_iter_lat=coordinates[:,0][idx_long]
                long_list_iter=np.arange(min(long_values_at_iter_lat),max(long_values_at_iter_lat),0.1)
                 
                for longitude in long_list_iter:
                        
                        data_coords.append([latitude,longitude])
                        
            return data_coords

#This function generates climatic data for a region of interest for a specific variable and replaces and records missing data 
def data_ready_model(data_coords,path_picked_files,data_feature):

        data_variable=[]

        for file in range(len(path_picked_files)):

            main_data= netCDF4.Dataset(path_picked_files[file])
            data=np.asarray(main_data.variables[data_feature]).T
            lat_data = np.asarray(main_data['latitude'])
            long_data = np.asarray(main_data['longitude'])
            meta_data_region_variable=[]
            missing_data_coords_nb=0
            nb_missing_data=0
            for kk in range(len(data_coords)):
                    
                    latitude_iter=data_coords[kk][0]
                    longitude_iter=data_coords[kk][1]
                    ID_lat = np.where(lat_data==latitude_iter)[0]
                    ID_long = np.where(long_data==longitude_iter)[0]
                    if len(ID_lat)==0 or len(ID_long)==0:
                        missing_data_coords_nb+=1
                        continue
                        
                    data_at_coord=data[ID_long][0][ID_lat]
                    
                    if np.isnan(data_at_coord[0].any())==True:
                        for i in range (len(data_at_coord)):
                            if np.isnan(data_at_coord[0][i])==True:
                                if i ==0:
                                    data_at_coord[0][i]==data_at_coord[0][i+1]
                                else:
                                    data_at_coord[0][i]==data_at_coord[0][i-1] 
                                nb_missing_data+=1
                    if data_at_coord[0].any()==-32767:
                        for i in range(len(data_at_coord)):
                            if data_at_coord[0][i]==-32767:
                                if i ==0:
                                    data_at_coord[0][i]==data_at_coord[0][i+1]
                                else:
                                    data_at_coord[0][i]==data_at_coord[0][i-1]   
                                nb_missing_data+=1
                    if data_at_coord[0].any()==0:
                        for i in range(len(data_at_coord)):
                            if data_at_coord[0][i]==0:
                                if i ==0:
                                    data_at_coord[0][i]==data_at_coord[0][i+1]
                                else:
                                    data_at_coord[0][i]==data_at_coord[0][i-1] 
                                nb_missing_data+=1
                                    
                    meta_data_region_variable.append(data_at_coord[0])
            
            meta_data_region_variable=np.asarray(meta_data_region_variable)
            nb_data_coords=len(data_coords)- missing_data_coords_nb
            meta_data_region_variable=meta_data_region_variable.reshape((nb_data_coords,len(data_at_coord[0])))

            data_variable=np.append(data_variable,np.mean(meta_data_region_variable,axis=0))
            
        return data_variable,nb_missing_data,missing_data_coords_nb
    
    
 #This function scales a given 1D array between 0 and 1   
def scaler(array):
    scaler = MinMaxScaler()
    array=array.reshape(-1,1)
    scaler.fit(array)
    array=scaler.transform(array)
    return array


#This function generates a data matrix for the timeframe, region and data features of our choice.
# This is also where we need tp choose if we scale the data or not
def data_gen_region(data_features,geojson_file,region_choice,scaling,year_start,year_finish):
    
    #Picking the period of data we want 
    
    path_picked_files,nb_years,nb_months,year_size_data=data_period_pick('era1/era5-land',year_start,year_finish,1,12)
    data_region_year=[]           
    data_coords=coordinates_region(geojson_file['Geometry'],region_choice)
    region_missing_data=[]
    region_missing_data_coords=[]
    for variable in range(len(data_features)):
        
        print(geojson_file['Region Name'][region_choice], ':',variable+1,'/',len(data_features), flush=True)

        data,nb_missing_data,missing_data_coords_nb = data_ready_model(data_coords,path_picked_files,data_features[variable])
        if scaling==True:
            data=scaler(data)
        
        data_variable_region_year= data_year_sep(data,year_size_data)   
        data_region_year.append(data_variable_region_year)
        region_missing_data.append(nb_missing_data)
        region_missing_data_coords.append(missing_data_coords_nb)
        
    return data_region_year,nb_months,nb_years,year_size_data,region_missing_data,region_missing_data_coords

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


# def period_means2(arr,year_size_data,choice,nb_measurements):
    
    
#     day_means=[]
#     _iter=0
#     while _iter < year_size_data :
#         day_actual=[]
#         for hour in range(nb_measurements):
#             day_actual.append(arr[_iter])
#             _iter+=1
#         day_means.append(np.mean(day_actual))
#     five_days_means=[]
    
#     while _iter < int(year_size_data/(nb_measurements*) :

#     if choice=='month':
#          month_means=[]
#         _iter=0
#         while _iter < month_size_data:
#             month_actual=[]
            
#             for hour in range(24):
                
#                 _iter+=1
#                 month_actual.append(arr[_iter])
#             month_means.append(np.mean(month_actual))
        
        return day_means    

def find_indices(array1, array2):
    indices = []
    for i, element in enumerate(array1):
        if element in array2:
            indices.append(i)
    return indices
   
def data_processing(climate_files,Regions,year_start,year_finish,data_features,scaling,nb_measurements):
    
    years_data=np.arange(year_start,year_finish+1,1)
    nb_years=len(years_data)
    nb_features=len(data_features)
    nb_examples= (len(Regions))*nb_years
    path_picked_files,_,_,year_size_data=data_period_pick(climate_files,year_start,year_finish,1,12)
    
    data_matrix = np.zeros((nb_examples,nb_features,int(min(year_size_data)/nb_measurements)))
    
    #Checking if meta_data is ordered
    time_=[]
    for file in path_picked_files:

        main_data= netCDF4.Dataset(file)
        data=np.asarray(main_data.variables['time'])
        time_.append(data)

    time_=np.hstack(time_)
    if all(b >= a for a, b in zip(time_, time_[1:]))==True:
        print('Data is ordered well timewise')
        


    data_matrix = np.zeros((nb_examples,nb_features,int(min(year_size_data)/nb_measurements)))
    missing_data=[]
    missing_data_coords=[]
    idx_region_year_data=0                       

    for region in range(len(Regions)):
        
        data_region,_,_,year_size_data,region_missing_data,region_missing_data_coords=data_gen_region(data_features,Regions,region,scaling,year_start,year_finish)
        missing_data.append(region_missing_data)
        missing_data_coords.append(region_missing_data_coords)
        for year in range(nb_years):

            for variable in range(len(data_features)):
                # LOUIS/: Moyenne du paramètre sur la région = 365 valeurs
                data_matrix[idx_region_year_data][variable]=period_means(data_region[variable][year],min(year_size_data),'day',24)

            idx_region_year_data+=1 
    print('Processing is done')
    return data_matrix,missing_data,missing_data_coords

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

#If a geojson file already exists, we can use this function to append chosen regions to it
def geojson_concat(Regions,geojson_file,geojson_files_choice,name_attribute,geometry_attribute):
    
    for choice in geojson_files_choice:
        region_name=geojson_file[name_attribute][choice]
        geometry=geojson_file[geometry_attribute][choice]   
        Regions.loc[len(Regions)] = [region_name, geometry]
    
    return Regions

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
                
        weight_for_0 = (1 / zero_class_nb) * (len(labels) / 2.0)
        weight_for_1 = (1 / one_class_nb) * (len(labels) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        return class_weight

    else:
        print('Incorrect argument')


def gen_labels_portion(years_data,production,regions,portion):
    
    #Méthode classification binaire
    binary_labels=[]
    for region in regions:
        
        prod_region=production[region]
        
#         for value in range(len(prod_region)):
            
#             prod_region[value]=prod_region[value].replace(',','.')

        
        prod_region=scaler(np.asarray(prod_region).astype(float))
        prod_ready_region=np.delete(prod_region,np.argmax(prod_region))
        prod_ready_region=np.delete(prod_ready_region,np.argmin(prod_ready_region))
        
        #Valeur de référence pour la définition des classes
        valeur_olympique=np.mean(prod_ready_region)*portion  
        
        for year in years_data:
            
            year_ref=np.where(production['Dept']==year)[0]
            
            if prod_region[year_ref]<valeur_olympique:
                binary_labels.append(int(1))
            if prod_region[year_ref]>=valeur_olympique:
                binary_labels.append(int(0))
            
    # Méthode multi classification
                             
    return binary_labels  
def gen_labels_portion_3(years_data,production,regions,portion):
    
    #Méthode classification binaire
    binary_labels=[]
    for region in regions:
        
        prod_region=production[region]
        
#         for value in range(len(prod_region)):
            
#             prod_region[value]=prod_region[value].replace(',','.')

        prod_region=scaler(np.asarray(prod_region).astype(float))
        
        #Valeur de référence pour la définition des classes
        valeur_olympique=np.mean(prod_region)*portion  
        
        for year in years_data:
            
            year_ref=np.where(production['Dept']==year)[0]
            
            if prod_region[year_ref]<valeur_olympique:
                binary_labels.append(int(1))
            if prod_region[year_ref]>=valeur_olympique:
                binary_labels.append(int(0))
            
    # Méthode multi classification
                             
    return binary_labels 
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

def region_year_filter2(region_choice,year_choice,nb_years_tot,data_matrix,nb_measurements,feature_choice):
    
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
                            
def remove_outliers(prob_region,portions):
        
        items_remove=[]
        new_vals=[]
        for item in range(1,len(portions)-1):
                amp_exter=prob_region[item+1]-prob_region[item-1]
                amp_inter_l=prob_region[item]-prob_region[item-1]
                amp_inter_r=prob_region[item+1]-prob_region[item]
                if amp_exter > amp_inter_l and amp_inter_r :
                    items_remove.append(item)
        
        for item in range(2,len(portions)-1):
                mean_iter=(prob_region[item+1]+prob_region[item-1]+prob_region[item-2])/3
                new_vals.append(mean_iter)
        smooth_prob=prob_region.copy()
        smooth_prob[2:len(portions)-1:1]=new_vals
        return items_remove,smooth_prob
                            
def remove_outliers2(prob_region,portions):
        
        new_vals=[]

        
        for item in range(1,len(prob_region)-1):
                mean_iter=(prob_region[item+1]+prob_region[item-1])/2
                new_vals.append(mean_iter)
        smooth_prob=prob_region.copy()
        smooth_prob[1:len(prob_region)-1:1]=new_vals
        return smooth_prob
    
def min_max_prob(a,b,c):
    
    x= np.linspace(0.2, 2.5, 5000)
    y = a * np.exp(b * x) + c
    try:
        idx_min=min(np.where(np.round(y,3)==0)[0])
        idx_max=max(np.where(np.round(y,3)==1)[0])
        idx_mid=np.where(np.round(y,3)==0.5)[0]
    except: 
        idx_max=np.argmax(y)
        idx_min=np.argmin(y)
        idx_mid=np.where(np.round(y,3)==0.5)[0]
        print('Problematic curve')
    x_max=x[idx_max]
    x_min=x[idx_min]
    x_mid=x[idx_mid]
    return x_min, x_max,x_mid

def min_max_prob2(a,b):
    
    x= np.linspace(0.2, 2.5, 5000)
    y = a * np.exp(b * x) -1
    try:
        idx_min=min(np.where(np.round(y,3)==0)[0])
        idx_max=max(np.where(np.round(y,3)==1)[0])
        idx_mid=np.where(np.round(y,3)==0.5)[0]
    except: 
        idx_max=np.argmax(y)
        idx_min=np.argmin(y)
        idx_mid=np.where(np.round(y,3)==0.5)[0]
        print('Problematic curve')
    x_max=x[idx_max]
    x_min=x[idx_min]
    x_mid=x[idx_mid]
    return x_min, x_max,x_mid
def min_max_prob3(a,b,c):
    
    x= np.linspace(0.2, 2.5, 5000)
    y = (a * np.sqrt(b * x))- c
    try:
        idx_min=min(np.where(np.round(y,3)==0)[0])
        idx_max=max(np.where(np.round(y,3)==1)[0])
        idx_mid=np.where(np.round(y,2)==0.5)[0]
    except: 
        idx_max=np.argmax(y)
        idx_min=np.argmin(y)
        idx_mid=np.where(np.round(y,2)==0.5)[0]
        print('Problematic curve')
    x_max=x[idx_max]
    x_min=x[idx_min]
    x_mid=x[idx_mid]
    return x_min, x_max,x_mid

def cut_prob(prob_region,portions,x_min_region,x_max_region):
    
    min_prob=min(np.where(np.round(portions,1)==x_min_region)[0])
    max_prob=max(np.where(np.round(portions,1)==x_max_region)[0])
                
    prob_region=prob_region[min_prob:max_prob+1]
    portions=portions[min_prob:max_prob+1]
    return prob_region,portions

def cut_prob2(prob_region,Portions,x_min_region,x_max_region):
    
    min_prob=max(np.where(portions<x_min_region)[0])
    max_prob=np.argmax(prob_region)
    print(min_prob,max_prob)
    prob_region=prob_region[min_prob:max_prob+1]
    Portions=portions[min_prob:max_prob+1]
    return prob_region,Portions

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