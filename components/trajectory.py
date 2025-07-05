def processDataframe(model, scaler, input_df):
    numeric_cols = ['bmi', 'physicalhealth', 'mentalhealth', 'sleeptime', 'age_ave_range']
    existing_cols = model.feature_names
    
    input_df['age_ave_range'] = input_df['agecategory'].apply(age_midpoint)
    
    input_df['sex_male'] = (input_df['sex'] == 'Male').astype(bool)
    input_df['sex_female'] = (input_df['sex'] == 'Female').astype(bool)
    
    input_df['diabetic_yes'] = (input_df['diabetic'] == True).astype(bool)
    input_df['diabetic_no'] = (input_df['diabetic'] == False).astype(bool)
    
    health_mapping = {
        'poor': 'health_status_poor',
        'fair': 'health_status_fair',
        'good': 'health_status_good',
        'very good': 'health_status_very_good',
        'excellent': 'health_status_excellent'
    }
    
    input_df[numeric_cols] = input_df[numeric_cols].astype(float)
    
    for key, col in health_mapping.items():
        input_df[col] = (input_df['genhealth'] == key).astype(bool)
        
    input_df = input_df.drop(['sex', 'diabetic', 'genhealth', 'agecategory'], axis=1)
    
    for cols in existing_cols:
        if cols not in numeric_cols:
            if cols not in input_df.columns:
                input_df[cols] = False
            else:
                input_df[cols] = input_df[cols].astype(bool)
        else:
            if cols not in input_df.columns:
                input_df[cols] = 0
    
    for _ in existing_cols:
        print(f'{_}: {input_df[_].values}')
                
    input_df[existing_cols] = scaler.transform(input_df[existing_cols])
    input_df = input_df[existing_cols]
    
    return predict_heart_disease(model, input_df)


def predict_heart_disease(model, processed_data):    
    prediction = model.predict(processed_data.values)[0]
    proba = model.predict_proba(processed_data.values)[:, 1]
    
    return prediction, proba


def age_midpoint(value): # converts string type age to mid value age (average)
    value = value.lower().strip()
    if '-' in value:
        split_num = value.split('-')

        return (int(split_num[0]) + int(split_num[1])) / 2
    elif 'or older' in value:
        num = ''.join(filter(str.isdigit, value))
        return int(num) + 5
    else:
        try:
            return float(value)
        except:
            return None