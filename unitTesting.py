import pymongo


def unitTest(inputValues , valueFromNaiveBayes , valueFromNaiveRandomForest):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["Testing"]
    collection = db["unitTesting"]
    columns = ['Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards',
           'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking',
           'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
           'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
           'Frequent Cold', 'Dry Cough', 'Snoring' , 'Naive Bayes' , 'RandomForest']


    level = 0
    for i in range(len(valueFromNaiveBayes)):
        if valueFromNaiveBayes[i] == max(valueFromNaiveBayes):
            level = i
            break

    
    values = inputValues
    values.append(level)
    values.append(valueFromNaiveRandomForest.tolist()[0])
    data = dict(zip(columns, values))

    print(values)

    insert_result = collection.insert_one(data)

    print("Inserted ID:", insert_result.inserted_id)
