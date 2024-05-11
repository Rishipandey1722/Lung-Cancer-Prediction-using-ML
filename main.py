
# Hi , this i backend code of FinalYearProject this program will connect mongodb and take input and finally displays the probability of each levels only
# main py file that will display probabilities....
# this program will not run if u dont have mongodb with specified database and collection name...

import pymongo



def fetch_values(inputList):
    client = pymongo.MongoClient("mongodb://localhost:27017/")  
    db_name = "hazard_analysis_db"
    collection_name = "Probability"
    db = client[db_name]
    collection = db[collection_name]
    cursor = collection.find({})
    output = [[] , [] , []]
    j = 0
    for i in cursor:
        DictLevel0 = i['data']['Level_0']
        DictLevel1 = i['data']['Level_1']
        DictLevel2 = i['data']['Level_2']
        
        # print(DictLevel0[str(inputList[j])] , "    " , DictLevel1[str(inputList[j])] , "    " , DictLevel2[str(inputList[j])])
        output[0].append(DictLevel0[str(inputList[j])])
        output[1].append(DictLevel1[str(inputList[j])])
        output[2].append(DictLevel2[str(inputList[j])])


        j+=1


    sum0 , sum1 , sum2 = 0,0,0
    decimalList = [0.0406, 0.0599, 0.0437, 0.039, 0.0402, 0.0171, 0.0749, 0.0963, 0.025, 0.0892, 0.0221, 0.1048, 0.0477, 0.0218, 0.0422, 0.0647, 0.0452, 0.037, 0.0233, 0.021, 0.0408]

    for i in range(len(output[0])):
        sum0 = sum0 + (decimalList[i] * output[0][i]) 
        sum1 = sum1 + (decimalList[i] * output[1][i] )
        sum2 = sum2 + (decimalList[i] * output[2][i]) 

    answer = [sum0 , sum1 , sum2]
    client.close()
    return answer

