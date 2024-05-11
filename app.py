from flask import Flask, render_template, request
from main import fetch_values
from randomforest_on__datset_final_year import processModel

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        input_data = []
        for field_name in request.form:
            input_value = request.form.get(field_name)
            input_data.append(int(input_value))

        outputList = fetch_values(input_data)
        # for random forest classifier...
        rfcList = processModel([0,0] + input_data)
        outputList.append(rfcList)

     
      

        
        return render_template('result.html', input_data=outputList)
    return 'Method not allowed'

if __name__ == "__main__":
    app.run(debug=True)
