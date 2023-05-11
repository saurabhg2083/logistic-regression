from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        occupation = float(request.form.get('occupation'))
        occupation_husb = float(request.form.get('occupation_husb'))

        if occupation==2:
            occ_2 = 1
            occ_3 = 0
            occ_4 = 0
            occ_5 = 0
            occ_6 = 0

        if occupation==3:
            occ_2 = 0
            occ_3 = 1
            occ_4 = 0
            occ_5 = 0
            occ_6 = 0
            
        if occupation==4:
            occ_2 = 0
            occ_3 = 0
            occ_4 = 1
            occ_5 = 0
            occ_6 = 0

        if occupation==5:
            occ_2 = 0
            occ_3 = 0
            occ_4 = 0
            occ_5 = 1
            occ_6 = 0
        
        if occupation==6:
            occ_2 = 0
            occ_3 = 0
            occ_4 = 0
            occ_5 = 0
            occ_6 = 1

        

        if occupation_husb==2:
            occ_husb_2 = 1
            occ_husb_3 = 0
            occ_husb_4 = 0
            occ_husb_5 = 0
            occ_husb_6 = 0

        if occupation_husb==3:
            occ_husb_2 = 0
            occ_husb_3 = 1
            occ_husb_4 = 0
            occ_husb_5 = 0
            occ_husb_6 = 0

        if occupation_husb==4:
            occ_husb_2 = 0
            occ_husb_3 = 0
            occ_husb_4 = 1
            occ_husb_5 = 0
            occ_husb_6 = 0

        if occupation_husb==5:
            occ_husb_2 = 0
            occ_husb_3 = 0
            occ_husb_4 = 0
            occ_husb_5 = 1
            occ_husb_6 = 0
        
        if occupation_husb==6:
            occ_husb_2 = 0
            occ_husb_3 = 0
            occ_husb_4 = 0
            occ_husb_5 = 0
            occ_husb_6 = 1

        data=CustomData(
            Intercept=1,
            occ_2 = occ_2,
            occ_3 = occ_3,
            occ_4 = occ_4,
            occ_5 = occ_5,
            occ_6 = occ_6,
            occ_husb_2 =occ_husb_2,
            occ_husb_3 = occ_husb_3,
            occ_husb_4 = occ_husb_4,
            occ_husb_5 = occ_husb_5,
            occ_husb_6 = occ_husb_6,
            rate_marriage = float(request.form.get('rate_marriage')),
            age = float(request.form.get('age')),
            yrs_married = float(request.form.get('yrs_married')),
            children = float(request.form.get('children')),
            religious = float(request.form.get('religious')),
            educ = float(request.form.get('educ'))
        )
       
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080, debug=True)
    


