from flask import Flask,request,jsonify
from model import  train_example_model


app=Flask(__name__)

model=train_example_model()

@app.route("/")
def index():
    return"Linear Regression API is running"

@app.route("/predict",methods=["POST"])
def predict():
    """hope the JSON body is like
    {
        "features":[[x11,x12,...],[x21,x22,...],...]
        }
        or just a single sample
        {
        "features":[x1,x2,...]
        }"""
    data=request.get_json()

    if not data or "features" not in data:
         return jsonify({"error":"JSON body must contain 'features' field"}),400
    
    features=data["features" ]
    try:
        predictions=model.predict(features)
        preds_list=predicitions.tolist()
    except Exception as e:
        return jsonify({"error":str(e)}),400
    return jsonify({"predictions":preds_list})

if __name__=="__main__":

    app.run(host="0.0.0.0",port=5000,debug=True)
    