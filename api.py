import os
from flask import Flask, jsonify, request
from core import model_wrapper
import configparser

MODEL_CONFIG = 'config/trainer/model.cfg'
USE_GPU = False
model_config = configparser.ConfigParser()
model_config.read(MODEL_CONFIG)


app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    samples = request.args.getlist('sample')
    respond = {}
    for sample in samples: 
        label, prob = model_wrapper.infer(classify_model, sample.lower())
        respond[sample] = f"{label} - {round(prob,2)}" 
    return respond

if __name__ == "__main__":


    # folder = 'data/all_category/data_run/_274k_fashion_and_other'
    # model_name = 'fashion_classify_semhash_5k.pt'
    # model_path = os.path.join(folder, model_name)
    # model_path = "data/all_category/data_run/_1M_shopee/model/official_model/shopee_train_token_5k_ep1.pt"
    model_path = "data/all_category/data_run/_30k_kiot/all_cate_data_preprocessed.pt"
    # model_path = "data/all_category/data_run/_1M_shopee/model/not_unknown_shopee_token_5k_ep1.pt"

    use_semhash = False
    classify_model = model_wrapper.load_model(model_path, model_config['intent'], False, False)

    app.debug = False
    host = os.environ.get('IP', '0.0.0.0')
    port = int(os.environ.get('PORT', 5555))
    app.run(host=host, port=port, threaded=True, use_reloader=True)