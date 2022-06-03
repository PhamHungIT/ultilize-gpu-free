import time
import torch
import numpy as np
import pandas as pd
from operator import itemgetter
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util


app = Flask(__name__, static_folder="../front/dist/front", static_url_path="")


def load_model():
    t0 = time.time()
    model = SentenceTransformer('all-MiniLM-L12-v2')
    print(f'Loading time...: {time.time() - t0}')

    return model


def read_data(infile):
    data = pd.read_csv(infile, delimiter='\t')
    categories = data['category'].values.tolist()
    new_categories = data['new_category'].values.tolist()
    product_names = data['product_name'].values.tolist()

    return categories, new_categories, product_names


@app.route('/classify', methods=['GET'])
def classify():
    product_name = request.args.get('product_name')
    new_categories = request.args.getlist('new_categories')

    # import json
    # new_categories_2 = json.loads(request.args.get('new_categories'))
    # print(f'new_categories_2: {new_categories_2}')

    # Category embedding
    if new_categories is not None and len(new_categories) > 0:
        new_category_embeddings = model.encode(new_categories)
        category_embeddings = torch.cat([torch.tensor(base_category_embeddings), torch.tensor(new_category_embeddings)], dim=0)
        categories = base_categories.copy()
        categories.extend(new_categories)

    else:
        category_embeddings = base_category_embeddings
        categories = base_categories.copy()

    # Product embedding
    product_embedding = model.encode(product_name)
    product_embedding = torch.from_numpy(product_embedding)
    product_embedding = product_embedding.unsqueeze(0)

    # Calculate similarity
    scores = util.cos_sim(product_embedding, category_embeddings).squeeze(0).tolist()

    idxes = [i for i, score in enumerate(scores)]
    scores, idxes = list(zip(*sorted(zip(scores, idxes), reverse=True)))

    predicted_category, predicted_score = [], []
    for idx, score in zip(idxes[:10], scores[:10]):
        predicted_category.append(categories[idx])
        predicted_score.append(score)

    return jsonify({
        'category': predicted_category,
        'score': predicted_score
    })


if __name__ == '__main__':
    _, base_categories, product_names = read_data('product_by_categories - product_by_categories.tsv')
    base_categories = list(set(base_categories))
    # with open('categories.txt', 'w') as fw:
    #     for line in sorted(categories_uniques):
    #         fw.write(line.strip() + '\n')
    model = load_model()
    t0 = time.time()
    base_category_embeddings = model.encode(base_categories)
    print(base_category_embeddings)
    print(torch.tensor(base_category_embeddings).size())
    print(f'Encoding category time...:  {time.time() - t0}')

    app.run(host='localhost', port=5005, debug=True)

