from codecs import getreader
import numpy as np
import pandas as pd

def get_res_infer(csv_res, out_xlsx, error_cate, statis_ict):
    raw = pd.read_excel(out_xlsx)
    res = pd.read_csv(csv_res, sep='\t')

    statis = pd.read_csv(statis_ict, sep=',')
    id_map = dict(zip(statis["lv4_name"], statis["lv4_id"]))    
    id_product = []
    for l in res["predicted_label"]:
        try:
            id_product.append(id_map[l])
        except:
            id_product.append(-1)
    res.insert(loc=3, column="id_lv4", value=id_product)


    id_na = list(res.loc[res["sample"]==error_cate].index)
    for i in id_na:
        res.iloc[i] = res.iloc[i].replace(res.iloc[i]["predicted_label"], np.nan)
        res.iloc[i] = res.iloc[i].replace(res.iloc[i]["id_lv4"], np.nan)
        res.iloc[i] = res.iloc[i].replace(res.iloc[i]["probs"], np.nan)
    raw.insert(loc=3, column="predicted_lv4_name", value=res["predicted_label"].values)
    raw.insert(loc=4, column="id_lv4", value=res["id_lv4"].values)
    raw.insert(loc=5, column="probs", value=res["probs"].values)
       
    return raw

if __name__ == "__main__":

    csv_res = "data/ict_category/validate/infer_test_predicted.csv"
    out_xlsx = "data/ict_category/balance_all_ict/product_ICT_KV.xlsx"
    statis_ict = "data/ict_category/statis/Dientu_dienmay_category_Kiotpro.csv"
    
    out_res = "data/ict_category/validate/predicted_train_by_shopee.xlsx"
    res = get_res_infer(csv_res, out_xlsx, "ahihi", statis_ict)
    res.to_excel(out_res, index=False)