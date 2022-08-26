from tqdm import tqdm, trange
from core import model_wrapper
from .help_function import *

import numpy as np
import pandas as pd
import configparser

class Build_Data:
    def __init__(self, path_source_data=None, path_model=None) -> None:
        self.source_data = pd.read_csv(path_source_data, sep='\t')
        self.export_data = pd.DataFrame(columns = self.source_data.columns)

        ict      = ["Máy tính & Laptop","Thiết Bị Điện Gia Dụng", "Điện Thoại & Phụ Kiện", "Cameras & Flycam", \
                        "Thiết Bị Âm Thanh", "Thiết Bị Điện Tử", "Máy ảnh - Máy quay phim"]
        cosmetic = ["Sắc Đẹp", "Sức Khỏe & Sắc Đẹp", ]
        mom_baby = ["Mẹ & Bé"]
        fashion  = ["Phụ Kiện Thời Trang"]
        unknown  = ["Sức Khỏe", "Gaming & Console", "Giặt giũ & Chăm sóc nhà cửa"]
        self.map_indus_2_cate = {"Điện tử - Điện máy" : ict, "Mỹ phẩm" : cosmetic, "Mẹ & Bé" : mom_baby, "Thời trang" : fashion, "unknown" : unknown}

        self.MODEL_CONFIG = 'config/trainer/model.cfg'
        self.USE_GPU = False
        self.model_config = configparser.ConfigParser()
        self.model_config.read(self.MODEL_CONFIG)
        self.classify_model = model_wrapper.load_model(path_model, self.model_config['intent'], False, False)

    def get_samples_by_industry(self, industry_name, num_each_category, column_category):
        list_cate = self.map_indus_2_cate[industry_name]
        respond_data = pd.DataFrame(columns=self.source_data.columns)
        for category in tqdm(list_cate):
            samples = get_by_category(data=self.source_data, column=column_category, category=category)[:num_each_category]
            respond_data = pd.concat([respond_data, samples])
        return respond_data
    
    def infer_samples(self, data:pd.DataFrame, column_sample):
        predict = []
        for i in trange(len(data)):
            predict.append(model_wrapper.infer(self.classify_model, data.iloc[i][column_sample])[0])
        data.insert(loc=len(data.columns), column="nn_predict_label", value=predict)
        return data
    
    def get_keys(self, file_name, folder="data/all_category/prepare/all_data/keywords/"):
        path = folder + file_name
        with open(path) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            return lines
    
    def validate_data(self, data, industry_name, column_sample):
        key_cosmetic = self.get_keys("key_cosmetic.txt")
        key_fashion = self.get_keys("key_fashion.txt")
        key_ict = self.get_keys("key_ict.txt")
        key_mom_baby = self.get_keys("key_mom_baby.txt")

        if industry_name == "Điện tử - Điện máy":
            industry_key = key_ict
            other_industry_keys = key_cosmetic + key_fashion + key_mom_baby 
        elif industry_name == "Mỹ phẩm":
            industry_key = key_cosmetic
            other_industry_keys = key_ict + key_fashion + key_mom_baby
        elif industry_name == "Mẹ & Bé":
            industry_key = key_mom_baby
            other_industry_keys = key_ict + key_fashion + key_cosmetic
        elif industry_name == "Thời trang":
            industry_key = key_fashion
            other_industry_keys = key_ict + key_cosmetic + key_mom_baby
        else:
            return data
        """ 
            Validate 1: Infer data  
        """
        data = self.infer_samples(data, column_sample)
        true_infer = data[data["nn_predict_label"] == industry_name]
        false_infer = data[data["nn_predict_label"] != industry_name]

        """
        Validate 2: Refine by keywords
         - With true-infer, eliminate the samples that contain keyword of other industry
         - With false-infer, take the samples that contain keyword respectively and eliminate if contain keyword of other industry
        """

        # True-infer
        samples_true_infer = true_infer[column_sample].values
        id_eliminate = []
        for id, s in tqdm(enumerate(samples_true_infer)):
            clean_s = clean_text(s)
            eliminate = False
            for w in clean_s.split():
                if w in other_industry_keys:
                    eliminate = True
            if eliminate:
                id_eliminate.append(id)
        
        index_eliminate = list(true_infer.iloc[id_eliminate].index)
        all_id = list(true_infer.index)
        index_keep = []
        for i in tqdm(all_id):
            if i not in index_eliminate:
                index_keep.append(i)
        validated_data = true_infer.loc[index_keep]
        
        # False-infer
        samples_false_infer = false_infer[column_sample].values
        id_keep = []
        for id, s in tqdm(enumerate(samples_false_infer)):
            clean_s = clean_text(s)
            keep = False
            for w in clean_s.split():
                if w in industry_key:
                    keep = True
                if w in other_industry_keys:
                    keep = False
            if keep:
                id_keep.append(id)
        index_keep = list(false_infer.iloc[id_keep].index)
        validated_data = pd.concat([validated_data, false_infer.loc[index_keep]])

        return validated_data

    def get_validated_samples(self, industry_name, num_each_category, column_sample="name", column_category="cats_name"):
        industry_data = self.get_samples_by_industry(industry_name, num_each_category, column_category)
        data = self.validate_data(industry_data, industry_name, column_sample)
        return data

    def build(self, num_each_industry, name_ict="Điện tử - Điện máy", name_mom_baby="Mẹ & Bé", name_cosmetic="Mỹ phẩm", name_unknown="unknown"):
        data_ict = self.get_validated_samples(industry_name=name_ict, num_each_category=80000)
        data_mom_baby = self.get_validated_samples(industry_name=name_mom_baby, num_each_category=500000)[:num_each_industry]
        data_cosmetic = self.get_validated_samples(industry_name=name_cosmetic, num_each_category=250000)[:num_each_industry]
        data_unknown = self.get_validated_samples(industry_name=name_unknown, num_each_category=num_each_industry)
        self.export_data = pd.concat([data_ict, data_cosmetic, data_mom_baby, data_unknown])

    def clean_export_data(self):
        self.export_data = pd.DataFrame(columns=self.source_data)