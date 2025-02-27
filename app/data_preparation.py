import pandas as pd
import joblib
import os


class dataTransformer:
    """Класс для предобработки входных данных перед подачей в модель."""
    def __init__(self):
        """
            Загружает разделенные на группы по типам признаки
            и предобученные кодировщики и преобразователи.
        """
        self.num_cols = ['tree_dbh', 'x_sp', 'y_sp']
        self.cat_cols = ['spc_common', 'community board', 'boro_ct']
        self.ohe_cols = ['steward', 'guards', 'borocode']
        self.bin_cols = ['curb_loc', 'sidewalk', 'root_stone', 'root_grate', 'root_other', 'trunk_wire',
                         'trnk_light', 'trnk_other', 'brch_light', 'brch_shoe', 'brch_other']

        path = os.path.dirname(os.path.abspath(__file__))

        self.std_scaler = joblib.load(
            os.path.join(path, 'encoders', 'std_scaler.pkl')
        )
        self.ohe_encoder = joblib.load(
            os.path.join(path, 'encoders', 'ohe_encoder.pkl')
        )
        self.target_encoder = joblib.load(
            os.path.join(path, 'encoders', 'target_encoder.pkl')
        )

    def transform(self, features):
        """
            Преобразует входные данные в список числовых значений,
            готовых для преобразования в тензор и передачи в модель.

            :param
                features:
                    объект Pydantic с признаками
            :return:
                feat_list: list
                    список числовых значений преобразованных признаков, готовых для подачи в модель
        """
        df = pd.DataFrame([features.dict(by_alias=True)])
        std_feats = list(self.std_scaler.transform(df[self.num_cols])[0])
        bin_feats = [0 if v in ['OnCurb', 'No', 'NoDamage'] else 1 for v in df.iloc[0][self.bin_cols].tolist()]
        ohe_feats = list(self.ohe_encoder.transform(df[self.ohe_cols])[0])
        cat_feats = self.target_encoder.transform(df[self.cat_cols]).iloc[0].tolist()
        feat_list = std_feats + bin_feats + ohe_feats + cat_feats
        return feat_list





