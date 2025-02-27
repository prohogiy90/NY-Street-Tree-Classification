import os
import torch
import joblib


class MLPModel:
    """Класс для загрузки модели MLP и выполнения прогнозов."""
    def __init__(self):
        """
            Инициализация модели и загрузка необходимых файлов:
            - Путь к модели
            - Устройство (CPU или GPU)
        """
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(
            os.path.join(self.path, 'saved_model', 'mlp_model.ckpt'), map_location=self.device)
        self.model.eval()

    def predict(self, x):
        """
        Предсказание класса на основе входных данных.
        :param
        x : list
                Входные данные для модели
        :return
        predicted_class : str
                Предсказанный класс
        """
        x = torch.tensor([x], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(x)
            _, pred = torch.max(output, 1)
            lbl_encoder = joblib.load(
                os.path.join(self.path, 'encoders', 'lbl_encoder.pkl')
            )
            predicted_class = lbl_encoder.classes_[pred.cpu().numpy()].item()
        return predicted_class
