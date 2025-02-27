# Классификация состояния деревьев NY 2015 Street Tree Census
Проект классификации состояния деревьев (Good/Fair/Poor) на данных NY 2015 Street Tree Census с использованием DL-фреймворка PyTorch.

## Данные
В проекте использованы данные о переписи деревьев в Нью-Йорке (версия 4 от 2024г.) [NY 2015 Street Tree Census - Tree Data](https://www.kaggle.com/datasets/new-york-city/ny-2015-street-tree-census-tree-data/data). Данные содержат информацию о видах деревьев, их диаметре, состоянии, наличии тех или иных проблем по категориям, а также различные данные об административной и территориальной принадлежности, координаты и прочее.


## Архитектура DL-модели

В проекте выбрана архитектура многослойный перцептрон (MLP). Использование MLP архитектуры для табличных данных обосновано тем, что она эффективно работает с категориальными и числовыми признаками, учитывая их нелинейные взаимодействия. При работе с признаками без временной и пространственных связей, MLP способен эффективно моделировать сложные зависимости в данных, а использование архитектур сверточных или рекуррентных сетей необосновано.

Архитектура MLP реализована с 3 скрытыми слоями, постепенно уменьшающейся размерности 512 -> 256 -> 128, которая способствует сжатию и обобщению информации, предотвращая переобучение и снижая вычислительные затраты. 

После каждого полносвязного слоя выполняется нормализация батча, что улучшает сходимость за счет стабилизации распределения активаций, уменьшая внутреннее ковариационное смещение.

В качестве функции активации выбрана ReLU в связи с простотой вычислений и отсутствием проблемы исчезающего градиента, что способствует более быстрой и стабильной сходимости.

Для предотвращения переобучения и более устойчивого обучений после каждого слоя применны Dropout с вероятностью 0.3.

В связи с значительным дисбалансом классов целевой переменной, была выбрана кастомная Loss-функция Focal Loss, которая уменьшает вес легко классифицируемых примеров и увеличивает вес сложных, помогая модели лучше распознавать миноритарные классы.

В качестве оптимизатора выбрано Adam, который сочетает преимущества AdaGrad и RMSProp и обеспечивает быструю сходимость.

Для более плавного обучения применен Learning Rate Scheduler для уменьшения скорости обучения через фиксированные интервалы, что помогает избежать застревания в локальных минимумах и колебаний около глобального минимума, обеспечивая более стабильную сходимость.

В качестве метрики выбрана F1-macro, которая позволяет учитывать качество классификации для каждого класса независимо, что важно при дисбалансе классов и обеспечивает равное внимание всем классам, включая миноритарные.

Параметры модели, включая размер батча, первоначальный learning rate, шаг и величина его уменьшения, а также коэффициенты gamma в focal loss, коэффициент dropout, размерности слоев и т.д. подобраны экспериментально в процессе обучения по мере их влияние на улучшение метрик.

## Структура проекта

`app/`: Файлы для запуска API и получения инференса модели  
    - `app.py`: Код API.  
    - `data_preparation.py`: Файл с кодом для подготовки и трансформации сырых данных для подачи на вход модели.  
    - `model.py`: Файл с кодом для загрузки обученной модели и получения ее инференса.  
    - `encoders/`: Каталог с сохраненными преобразователями данных.  
    - `saved_model/`: Каталог с сохраненной моделью Pytorch.  

`EDA&Train model.ipynb`: Ноутбук с EDA, подготовкой данных и обучением модели MLP на Pytorch.  
`test_api.ipynb`: Ноутбук с кодом для проверки работы API.  
`requirements.txt`: Файл с зависимостями.  

## Запуск API:

* В каталоге `app/` выполнить: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
* API будет доступно по адресу: http://127.0.0.1:8000/docs
* Пример обращения к API и ответа с прогнозом модели можно найти и протестировать в `test_api.ipynb`.
