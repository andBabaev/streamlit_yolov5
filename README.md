# Пример приложения на Streamlit
В этой папке находится пример приложения на Streamlit для визуализации работы YOLOv5. 

[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html) - фреймворк, позволяющий создать приложение для визуализации данных или демонстрации возможностей модели. Фреймворк предназначен для быстрого построения UI, однако не позволяет выполнять персонализацию элементов страницы с помощью CSS. Для настройки внешнего вида интерфейса необходимо использовать другой фреймворк, например, Dash Plotly

[YOLOv5](https://github.com/ultralytics/yolov5) - CNN для разспознавания объектов. Используется [версия модели](https://pytorch.org/hub/ultralytics_yolov5/) из библиотки Torch Hub, обученная обнаружению 80 классов объектов из датасета MS COCO. Список классов можно найти в файле `config.py`

## Пример работы
Полную версию демонстрации можно найти на [youtube](https://youtu.be/f_gbRXk6V0Y)
![caption](content/demo.gif)

## Требования
Работоспособность проверена на Ubuntu 20.04 Python 3.8. Ожидается, что приложение должно работать на Python 3.6-3.9

## Файлы приложения
* `app.py` - скрипт для запуска приложения
* `config.py` - скрипт, содержащий константы

## Запуск
Находясь в папке `dashboards/streamlit_guide`, выполнить
```
pip install -r requirements.txt
streamlit run app.py
```
и перейти по ссылке http://localhost:8501

Или через docker
```
docker build -t st .
docker run -p 8501:8501 st
```
**Замечание:** При использовании docker Streamlit выведет в терминале External URL и Network URL. При переходе по ним веб камера не будет работать из-за протокола HTTP (нужен HTTPS). Но если перейти по ссылке http://localhost:8501, то проблема будет решена

## Детали реализации
### Streamlit
Streamlit позволяет выводить на странице приожения текст, изображения, видео, аудио, графики, датафреймы. Также возможно настроить стрим с веб камеры. Подробнее обо всех поддерживаемых для вывода типов данных можно узнать в [документации](https://docs.streamlit.io/en/stable/api.html).

**Добавление компонентов**

Добавление элементов интерфейса на страницу осуществляется в том порядке, в котором были вызваны соответствующие им функции:
```
import streamlit as st
st.title('Streamlit app')
st.markdown('Streamlit is **_really_ cool**.')
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
```
![example](content/streamlit_simple_app.png)

Все добавленные таким образом элменты будут размещены в основной части страницы. Однако добавив `sidebar` при создании элемента, он будет добавлен на боковой виджет:
```
...
if st.sidebar.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
```
![example](content/simple_app_sidebar.png)

**Кэширование**

При изменении состояния любого элемента управления весь код скрипта с приложением будет выполнен заново. Чтобы избежать повторного выполнения вычислительно затратных операций, в Streamlit применяется кэширование. Для включения кэширования функции необходимо указать декоратор [streamlit.cache()](https://docs.streamlit.io/en/stable/api.html#streamlit.cache):
```
@st.cache(max_entries=2)
def get_yolo5(model_type='s'):
    ...
```
Для того чтобы решить, можно ли использовать хранящиеся в кэше данные, Streamlit проверяет:
1. Входные параметры
2. Тело функции
3. Значения глобальных переменных, используемых в теле функции

Если в кэше присутствует объект, совпадающий с текущим состоянием функции по всем трем пунктам, то он будет использован. По умолчанию вместимость кэша не ограничена, но с помощью аргумента `max_entries` можно установить лимит. 

При работе с кэшем необходимо помнить, что для входных и выходных данных в кэше будет сохранена ссылка на них. Следовательно любое изменение этих данных в коде после вызова кэшируемой функции повлечен изменение их значений в кэше. Поэтому необходимо создавать копии входных и выходных данных:
```
# get_preds - кэшируемая функция
result = get_preds(img)
result_copy = result.copy() #для списков - deepcopy(list)
img_copy = img.copy()
# result_copy и img_copy можно использовать, не боясь изменить result и img
```
Подробнее о кэшировании можно прочесть в [документации](https://docs.streamlit.io/en/stable/caching.html)

### streamlit-webrtc
Для стриминга аудио и видео по сети в том числе в веб камеры, необходимо использовать библиотеку [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc). 

С реализацией стрминга видео с веб камеры можно ознакомиться в файле `app.py`. Класс `VideoTransformer` позволяет задать обработку каждого кадра в функции `transform()`. Для инициализации стрима используется функция `webrtc_streamer()`. Для того, чтобы при обновлениях скрипта приложения передавать новые значения глобальных переменных в `VideoTransformer`, необходимо объявить их как атрибуты класса и при обновлении скрипта передавать:
```
WEBRTC_CLIENT_SETTINGS = ClientSettings(
        media_stream_constraints={"video": True, "audio": False},
    )

ctx = webrtc_streamer(
        key="example", 
        video_transformer_factory=VideoTransformer,
        client_settings=WEBRTC_CLIENT_SETTINGS,)

# необходимо для того, чтобы объект VideoTransformer подхватил новые данные
# после обновления страницы streamlit
if ctx.video_transformer:
    ctx.video_transformer.model = model
    ctx.video_transformer.rgb_colors = rgb_colors
    ctx.video_transformer.target_class_ids = target_class_ids
```
В коде выше использован класс `ClientSettings()`. Он позволяет при выключить/включить стриминг аудио/видео.

### YOLOv5
Для загрузки YOLOv5 был использован torch.hub() - это модуль pytorch, позволяющий загружать предобученные модели.
```
import torch
from PIL import Image
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# load image
img = Image.open('example.png')

# or
# img = cv2.imread('example.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

preds = model([img])
preds = preds.xyxy[0].numpy()
# preds = preds.xyxy[0].cpu().numpy() # если используется gpu
```
* `xyxy` вернет найденные боксы в формате:
```
[[xmin,ymin,xmax,ymax,conf,label],
...]
```
* `xyxyn` вернет нормализованные координаты
* `xywh`,`xywhn` вернут ширину и высоту бокса вместо xmax и ymax, соответственно

Для получения боксов только нужных классов:
```
target_class_ids = [0,1] # люди и велосипеды
#обязательно копируем, если работает с кэшем Streamlit
result_copy = result.copy() 
result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
```