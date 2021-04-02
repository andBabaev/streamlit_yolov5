from typing import List

import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.colors as mcolors
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

from config import CLASSES, WEBRTC_CLIENT_SETTINGS

#изменим название страницы, отображаемое на вкладке браузера
#set_page_config должна вызываться до всех функций streamlit
st.set_page_config(
    page_title="YOLOv5 demo",
)

st.title('YOLOv5 demo')

#region Functions
# --------------------------------------------

@st.cache(max_entries=2)
def get_yolo5(model_type='s'):
    '''
    Возвращает модель YOLOv5 из Torch Hub типа `model_type`

    Arguments
    ----------
    model_type : str, 's', 'm', 'l' or 'x'
        тип модели - s - самая быстрая и неточная, x - самая точная и медленная

    Returns
    -------
    torch model
        torch-модель типа `<class 'models.common.autoShape'>`
    '''
    return torch.hub.load('ultralytics/yolov5', 
                          'yolov5{}'.format(model_type), 
                          pretrained=True
                          )

@st.cache(max_entries=10)
def get_preds(img : np.ndarray) -> np.ndarray:
    """
    Возвращает прогнозы, полученные от YOLOv5

    Arguments
    ---------
    img : np.ndarray
        RGB-изображение загруженное с помощью OpenCV

    Returns
    -------
    2d np.ndarray
        Список найденных объектов в формате 
        `[xmin,ymin,xmax,ymax,conf,label]`
    """
    return model([img]).xyxy[0].numpy()

def get_colors(indexes : List[int]) -> dict:
    '''
    Возвращает цвета для всех выбранных классов. Цвета формируются 
    на основе наборов TABLEAU_COLORS и BASE_COLORS из Matplotlib

    Arguments
    ----------
    indexes : list of int
        список индексов классов в порядке по умолчанию для 
        MS COCO (80 классов, без background)

    Returns
    -------
    dict
        словарь, в котором ключи - id классов, указанные в 
        indexes, а значения - tuple с компонентами rgb цвета, например, (0,0,0)
    '''
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) 
                                                for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5

    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)

    return color_dict

def get_legend_color(class_name : int):
    """
    Возвращает цвет ячейки для `pandas.Styler` при создании легенды. 
    Раскарасит ячейку те же цветом, который имеют боксы соотвествующего класс

    Arguments
    ---------
    class_name : int
        название класса согласно списку классов MS COCO

    Returns
    -------
    str
        background-color для ячейки, содержащей class_name
    """  

    index = CLASSES.index(class_name)
    color = rgb_colors[index]
    return 'background-color: rgb({color[0]},{color[1]},{color[2]})'.format(color=color)

class VideoTransformer(VideoTransformerBase):
    """Компонент для создания стрима веб камеры"""
    def __init__(self):
        self.model = model
        self.rgb_colors = rgb_colors
        self.target_class_ids = target_class_ids

    def get_preds(self, img : np.ndarray) -> np.ndarray:
        return self.model([img]).xyxy[0].numpy()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.get_preds(img)
        result = result[np.isin(result[:,-1], self.target_class_ids)]
        
        for bbox_data in result:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img = cv2.rectangle(img, 
                                    p0, p1, 
                                    self.rgb_colors[label], 2) 

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#endregion


#region Load model
# ---------------------------------------------------

model_type = st.sidebar.selectbox(
    'Select model type',
    ('s', 'm', 'l', 'x'),
    index=1,
    format_func=lambda s: s.upper())

with st.spinner('Loading the model...'):
    model = get_yolo5(model_type)
st.success('Loading the model.. Done!')
#endregion


# UI elements
# ----------------------------------------------------

#sidebar
prediction_mode = st.sidebar.radio(
    "",
    ('Single image', 'Web camera'),
    index=0)
    
classes_selector = st.sidebar.multiselect('Select classes', 
                                        CLASSES, default='person')
all_labels_chbox = st.sidebar.checkbox('All classes', value=False)


# Prediction section
# ---------------------------------------------------------

#target labels and their colors
#target_class_ids - индексы выбранных классов согласно списку классов MS COCC
#rgb_colors - rgb-цвета для выбранных классов
if all_labels_chbox:
    target_class_ids = list(range(len(CLASSES)))
elif classes_selector:
    target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
else:
    target_class_ids = [0]

rgb_colors = get_colors(target_class_ids)
detected_ids = None


if prediction_mode == 'Single image':

    # добавляет форму для загрузки изображения
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['png', 'jpg', 'jpeg'])

    # если файл загружен
    if uploaded_file is not None:

        # конвертация изображения из bytes в np.ndarray
        bytes_data = uploaded_file.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img)

        #скопируем результаты работы кэшируемой функции, чтобы не изменить кэш
        result_copy = result.copy()
        #отберем только объекты нужных классов
        result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
        

        detected_ids = []
        #также скопируем изображение, чтобы не изменить аргумент кэшируемой 
        # функции get_preds
        img_draw = img.copy().astype(np.uint8)
        # нарисуем боксы для всех найденных целевых объектов
        for bbox_data in result_copy:
            xmin, ymin, xmax, ymax, _, label = bbox_data
            p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
            img_draw = cv2.rectangle(img_draw, 
                                    p0, p1, 
                                    rgb_colors[label], 2) 
            detected_ids.append(label)
        
        # выведем изображение с нарисованными боксами
        # use_column_width растянет изображение по ширине центральной колонки
        st.image(img_draw, use_column_width=True)

elif prediction_mode == 'Web camera':
    
    # создаем объект для вывода стрима с камеры
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

# выведем список найденных классов при работе с изображением или список всех
# выбранных классов при работе с видео
detected_ids = set(detected_ids if detected_ids is not None else target_class_ids)
labels = [CLASSES[index] for index in detected_ids]
legend_df = pd.DataFrame({'label': labels})
st.dataframe(legend_df.style.applymap(get_legend_color))
