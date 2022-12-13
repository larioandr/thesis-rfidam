import locale
import matplotlib
import matplotlib.pyplot as plt
import os
import platform
from typing import Union, Sequence


CMAP_NAME = 'inferno'  # Цветовая схема для графиков
LIGHT_CMAP_NAME = "Pastel2"
SCATTER_CMAP_NAME = "flare"
IMAGE_BASE_DIR = "images"  # Здесь будут храниться все построенные изображения
IMAGE_EXTENSIONS = ("pdf", "png")  # В каких форматах сохранять изображения

DARK_CMAP = matplotlib.cm.get_cmap(CMAP_NAME)
LIGHT_CMAP = matplotlib.cm.get_cmap(LIGHT_CMAP_NAME)
BW_CMAP = matplotlib.cm.get_cmap('Greys')

def setup_matplotlib() -> None:
    """
    Настроить параметры matplotlib и локали.
    """
    matplotlib.rcParams.update({
        'image.cmap': CMAP_NAME,
        'axes.formatter.use_locale': True,
        'font.size': 16,
        'font.family': 'sans-serif',
        'axes.titlesize': 'medium',
        'figure.titlesize': 'medium',

        # Шрифт PT Serif Caption можно установить с Google Fonts.
        # После установки шрифта нужно удалить кэш matplitlib,
        # на Ubuntu: ~/.cache/matplotlib 
        'font.sans-serif': ['PT Serif Caption',],
    })

    locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')

    
def get_color(x: float):
    """
    Получить цвет из текущей карты.
    
    Args:
        x (float): число от 0 до 1
    
    Returns:
        color
    """
    return DARK_CMAP(x)


def get_bw_color(x: float):
    """
    Получить цвет из текущей карты.
    
    Args:
        x (float): число от 0 до 1
    
    Returns:
        color
    """
    return BW_CMAP(x)


def savefig(name: str, exts: Sequence[str] = IMAGE_EXTENSIONS,
            directory: str = IMAGE_BASE_DIR) -> None:
    """
    Сохранить изображение в файлы с общим именем и разными расширениями.
    Изображения будут сохранены в папку directory.
    
    Если name - пустая строка, ничего сохраняться не будет.
    
    Args:
        name (str): название файла без расширения
        exts (list of str): набор расширений, по-умолчанию png и pdf
        directory (str): куда сохранить, по-умолчанию IMAGE_BASE_DIR
    """
    if name is None:
        return
    name = name.strip()
    if name:
        for ext in exts:
            file_path = os.path.join(directory, f"{name}.{ext}")
            plt.savefig(file_path, bbox_inches="tight")


def set_axes_formatter(
        *axes, 
        use_x: bool = False, 
        use_y: bool = False, 
        platforms: Sequence[str] = ("Linux",)
) -> None:
    """
    У некоторых нормальных шрифтов нет глифов, которые нужны
    в русской локали. Поэтому стараемся избежать ситуации,
    когда эти глифы нужны.
    
    Сейчас обходим глиф-пробел 8239 в формате больших целых.
    
    Args:
        axes (list of Axes): графики, к которым применить
        use_x (bool): применять ли к OX (False)
        use_y (bool): применять ли к OY (False)
        platforms (list of str): на каких платформах надо применять ("Linux")
    """
    def fmt(value, tick_num):
        if isinstance(value, int):
            return f"{value:g}"
        elif isinstance(value, float):
            if int(value) == value:
                return f"{value:g}"
            else:
                return f"{value:n}"
        return str(value)
    
    if not platforms or platform.system() in platforms:
        for axes_ in axes:
            if not (isinstance(axes_, list) or isinstance(axes_, tuple)):
                axes_ = (axes_,)
            for ax in axes_:
                if use_y:
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
                if use_x:
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt))
