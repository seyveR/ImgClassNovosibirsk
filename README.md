# Кейс классификация снимков с фотоловушек

В нашей стране большое количество заповедных мест. Многие из них труднодоступны или совсемзакрыты для туристов. В заповедниках живут разные виды животных, за которыми ученые наблюдают через фотоловушки: высчитывают популяцию и разные другие важные показатели.

Проблема заключается в невозможности контролировать качество каждого снимка фотоловушки. Т.к. количество снимков с каждой из них насчитывает несколько тысяч в год, просматривать каждую фотографию проблематично.

Участникам предлагается разработать программный модуль с использованием технологий искусственного интеллекта, позволяющий классифицировать фотографии на качественные и некачественные. В результате создания такого решения существенно сократится количество времени, затрачиваемое на анализ полученных снимков.


## Команда: Chikie$

**Краткое описание**: Модель классифицирует изобаржение к одному из трех классов (некачественное, с животным на изображении и без животного на изображении)

Стэк: Python, ResNext101, Yolov8

Для работы модели необходимо скачать веса модели по ссылке **https://disk.yandex.ru/d/ICYs-vj1BhLvTQ** и поместить их в корень проекта

Для локального запуска с сервером необходимо клонировать проект, а также запустить следующие команды (необходимо наличие зависимостей из requirements.txt):

```
    git clone https://github.com/seyveR/ImgClassNovosibirsk
    streamlit run main.py
```