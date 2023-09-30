import streamlit as st
from PIL import Image
import yadisk
import streamlit.components.v1 as components
from urllib.parse import parse_qs, urlparse
from streamlit_javascript import st_javascript
from model import get_model, tensor_from_images
from PIL import Image
import requests
from io import BytesIO
import torch
import posixpath
from datetime import datetime
import gc

model = get_model()

st.set_page_config(
    page_title="Chikies",
    page_icon=":camera:",
    layout="wide"
)
route = st.experimental_get_query_params().get('route')[0]
# print(route)

url = st_javascript("await fetch('').then(r => window.parent.location.href)")
parsed_url = urlparse(url)
query_params = parse_qs(parsed_url.fragment)
access_token = query_params.get('access_token', [''])[0]
# print("access_token:", access_token)

if route == 'home':
   components.html("""
<!doctype html>
<html lang="ru">

<head>
<meta charSet="utf-8" />
<meta name='viewport' content='width=device-width, initial-scale=1, maximum-scale=1, minimum-scale=1, shrink-to-fit=no, viewport-fit=cover'>
<meta http-equiv='X-UA-Compatible' content='ie=edge'>
<style>
   html,
   body {
      background: #eee;
   }
</style>
<script src="https://yastatic.net/s3/passport-sdk/autofill/v1/sdk-suggest-with-polyfills-latest.js"></script>
</head>

<body>
   <script>
   window.onload = function() {
      
      window.YaAuthSuggest.init({
                  client_id: '57fe3796c81246f6b32881612ef3fd22',
                  response_type: 'token',
                  redirect_uri: 'https://d2cb-95-53-28-234.ngrok-free.app?route=token'
               },
               'https://d2cb-95-53-28-234.ngrok-free.app?route=token', {
                  view: 'button',
                  parentId: 'container',
                  buttonView: 'main',
                  buttonTheme: 'light',
                  buttonSize: 'm',
                  buttonBorderRadius: 0
               }
            )
            .then(function(result) {
               return result.handler()
            })
            .then(function(data) {
               console.log('Сообщение с токеном: ', data);
               document.body.innerHTML += `Сообщение с токеном: ${JSON.stringify(data)}`;
            })
            .catch(function(error) {
               console.log('Что-то пошло не так: ', error);
               document.body.innerHTML += `Что-то пошло не так: ${JSON.stringify(error)}`;
            });
      };
   </script>
</body>

</html>
                   
""")

elif route == 'token':

   y: yadisk.YaDisk = yadisk.YaDisk(token=access_token)

   folder_absolute_path = st.text_input("Введите название папки на диске через '/' (например, /photo):")

    # Если пользователь ввел абсолютный путь к папке, то поиск файлов
   if folder_absolute_path:
      try:
         resources = y.listdir(folder_absolute_path)

         image_files: list[yadisk.objects.resources.ResourceObject] = [resource for resource in resources if resource.type == "file" and (resource.name.lower().endswith(".jpg") or resource.name.lower().endswith(".png"))]

         print('.............')
         results:list[torch.Tensor] = []
         for image in image_files:
            response = requests.get(image.get_download_link())
            image = Image.open(BytesIO(response.content))
            results.append(model(tensor_from_images(image)))
            
            gc.collect()

         print(len(results))
         # print(results)

         map = {0: 'animal', 1:'no_animal', 2: 'broken'}
         for result in results:
            class_indx = result.squeeze(0).detach().cpu().argmax().item()
            class_name = map[class_indx]
            print(class_name)
         import os

         current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
         output_dir = f"output_{current_datetime}"
         class_dirs = {0: 'animal', 1:'no_animal', 2: 'broken'}
        
        # Создание папок для каждого класса, если они еще не созданы
         for class_dir in class_dirs.values():
             os.makedirs(os.path.join(output_dir, class_dir), exist_ok=True)
        
         for image_file, result in zip(image_files, results):
             response = requests.get(image_file.get_download_link())
             image = Image.open(BytesIO(response.content))
             class_indx = result.squeeze(0).detach().cpu().argmax().item()
             class_name = class_dirs[class_indx]
             print(class_name)
                   
             image.save(os.path.join(output_dir, class_name, image_file.name))

         def mkdir_p(y, path):
            head, tail = posixpath.split(path)
            if not tail:
                head, tail = posixpath.split(head)
            if head and tail and not y.exists(head):  # рекурсивно создаем все промежуточные папки
                try:
                    y.mkdir(head)  # создаем папку и игнорируем ошибку, если папка уже существует
                except yadisk.exceptions.DirectoryExistsError:
                    pass
                mkdir_p(y, head)
            if tail:  # создаем последнюю папку в пути
                try:
                    y.mkdir(path)
                except yadisk.exceptions.DirectoryExistsError:
                    pass

         

         for root, dirs, files in os.walk(output_dir):
            for dir in dirs:
                local_dir_path = os.path.join(root, dir)
                remote_dir_path = posixpath.join(f"/output_{current_datetime}", os.path.relpath(local_dir_path, output_dir).replace(os.path.sep, '/'))
                if not y.exists(remote_dir_path):
                    print(f"Creating directory {remote_dir_path}...")
                    mkdir_p(y, remote_dir_path)
            for file in files:
                local_file_path = os.path.join(root, file)
                remote_file_path = posixpath.join(f"/output_{current_datetime}", os.path.relpath(local_file_path, output_dir).replace(os.path.sep, '/'))
                print(f"Uploading {local_file_path} to {remote_file_path}...")
                y.upload(local_file_path, remote_file_path)
    

      except Exception as e:
         st.write(f"Произошла ошибка: {str(e)}")
