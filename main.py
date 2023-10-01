import streamlit as st
from PIL import Image
import yadisk
import streamlit.components.v1 as components
from urllib.parse import parse_qs, urlparse
from streamlit_javascript import st_javascript
from model import get_model, tensor_from_images, paint_boxes
from PIL import Image
import requests
from io import BytesIO
import torch
import posixpath
import os
from datetime import datetime
from streamlit.runtime.uploaded_file_manager import UploadedFile


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
      display: flex;   
      justify-content: center;  
      align-items: center;        
   }
   .container {
      display: flex;
      justify-content: space-between;
      align-items: center;
   }

                   
</style>
<script src="https://yastatic.net/s3/passport-sdk/autofill/v1/sdk-suggest-with-polyfills-latest.js"></script>
</head>

<body>
   <div class="container">
      <div id="container"></div>
      <img src="https://psv4.userapi.com/c240331/u133344394/docs/d18/1f8343f91ade/logo_black.png?extra=vLepWcw-ZEFYwyCRYqp2G2fPAHQvbcjGx_gtYJ5eA9BumkBAJqY9dm_HfUF9VbKsY5JVHIy1XE6o8wpKZ9RGzT1TFmbcYoaYUEivZVtPq_omFIkJrn9VVRlMhWGrsaSlGeyHe9dh9n5qym7b2f6WppqT" width="150px" height="150px">
   </div>
              
   <script>
   window.onload = function() {
      
      window.YaAuthSuggest.init({
                  client_id: 'd555e48d5d224263bee0dce3d549f296',
                  response_type: 'token',
                  redirect_uri: 'https://9b78-95-54-231-132.ngrok-free.app?route=token'
               },
               'https://9b78-95-54-231-132.ngrok-free.app?route=token', {
                  view: 'button',
                  parentId: 'container',
                  buttonView: 'main',
                  buttonTheme: 'light',
                  buttonSize: 'xxl',
                  buttonBorderRadius: 18
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
   boxes_checkbox = st.checkbox('Детекция животного', value=True)
   uploaded_files: list[UploadedFile] = st.file_uploader("Загрузите файлы", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

   results:list[torch.Tensor] = []
   for uploaded_file in uploaded_files:
      # print(uploaded_file)
      image = Image.open(uploaded_file)
      results.append(model(tensor_from_images(image)))
   
   map = {0: 'no_animal', 1:'animal', 2: 'broken'}

   current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   output_dir = f"output_{current_datetime}"
   class_dirs = {0: 'no_animal', 1:'animal', 2: 'broken'}
   
   # for class_dir in class_dirs.values():
         # os.makedirs(os.path.join(output_dir, class_dir), exist_ok=True)
   
   for image_file, result in zip(uploaded_files, results):
      # response = requests.get(image_file.get_download_link())
      # image = Image.open(BytesIO(response.content))
      class_indx = result.squeeze(0).detach().cpu().argmax().item()
      class_name = class_dirs[class_indx]
      print(class_name)
      st.markdown(f"<h1 style='text-align: center; color: black; font-size: 200%;'>{class_name}</h1>", unsafe_allow_html=True)

      if boxes_checkbox:
         image_boxes = paint_boxes(image_file)
         st.image(image_boxes, caption=image_file.name, use_column_width=True)
      else:
         st.image(image_file, caption=image_file.name, use_column_width=True)
            
      # image.save(os.path.join(output_dir, class_name, image_file.name))

elif route == 'token':

   y: yadisk.YaDisk = yadisk.YaDisk(token=access_token)

   folder_absolute_path = st.text_input("Введите название папки на диске через '/' (например, /photo):")
   boxes_checkbox = st.checkbox('Детекция животного')
   
   if folder_absolute_path:
      st.write("Начался процесс обработки файлов...")
      try:
         resources = y.listdir(folder_absolute_path)

         image_files: list[yadisk.objects.resources.ResourceObject] = [resource for resource in resources if resource.type == "file" and (resource.name.lower().endswith(".jpg") or resource.name.lower().endswith(".png"))]

         print('.............')
         results:list[torch.Tensor] = []
         for image in image_files:
            response = requests.get(image.get_download_link())
            image = Image.open(BytesIO(response.content))
            results.append(model(tensor_from_images(image)))

         print(len(results))

         map = {0: 'no_animal', 1:'animal', 2: 'broken'}
         for result in results:
            class_indx = result.squeeze(0).detach().cpu().argmax().item()
            class_name = map[class_indx]
            print(class_name)


         current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
         output_dir = f"output_{current_datetime}"
         class_dirs = {0: 'no_animal', 1:'animal', 2: 'broken'}
        
         for class_dir in class_dirs.values():
             os.makedirs(os.path.join(output_dir, class_dir), exist_ok=True)
        
         for image_file, result in zip(image_files, results):
            response = requests.get(image_file.get_download_link())
            image = Image.open(BytesIO(response.content))
            class_indx = result.squeeze(0).detach().cpu().argmax().item()
            class_name = class_dirs[class_indx]
            print(class_name)

            if boxes_checkbox:
               image_boxes = Image.fromarray(paint_boxes(BytesIO(response.content)))
               image_boxes.save(os.path.join(output_dir, class_name, image_file.name))
            else:
               image.save(os.path.join(output_dir, class_name, image_file.name))


         def mkdir_p(y, path):
            head, tail = posixpath.split(path)
            if not tail:
                head, tail = posixpath.split(head)
            if head and tail and not y.exists(head): 
                try:
                    y.mkdir(head) 
                except yadisk.exceptions.DirectoryExistsError:
                    pass
                mkdir_p(y, head)
            if tail: 
                try:
                    y.mkdir(path)
                except yadisk.exceptions.DirectoryExistsError:
                    pass

         st.write("Начался процесс выгрузки файлов на Яндекс Диск...")

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

         st.write(f"Проверьте ваш Диск, все фото загружены в новую папку output_{current_datetime}")
    

      except Exception as e:
         st.write(f"Произошла ошибка: {str(e)}")
