import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import cv2
import requests
from io import BytesIO

def get_palette(image_path, n_colors):
    # Leer la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen de BGR a RGB (OpenCV carga imágenes en formato BGR por defecto)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convertir la imagen en un arreglo 1D
    pixels = image_rgb.reshape(-1, 3)
    
    # Realizar k-means para agrupar los colores
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)
    
    # Obtener los colores de los centroides de los grupos
    colors = kmeans.cluster_centers_
    
    # Convertir los valores de los colores a enteros
    colors = colors.astype(int)
    
    # Convertir los colores de 1D a 3D
    palette = colors.reshape(1, -1, 3)
    
    # Obtener las etiquetas de los grupos para cada pixel
    labels = kmeans.labels_

    # Contar la frecuencia de cada etiqueta (cada grupo)
    counts = np.bincount(labels)

    # Calcular la frecuencia relativa de cada etiqueta (porcentaje)
    percentages = counts / len(labels)

    # Crear la lista de colores y porcentajes para la paleta
    palette_with_percentages = []
    for i in range(n_colors):
        color = colors[i].astype(int)
        percentage = percentages[i] * 100
        palette_with_percentages.append((color, percentage))

    return palette_with_percentages

@st.cache_data
def load_image_from_link(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return image


def show_palette(palette_with_percentages):
    
    # Calcular las longitudes de las barras según los porcentajes
    total_width = 705
    bar_height = 50
    bar_lengths = [percentage * total_width / 100 for _, percentage in palette_with_percentages]

    # Dibujar la barra de colores según los porcentajes
    colors_html = ""
    for i, (color, percentage) in enumerate(palette_with_percentages):
        color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        colors_html += f'<div style="background-color:{color_hex}; width:{bar_lengths[i]}px; height:{bar_height}px; float:left;"></div>'

    st.markdown(
        f'<div style="background-color:#f0f0f0; width:{total_width}px; height:{bar_height}px;">{colors_html}</div>',
        unsafe_allow_html=True
    )


def main():
    st.title("Image color palette")

    # Opciones para cargar la imagen
    option = st.radio("",options= ["Upload from device", "Load from link"])

    if option == "Upload from device":
        # Cargar una imagen desde el dispositivo
        uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
    else:
        # Cargar una imagen desde un link
        image_url = st.text_input("Image link:")
        if image_url:
            try:
                image = load_image_from_link(image_url)
            except:
                st.error("Could not load image from link.")

    if 'image' in locals():
        # Obtener el número de colores deseado
        n_colors = st.slider("Number of colors", min_value=1, max_value=10, value=5, step=1)

        # Obtener la paleta de colores usando la función get_palette
        image_path = "temp_image.jpg"
        image.save(image_path)  # Guardar la imagen temporalmente para procesarla
        palette_with_percentages = get_palette(image_path, n_colors)
        # st.subheader("Paleta de colores:")

        show_palette(palette_with_percentages)

        # Mostrar la imagen cargada
        st.image(image, use_column_width="auto")
        

if __name__ == "__main__":
    main()
