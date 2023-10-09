import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    image = Image.fromarray(image.astype('uint8'))

    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="PNG")
    img_binary = img_byte_array.getvalue()
    
    # Envoyer l'image à l'API pour la prédiction
    api_url = "http://localhost:5000/predict"  # Remplacez par l'URL de l'API que vous utilisez
    response = requests.post(api_url, data=img_binary)

    if response.status_code == 200:
        # Obtenez la prédiction à partir de la réponse de l'API (c'est une hypothèse, cela dépendra de l'API que vous utilisez)
        prediction = response.json()["prediction"] # Assurez-vous d'utiliser la méthode de désérialisation appropriée ici
        
        # Retournez la prédiction
        return prediction
    else:
        print("Échec de la requête à l'API")
        return None

if __name__=='__main__':

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=True);