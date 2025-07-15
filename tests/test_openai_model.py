import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para que las importaciones funcionen
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from SEJO_SDK.models.model_openai import OpenAIModel

def test_openai_model():
    # Cargar variables de entorno
    load_dotenv()
    
    # Obtener la clave API de las variables de entorno
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: No se encontró OPENAI_API_KEY en las variables de entorno")
        return
    
    try:
        # Inicializar el modelo OpenAI
        print("Inicializando modelo OpenAI...")
        model = OpenAIModel(
            api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
        
        # Probar el envío de un prompt
        print("\nEnviando prompt...")
        response = model.send_prompt("Dime un hecho interesante sobre la inteligencia artificial")
        print("\nRespuesta del modelo:")
        print(response)
        
        # Probar el streaming de respuesta
        print("\nProbando streaming de respuesta...")
        print("Respuesta en stream:")
        for chunk in model.stream_response("Explica en una oración qué es el machine learning"):
            print(chunk, end="", flush=True)
        print("\n")
        
        print("\n¡Prueba completada con éxito!")
        
    except Exception as e:
        print(f"\nError al probar el modelo: {str(e)}")

if __name__ == "__main__":
    test_openai_model()
