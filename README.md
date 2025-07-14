# SEJO SDK

<p align="center">
  <img src="SEJO_SDK/logo/Sejo_logo.png" alt="SEJO SDK Logo" width="300">
</p>

## 🚀 Descripción

SEJO SDK es una biblioteca Python moderna y versátil que proporciona una interfaz unificada para trabajar con diferentes modelos de lenguaje AI. Con SEJO SDK, puedes interactuar con OpenAI, Anthropic, Google Gemini y DeepSeek de manera sencilla y consistente.

## 🎯 Características Principales

- 🔒 Interfaz unificada para múltiples proveedores de IA
- 🤖 Implementación de agentes inteligentes
- 🧠 Sistema de memoria contextual
- 🛠️ Soporte para herramientas integradas
- 🔧 Configurable y extensible
- 📚 Documentación completa

## 📦 Requisitos

- Python 3.8 o superior
- Dependencias específicas para cada modelo (OpenAI, Anthropic, etc.)

## 📋 Instalación

```bash
pip install sejo-sdk
```

## 📚 Uso Básico

```python
from SEJO_SDK.model import OpenAIModel
from SEJO_SDK.agent import Agent

# Configurar el modelo
model = OpenAIModel(
    api_key="tu_api_key",
    model_name="gpt-4"
)

# Crear un agente
agent = Agent(model=model)

# Ejecutar una consulta
response = agent.run("¿Cuál es la capital de España?")
```

## 📁 Estructura del Proyecto

```
SEJO_SDK/
├── models/           # Implementaciones de diferentes modelos
│   ├── model_openai.py
│   ├── model_anthropic.py
│   ├── model_gemini.py
│   └── model_deepseek.py
├── agent.py          # Implementación del agente
├── model.py          # Clase base Model
└── memory.py         # Sistema de memoria
```

## 🔧 Contribución

¡Nos encantaría que contribuyas a SEJO SDK! Para hacerlo:

1. Haz un Fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## 👥 Contacto

- GitHub: [jmiguelmangas](https://github.com/jmiguelmangas)
- Email: jmmangas@gmail.com

## 🙏 Agradecimientos

- A todos los contribuidores que han ayudado a hacer este proyecto mejor
- A la comunidad open source por su apoyo continuo

---

Hecho con ❤️ por [Jose Miguel Mangas](https://github.com/jmiguelmangas)