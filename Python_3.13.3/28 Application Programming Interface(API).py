"""
Interfaz de Programación de Aplicaciones (API)

API

API significa Interfaz de Programación de Aplicaciones. El tipo de API que abordaremos en esta sección serán las API web. Las API web son las interfaces definidas a través de las cuales se producen las interacciones entre una empresa y las aplicaciones que utilizan sus activos, constituyendo también un Acuerdo de Nivel de Servicio (SLA) para especificar las funciones del proveedor y exponer la ruta de servicio o URL a los usuarios de la API.

En el contexto del desarrollo web, una API se define como un conjunto de especificaciones, como los mensajes de solicitud del Protocolo de Transferencia de Hipertexto (HTTP), junto con una definición de la estructura de los mensajes de respuesta, normalmente en un formato XML o en Notación de Objetos JavaScript (JSON).

Las API web se han alejado de los servicios web basados en el Protocolo Simple de Acceso a Objetos (SOAP) y la arquitectura orientada a servicios (SOA) para acercarse más directamente a los recursos web en un estilo de transferencia de estado representacional (REST).

En el ámbito de los servicios de medios sociales, las API web han permitido a las comunidades web compartir contenidos y datos entre distintas comunidades y plataformas.

Gracias a las API, los contenidos creados en un lugar pueden publicarse y actualizarse dinámicamente en varios sitios web.

Por ejemplo, la API REST de Twitter permite a los desarrolladores acceder a los datos básicos de Twitter, y la API de búsqueda proporciona métodos para que los desarrolladores interactúen con los datos de búsqueda y tendencias de Twitter.

Muchas aplicaciones proporcionan puntos finales de API. Algunos ejemplos de API son la API de países o la API de razas de gatos.

En esta sección, cubriremos una API RESTful que utiliza métodos de solicitud HTTP para obtener datos mediante GET, actualizar mediante PUT, crear mediante POST y eliminar mediante DELETE.

Conceptos básicos de la API

Una API (interfaz de programación de aplicaciones) permite que dos sistemas informáticos interactúen entre sí. Por ejemplo, si creamos una automatización que genera un informe y lo envía por correo electrónico, el envío de ese correo electrónico no se hace manualmente, lo hará el propio script. Para ello, Python (o el lenguaje que utilicemos), debe pedir a Gmail que envíe ese correo electrónico, con ese informe adjunto a determinadas personas. La forma de hacerlo es a través de una API, en este caso la API de Gmail.

Bien, ahora que ya sabes qué es una API, veamos cuáles son las partes principales de una API:

Protocolo de transferencia HTTP: es la principal forma de comunicar información en la web. Existen diferentes métodos, cada uno de ellos utilizado para diferentes cuestiones:

  GET: este método permite obtener información de la base de datos o de un proceso.
  POST: permite enviar información, ya sea para añadir información a una base de datos o para pasar la entrada de un modelo de machine learning, por ejemplo.
  PUT: actualizar información. Generalmente se utiliza para gestionar la información de la base de datos.
  DELETE: este método se utiliza para eliminar información de la base de datos.

Url: es la dirección donde podemos encontrar nuestra API. Básicamente, esta URL constará de tres partes:

  Protocolo: como cualquier dirección, puede ser o .http://https://
  Dominio: el host en el que está alojado, que va desde el protocolo hasta el final de .com, o lo que sea que tenga la url. En mi sitio web, por ejemplo, el dominio es .anderfernandez.com
  Endpoint: al igual que un sitio web tiene varias páginas (/blog), (/legal), una misma API puede incluir múltiples puntos y cada uno hace cosas diferentes. Al crear nuestra API en Python indicaremos los endpoints, por lo que debemos asegurarnos de que cada enpoint sea representativo de lo que hace la API que hay detrás.

Cómo crear una API en Python

Existen diferentes formas de crear una API en Python, siendo las más utilizadas FastAPI y Flask. Así pues, te explicaré cómo funcionan ambas, para que puedas utilizar la forma de crear APIs en Python que más te guste. Comencemos con FastAPI.

Cómo crear una API en Python con FastAPI

Requisitos para usar FastAPI

FastAPI es una forma de crear APIs en Python que salió a finales de 2018. Es muy rápido, aunque solo se puede usar con Python 3.6+

Para usarlo, debe instalar dos bibliotecas: y .fastapiuvicorn

pip install fastapi
pip install uvicorn

Ahora que ya tenemos instalados los paquetes, simplemente tenemos que crear un fichero en Python donde definiremos nuestra API. En este archivo, debemos crear una app, donde incluiremos las APIs, con sus endpoints, parámetros, etc.

Una vez que tenemos la aplicación, ahí es donde definimos la información que requiere la API: endpoint, método HTTP, argumentos de entrada y qué hará la API detrás de él.
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/my-first-api")
def hello():
  return {"Hello world!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
"""
Una vez se abre la pagina web http://localhost:8000/my-first-api, podemos ver el mensaje {"Hello world!"}

- `from fastapi import FastAPI`: Esto importa la clase `FastAPI` del módulo `fastapi`. Necesitas una instancia de esta clase para crear tu aplicación.

- `import uvicorn`: Uvicorn es un servidor ASGI que puede servir tu aplicación FastAPI. Lo estás importando para poder ejecutar tu aplicación más tarde.

- `app = FastAPI()`: Esto crea una instancia de la clase `FastAPI`. Esta es tu aplicación.

- `@app.get("/my-first-api")`: Esto es un decorador que le dice a FastAPI que la función a continuación debe ser llamada cuando el servidor recibe una solicitud GET a la ruta "/my-first-api".

- `def hello():`: Esta es la función que se llama cuando se recibe una solicitud GET a la ruta "/my-first-api". Puede devolver datos que se convierten automáticamente en JSON.

- `return {"Hello world!"}`: Esto es lo que devuelve la función cuando se llama. En este caso, está devolviendo un diccionario con una única clave y valor. FastAPI convertirá automáticamente este diccionario en una respuesta JSON.

- `if __name__ == "__main__":`: Esta línea asegura que el servidor sólo se ejecuta si este script se ejecuta directamente (es decir, no cuando se importa como un módulo).

- `uvicorn.run(app, host="0.0.0.0", port=8000)`: Esto inicia el servidor Uvicorn para servir tu aplicación. Está configurado para escuchar en todas las interfaces de red (`0.0.0.0`) en el puerto 8000.

Por lo tanto, si ejecutas este código y visitas http://localhost:8000/my-first-api en tu navegador (o haces una solicitud GET a esa URL con una herramienta como curl o Postman), verás una respuesta que es un documento JSON que se ve así: `{"Hello world!"}`.

Con esto ya tendríamos creada una API muy sencilla, que simplemente devuelve “¡Hola mundo!”. Como verás, en unas pocas líneas hemos definido: el método (get), el endpoint (“/”) y la función que debe ejecutar esta API.

Incluso podríamos pasar argumentos a nuestra API, para que los utilice en su función. Siempre que pasemos un argumento a nuestra función debemos indicar el tipo de dato que debe ser (número, texto, etc.).

Importante : FastAPI realiza una comprobación de que el tipo de datos que le pasamos en la llamada es el que le hemos indicado que debe ser. Esto es esencial para garantizar que nuestra API funcione correctamente y es algo que otros marcos de creación de API (como Flask) no incluyen.

Veamos cómo funciona en un ejemplo:
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/my-first-api")
def hello(name: str):
  return {'Hello ' + name + '!'} 

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
Ahora, cuando realicemos la solicitud a esta API, tendremos que pasarle el parámetro de nombre para que funcione. Es decir, si antes nos bastaba con ir a http://127.0.0.1:8000/my-first-api, ahora tendremos que pasar el nameparámetro. Por lo tanto, la solicitud tendrá el siguiente aspecto: http://127.0.0.1:8000/my-first-api?name=Alvaro y el resultado sera: ["Hello Alvaro!"]

Como hemos incluido el nameargumento, este argumento es obligatorio: si no se incluye en la solicitud, no funcionará. Sin embargo, es posible que queramos pasar argumentos opcionales.

Vamos a crear una API a la que le puedes pasar, o no, la variable name. Si lo pasa, devolverá "¡Hola {nombre}!" y, si no, simplemente “¡Hola!”.
"""


from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/my-first-api")
def hello(name = None):

    if name is None:
        text = 'Hello!'

    else:
        text = 'Hello ' + name + '!'

    return text

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
En este caso si vamos a http://127.0.0.1:8000/my-first-apila API se ejecutará correctamente y devolverá “¡Hola!”, mientras que si le pasamos el parámetro, por ejemplo, http://127.0.0.1:8000/my-first-api?name=Alvaro lo utilizará y seguirá funcionando correctamente, devolviendo Hello Alvaro!.

Devuelve diferentes tipos de datos con FastAPI: La gran mayoría de veces una API suele devolver texto (una predicción, datos, etc.), aunque muchas veces puede devolver otro tipo de datos, como un DataFrame o una imagen, por ejemplo.

Cuando se trata de objetos "normales", como un DataFrame, FastAPI lo convertirá directamente en un archivo JSON. Ejemplo:
"""

from fastapi import FastAPI
import uvicorn
import pandas as pd

app = FastAPI()

@app.get("/get-iris")
def get_iris():
    url ='https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    iris = pd.read_csv(url)

    # Convertir el DataFrame a una lista de diccionarios
    return iris.to_dict('records')

print(get_iris())

"""
[{'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2, 'species': 'setosa'}, {'sepal_length': 4.9, 'sepal_width': 3.0, 'petal_length': 1.4, 'petal_width': 0.2, 'species': 'setosa'},.....

Como puede ver, Fast API convierte el marco de datos directamente en un objeto JSON. Sin embargo, ¿qué pasa con las imágenes o los vídeos?

FastAPI se basa en starlette, por lo que para responder con imágenes o íconos podemos usar ambos StreamingResponsey FileResponse. En cualquier caso, necesitaremos instalar aiofiles.

Entonces, si queremos mostrar una imagen en nuestra API, nuestra aplicación FastAPI se verá así:
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pandas as pd
import matplotlib.pyplot as plt
import uvicorn

app = FastAPI()

@app.get("/plot-iris")
def plot_iris():
    url ='https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    iris = pd.read_csv(url)

    plt.scatter(iris['sepal_length'], iris['sepal_width'])
    plt.savefig('iris.png')
    file = open('iris.png', mode="rb")

    return StreamingResponse(file, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)