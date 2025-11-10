"""
Python para Web
Python es un lenguaje de programación de propósito general y se puede utilizar en muchos lugares. En esta sección, veremos cómo usamos Python para la web. Hay muchos trabajos de marcos web de Python. Django y Flask son los más populares. Hoy veremos cómo utilizar Flask para el desarrollo web.

Flask
Flask es un marco de desarrollo web escrito en Python. Flask utiliza el motor de plantillas Jinja2. Flask también se puede utilizar con otras bibliotecas frontales modernas como React.

Estructura de carpetas
Después de completar todos los pasos, la estructura de archivos de su proyecto debería verse así:

python_for_web
├── app.py
├── venv
│   ├── Scripts
│   ├── Lib
│   ├── .gitignore
│   ├── pyvenv.cfg
├── static
│   └── css
│       └── layaut.css
│   └── js
│       └── layaut.js
└── templates
    ├── index.html
    ├── layout.html
    ├── contacto.html
    ├── 404.html

Configurando el directorio de su proyecto
Siga los siguientes pasos para comenzar con Flask.

Paso 1: instale virtualenv usando el siguiente comando.

pip install virtualenv

Creamos un directorio de proyecto llamado "python_for_web". Dentro del proyecto creamos un entorno virtual venv y luego activamos el entorno virtual (Para crearlo y activarlo ir a las intrucciones en 23 Setting up virtual enviroments). Usamos pip frozen para verificar los paquetes instalados en el directorio del proyecto. El resultado de verificar los paquetes instalados debe estar vacio, porque no se ha instalado nada aun.

Nota: Podra observar el entorno virtual activo, cuando observe esto: (venv) PS C:\Users\alvan\OneDrive\Documentos\Python\Curso Python> 

Instalamos la libreria flask en el entorno activo de la siguiente manera, escribiendo y ejecutando en la terminal el siguiente codigo:

pip install flask

Ahora, creemos el archivo app.py en el directorio del proyecto "python_for_web" y le escribamos el siguiente código. 
"""

from flask import Flask
import os

app = Flask(__name__)

if __name__ == '__main__':
   app.run()

"""
Vamos al terminal y ejecutamos el siguiente codigo de la ubicacion del archivo app.py

python .\python_for_web\app.py

veremos la siguiente respuesta:

 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
127.0.0.1 - - [24/Oct/2023 11:29:49] "GET / HTTP/1.1" 404 -
 
Vamos al navegador y escribimos la siguiente direccion referida en el mensaje anterior: http://127.0.0.1:5000. Veremos la siguiente respuesta:

Not Found
The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.

Como vemos, despues de la ejecucion de python .\python_for_web\app.py vemos en la respuesta "debug mode: off", esto quiere eecir que no va a tomar los cambios que realicemos en el codigo, por tanto, debemos detener el servidor y volverlo a ejecutar, para que tome los cambios que realicemos en el codigo. Para detener el servidor, presionamos las teclas Ctrl + C. De esta manera en el terminal se detendra el servidor y podremos ejecutar mas comandos en la terminal.

La repuesta del servicor "Not found..." se debe a que no hemos creado ninguna ruta, por tanto, vamos a crear una ruta para la pagina de inicio, para ello, vamos al archivo app.py y complementamos el codigo:
"""

from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
   return '¡Hola Mundo!'

if __name__ == '__main__':
   app.run()

"""
Nuevamente ejecutamos python .\python_for_web\app.py, no sin antes actualizar el codigo en app.py presionando Ctrl + C.

Explicacion: 
@app.route('/'): Esto es un decorador que indica la ruta a la que se asocia la función index(). En este caso, la ruta es la raíz del dominio o http://127.0.0.1:5000 si se ejecuta en el servidor local.

def index(): Esta es la definición de la función index. Esta función se ejecuta cuando el usuario accede a la ruta especificada en el decorador. En este caso, cuando el usuario accede a la ruta principal de la aplicación.

return '¡Hola Mundo!': Esta línea devuelve una cadena de texto '¡Hola Mundo!' como respuesta a la solicitud del usuario. En otras palabras, cuando un usuario accede a la ruta principal de la aplicación, la función index() devuelve '¡Hola Mundo!' como respuesta y se muestra en el navegador.

Si vamos nuevamente al servidor "http://127.0.0.1:5000" observaremos el mensaje:

¡Hola Mundo!

Vamos al terminal y deteenmos el servidor presionando Ctrl + C.

Vamos activar el modo depuracion (debug mode: True), para que tome los cambios que realicemos en el codigo, para ello, vamos al archivo app.py y complementamos el codigo:
"""
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
   return '¡Hola Nicolas!'

if __name__ == '__main__':
   app.run(debug = True, port = 5000)
   
"""
No olvidemos actualizar los cambios en el archivo app.py presionando Ctrl + S.

Ya no sera necesario parar el servidor y volverlo a ejecutar, ya que el modo depuracion (debug mode: True) toma los cambios que realicemos en el codigo, por tanto, solo vamos al navegador y actualizamos la pagina.

Ahora cambiamos el formato de la etiqueta del mensaje, anexando <h1> al inicio y final del mensaje
"""
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
   return '<h1>¡Hola Nicolas!<h1>'

if __name__ == '__main__':
   app.run(debug = True, port = 5000)
   
"""
La renderización de plantillas: Se refiere al proceso de combinar datos con una plantilla predefinida para producir contenido web dinámico. En este caso la linea "return ..." se cambia por un archivo HTML, para ello, creamos un directorio llamado templates dentro de directorio "python_for_web y dentro de este creamos un archivo llamado index.html, con el siguiente codigo que corresponde a una estructura basica de html5 :

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Home</title>
  </head>

  <body>
    <h1>¡Hola Nicolas!<h1>
  </body>
</html>

Nota: flask va a reconocer el directorio templates, por tanto, no es necesario especificar la ruta del directorio templates. Modificamos el codigo en app.py, adicionando ", render_template" en la linea "from flask import Flask" para que la pagina corra en funcion del archivo index.html
"""

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True, port = 5000)

"""
Nuevamente observemos la pagina en el navegador, y veremos el mismo resultado que en el paso anterior, pero ahora la pagina corre en funcion del archivo index.html.

Hora adicionemos el diccionario data y el parametro data=data en la linea "return render_template('index.html')". 
"""

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    data = {'titulo': 'Index', 'Bienvenida': '¡Saludos!'}
    return render_template('index.html', data=data)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)

"""
Ahora, vamos a modificar el archivo index.html en el directorio templates, adicionando juego de llave de inicio "<h1>{{data}}<h1>" (Motor de plantillas Jinja 2).

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Home</title>
  </head>

  <body>
    <h1>{{data}}<h1>
  </body>
</html>

Nuevamente observemos la pagina en el navegador, recuerde actualizar codigo con Ctrl + S, veremos el siguiente resultado:

{'titulo': 'Index', 'Bienvenida': '¡Saludos!'}

Ahora modifiquemos el archivo index.html y modifiquemos la linea "<h1>{{data.bienvenida}}<h1> asi "...data.bienvenida...". 

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Home</title>
  </head>

  <body>
    <h1>{{data.Bienvenida}}<h1>
  </body>
</html>

Corramos la pagina y observaremos el siguiente resultado:

¡Saludos!

Vemos que ahora corre la clave de "Bienvenida" del diccionario data.

Ahora cambiemos la clave de 'titulo': 'Index' del dicionario data por 'titulo': 'Index123' en el archivo app.py.
"""
from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!'}
    return render_template('index.html', data=data)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)

"""
Tambien cambiemos en el archivo index.html, la linea "<title>Home</title>" por "<title>{{data.titulo}}</title>".

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{data.titulo}}</title>
  </head>

  <body>
    <h1>{{data.Bienvenida}}<h1>
  </body>
</html>

Corramos la pagina y observaremos el siguiente resultado:

El rotulo de la pagina ahora se llama Index123, antes se llamaba Home.

Ahora modifiquemos archivo app.py e index.html y observemos los cambios que se van dando en la pagina. 
"""
from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)

"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{data.titulo}}</title>
  </head>

  <body>
    <h1>{{data.Bienvenida}}<h1>
    <ul>
      {% for c in data.cursos %}
      <li>{{c}}</li>
      {% endfor %}
    </ul>

  </body>
  
La plantilla HTML que utiliza la sintaxis de plantillas de Jinja2 para mostrar una lista de cursos en una página web. La plantilla HTML utiliza un bucle `for` para iterar sobre una lista de cursos y mostrar cada curso en una etiqueta `<li>`.

La sintaxis `{% for c in data.cursos %}` indica que se va a iniciar un bucle `for` que iterará sobre la lista de cursos `data.cursos`. La variable `c` se utiliza para representar cada curso en la lista.

Dentro del bucle `for`, la plantilla HTML utiliza la sintaxis `{{c}}` para mostrar el nombre de cada curso en una etiqueta `<li>`. La sintaxis `{{c}}` indica que se va a mostrar el valor de la variable `c` en la plantilla HTML.

La sintaxis `{% endfor %}` indica que se ha finalizado el bucle `for`.

Vamos a la pagina y observemos el siguiente resultado:

¡Saludos!

Python
Flask
Django
Ruby
Java

Nuevamente modifiquemos archivo app.py e index.html y observemos los cambios que se van dando en la pagina. 

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{{data.titulo}}</title>
  </head>

  <body>
    <h1>{{data.Bienvenida}}<h1>
    {% if data.numero_cursos > 0 %}
    <ul>
      {% for c in data.cursos %}
      <li>{{c}}</li>
      {% endfor %}
    </ul>
    {% else %}
    <h2>No existen cursos</h2>
    {% endif %}
  </body>
</html>

La plantilla HTML que utiliza la sintaxis de plantillas de Jinja2 para mostrar una lista de cursos en una página web. La plantilla HTML utiliza una estructura de control `if-else` para verificar si hay cursos disponibles y mostrarlos en una lista o mostrar un mensaje de "No existen cursos" si no hay cursos disponibles.

La sintaxis `{% if data.numero_cursos > 0 %}` indica que se va a iniciar una estructura de control `if` que verificará si el número de cursos en la lista `data.cursos` es mayor que cero. Si el número de cursos es mayor que cero, se mostrará una lista de cursos. Si el número de cursos es cero, se mostrará un mensaje de "No existen cursos".

Dentro de la estructura de control `if`, la plantilla HTML utiliza la sintaxis `{% for c in data.cursos %}` para iniciar un bucle `for` que iterará sobre la lista de cursos `data.cursos`. La variable `c` se utiliza para representar cada curso en la lista.

Dentro del bucle `for`, la plantilla HTML utiliza la sintaxis `{{c}}` para mostrar el nombre de cada curso en una etiqueta `<li>`. La sintaxis `{{c}}` indica que se va a mostrar el valor de la variable `c` en la plantilla HTML.

La sintaxis `{% endfor %}` indica que se ha finalizado el bucle `for`.

La sintaxis `{% else %}` indica que se va a iniciar una estructura de control `else` que se ejecutará si el número de cursos en la lista `data.cursos` es cero. Dentro de la estructura de control `else`, la plantilla HTML muestra un mensaje de "No existen cursos" utilizando la etiqueta `<h2>`.

Modifiquemos archivo app.py , especificamente la linea "data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!', 'cursos': cursos, 'numero cursos': len(cursos)}" asignando a 'numero cursos': 0' y observemos los cambios que se van dando en la pagina.
"""

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': 0}
    return render_template('index.html', data=data)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)

"""
Observemos en resultado en la pagina:

¡Saludos!

No existen cursos

Vamos a crear un nuevo archivo en el directorio templates llamado "layout.html". Este archivo contendra la estructura basica de html5, que se repetira en todas las paginas de la aplicacion. 

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{% block title %}{% endblock %}</title>
  </head>

  <body>
    {% block body %}
    {% endblock %}
</body>
</html>

Ahora modifiquemos el archivo index.html.

{% extends './layaut.html' %}

{% block title %}{{data.titulo}}{% endblock %}

{% block body %}
<h1>{{data.Bienvenida}}</h1>
{% if data.numero_cursos > 0 %}
<ul>
    {% for c in data.cursos %}
    <li>{{c}}</li>
    {% endfor %}
</ul>
{% else %}
<h2>No existen cursos</h2>
{% endif %}
{% endblock %}

La plantilla HTML que utiliza la sintaxis de plantillas de Jinja2 para extender otra plantilla HTML llamada `layaut.html`. La plantilla HTML define dos bloques de contenido llamados `title` y `body`.

La sintaxis `{% extends './layaut.html' %}` indica que esta plantilla HTML extiende la plantilla HTML `layaut.html`. Esto significa que la plantilla HTML `index.html` hereda todo el contenido de la plantilla HTML `layaut.html`.

La sintaxis `{% block title %}{{data.titulo}}{% endblock %}` define un bloque de contenido llamado `title`. El contenido del bloque se establece en el valor de la variable `data.titulo`. Esto significa que el título de la página se establecerá en el valor de la variable `data.titulo`.

La sintaxis `{% block body %}` define un bloque de contenido llamado `body`. Este bloque se utiliza para definir el contenido principal de la página web.

La sintaxis `{% endif %}` Se utiliza para cerrar el bloque if-else.

La sintaxis `{% endblock %}` se utiliza para finalizar un bloque de contenido. En este caso, se utiliza para finalizar los bloques `title` y `body`.

Nota: Cuando haces cambios en un documento HTML y no se reflejan después de actualizar la página, es probable que se deba al almacenamiento en caché del navegador. Los navegadores almacenan en caché los archivos para mejorar el rendimiento y la velocidad de carga de las páginas web, lo que puede provocar que se muestren versiones antiguas del sitio web en lugar de las versiones actualizadas. Para solucionar este problema, puedes presionar "CTRL + F5" o "CTRL + SHIFT + R" para hacer que todos los recursos de la aplicacion web se vuelvan a cargar desde cero.  

Ahora dentro de el directorio python_for_web creamos un directorio llamado static, dentro de este creamos dos directorio, uno llamado css y otro js.  Luego creamos un archivo llamado layaut.css dentro del directorio css. Este archivo contendra el siguiente codigo css:

body {background-color: peru;}

El código proporcionado es una regla CSS que establece el color de fondo del cuerpo de una página web en "peru". La sintaxis `body` se utiliza para seleccionar el elemento `body` de la página web y aplicar estilos a ese elemento.

La sintaxis `{background-color: peru;}` establece el color de fondo del elemento `body` en "peru". El valor "peru" es un nombre de color predefinido en CSS que representa un tono marrón claro.

Luego vamos al archivo layaut.html y adicionamos la siguiente linea de codigo: <link rel="stylesheet" href="{{ url_for('static', filename='css/layaut.css') }}">.

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{% block title %}{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/layaut.css') }}">
  </head>

  <body>
    {% block body %}
    {% endblock %}
</body>
</html>

La línea de código es un enlace a una hoja de estilo CSS que se utiliza para aplicar estilos a la página web.

La sintaxis `<link rel="stylesheet" href="{{ url_for('static', filename='css/layaut.css') }}">` se utiliza para crear un enlace a una hoja de estilo CSS llamada `layaut.css`. La sintaxis `{{ url_for('static', filename='css/layaut.css') }}` se utiliza para generar una URL que apunta al archivo `layaut.css` en la carpeta `static/css` de la aplicación Flask.

La sintaxis `rel="stylesheet"` se utiliza para indicar que el enlace es una hoja de estilo CSS.

El el directorio js creamos un archivo llamado layaut.js, con el siguiente codigo:

alert('¡Hola Mundo!');

Ahora agrego la linea "<script src="{{ url_for('static', filename='js/layaut.js') }}"></script>" a layaut.html, para que se ejecute el archivo layaut.js:

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{% block title %}{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/layaut.css') }}">
  </head>

  <body>
    {% block body %}
    {% endblock %}
    <script src="{{ url_for('static', filename='js/layaut.js') }}"></script>
</body>
</html>

La línea de código es una etiqueta `<script>` que se utiliza para enlazar un archivo JavaScript llamado `layaut.js` a la página web.

La sintaxis `<script src="{{ url_for('static', filename='js/layaut.js') }}"></script>` se utiliza para crear un enlace a un archivo JavaScript llamado `layaut.js`. La sintaxis `{{ url_for('static', filename='js/layaut.js') }}` se utiliza para generar una URL que apunta al archivo `layaut.js` en la carpeta `static/js` de la aplicación Flask.

La sintaxis `src` se utiliza para indicar la ubicación del archivo JavaScript.

En resumen, esta línea de código HTML se utiliza para enlazar un archivo JavaScript llamado `layaut.js` a la página web. El archivo JavaScript se encuentra en la carpeta `static/js` de la aplicación Flask y se utiliza para agregar interactividad y funcionalidad a la página web.

Corramos la pagina web y veamos el resultado en el navegador.

Para seguir con el ejercicio vamos a neutralizar el archivo layaut.js, comentando la linea de codigo con "//":
//alert('¡Hola Mundo!');

Creamos un archivo llamado contacto.html en el directorio templates, con el siguiente codigo:

{% extends './layaut.html' %}

{% block title %}{{data.titulo}}{% endblock %}

{% block body %}
<h1>{{data.nombre}}</h1>
{% endblock %}

Adicionamos otro decorador que se explicara mas adelante, en el archivo app.py:

"""

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>')
def contacto(nombre):
    data={'titulo': 'Contacto', 'nombre': nombre}
    return render_template('contacto.html', data=data)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)
   
"""
La selección activa es una función de Python que se encuentra en el archivo `app.py`. La función se llama `contacto` y se utiliza para manejar una solicitud HTTP GET a la ruta `/contacto/<nombre>`.

La sintaxis `@app.route('/contacto/<nombre>')` se utiliza para decorar la función `contacto` y asociarla con la ruta `/contacto/<nombre>`. La parte `<nombre>` de la ruta es un parámetro dinámico que se puede utilizar para pasar un valor de nombre a la función.

Dentro de la función `contacto`, se define un diccionario llamado `data` que contiene dos claves: `titulo` y `nombre`. La clave `titulo` se establece en el valor `'Contacto'` y la clave `nombre` se establece en el valor del parámetro `nombre` que se pasa a la función.

La función `contacto` utiliza la función `render_template` de Flask para renderizar la plantilla HTML `contacto.html` y pasar el diccionario `data` a la plantilla. Esto significa que la plantilla HTML tendrá acceso a las claves y valores del diccionario `data`.

En resumen, esta función de Python se utiliza para manejar una solicitud HTTP GET a la ruta `/contacto/<nombre>` y renderizar la plantilla HTML `contacto.html` con un diccionario `data` que contiene un título y un nombre.

Vamos a la pagina web y escribimos a continuacion: http://127.0.0.1:5000/contacto/Alvaro. Corramos la pagina web y veamos el resultado en el navegador. Esto se conoce como construccion de URL dinamicas a traves del paso de parametros

Ahora vamos a complementar el archivo app.py, agregando, la variable edad al segundo decorador:
"""

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

if __name__ == '__main__':
   app.run(debug = True, port = 5000)

"""
Ahora vamos a complementar el archivo contactos.html, agregando al cuerpo la edad:

{% extends './layaut.html' %}

{% block title %}{{data.titulo}}{% endblock %}

{% block body %}
<h1>{{data.nombre}}</h1>
<h2>Tu edad es: {{data.edad}}</h2>
{% endblock %}

Vamos a la pagina web y escribimos a continuacion: http://127.0.0.1:5000/contacto/Alvaro/56. Corramos la pagina web y veamos el resultado en el navegador.

query string: Es una cadena de consulta que se utiliza para pasar datos de un lado a otro entre el cliente y el servidor. 

Vamos a complementar el archivo app.py, agregando, la funcion query_string y llamando la funcion request:
"""
from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    return 'ok'
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.run(debug = True, port = 5000)

"""
El código define una función llamada `query_string` que se utiliza para manejar una solicitud HTTP GET a la ruta `/query_string`.

La función `query_string` utiliza la función `print` de Python para imprimir el objeto `request`. El objeto `request` es un objeto de solicitud que contiene información sobre la solicitud HTTP, como los parámetros de consulta y los encabezados.

La función `query_string` devuelve la cadena `'ok'` como respuesta a la solicitud HTTP.

Dentro del bloque `if __name__ == '__main__':`, se utiliza la función `add_url_rule` de Flask para agregar una regla de URL que asocia la ruta `/query_string` con la función `query_string`.

Vamos a la pagina web http://127.0.0.1:5000/query_string y observemos el resultado en el navegador.

ahora vamos a complementar el archivo app.py, agregando, la funcion query_string y llamando la funcion request, para que muestre los parametros de la query string:
"""

from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param1'))
    print(request.args.get('param2'))
    return 'ok'
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.run(debug = True, port = 5000)


"""
Ahora vamos a darle parametros a param1, Alvaro y a param2, 56. Vamos a la pagina web con la direccion y parametros definidos: http://127.0.0.1:5000/query_string?param1=Alvaro&param2=59. No observamos niunguna cambio en la pagina web, pero si vamos al teminal del editor veremos dos llaves, observemos:

<Request 'http://127.0.0.1:5000/query_string?param1=Alvaro&param2=56' [GET]>
ImmutableMultiDict([('param1', 'Alvaro'), ('param2', '56')])
Alvaro
56


y observemos el resultado en el navegador.

Vamos a la pagina web http://127.0.0.1:5000/register, ejecutemos y observamos el siguiente mensaje en el cuerpo de la pagina web:

No se encuentra esta página 127.0.0.1No se encontró una página web para la siguiente dirección web: http://127.0.0.1:5000/Registro
HTTP ERROR 404

Deseamos un mensaje especifico para cuando no se encuentre la pagina web, para ello, vamos a complementar el archivo app.py. Veamos:
"""

from flask import Flask, render_template, request
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param1'))
    print(request.args.get('param2'))
    return 'ok'

def pagina_no_encontrada(error):
    return render_template('404.html'), 404
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug = True, port = 5000)
    
"""
La función se llama `pagina_no_encontrada` y se utiliza para manejar errores 404 en la aplicación Flask.

La función `pagina_no_encontrada` toma un parámetro `error`, que es el objeto de error que se produce cuando se produce un error 404 en la aplicación Flask.

Dentro de la función `pagina_no_encontrada`, se utiliza la función `render_template` de Flask para renderizar la plantilla HTML `404.html`. La función devuelve la plantilla HTML y el código de estado HTTP 404.

La sintaxis `app.register_error_handler(404, pagina_no_encontrada)` se utiliza para registrar la función `pagina_no_encontrada` como la función de manejo de errores para el código de estado HTTP 404 en la aplicación Flask. Esto significa que cuando se produce un error 404 en la aplicación Flask, la función `pagina_no_encontrada` se ejecutará para manejar el error.

Ahora creemos el archivo 404.html en el directorio templates. Este archivo contendra el siguiente codigo:

{% extends './layaut.html' %}

{% block title %}Pagina no encontrada{% endblock %}

{% block body %}
<h2>Opsss... La pagina que buscas no existe!</h2>
{% endblock %}

Vamos a la pagina http://127.0.0.1:5000/register y observemos el resultado en el navegador.

Cuando deseamos redireccionar a otra pagina web, utilizamos la funcion redirect, para ello, vamos a complementar el archivo app.py llamando las funciones url_for y redirect  Veamos:
"""

from flask import Flask, render_template, request, url_for, redirect
import os

app = Flask(__name__)

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param1'))
    print(request.args.get('param2'))
    return 'ok'

def pagina_no_encontrada(error):
    return redirect(url_for('index'))
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug = True, port = 5000)

"""
La línea de código se utiliza para redirigir al usuario a la página de inicio de la aplicación Flask.

La sintaxis `redirect(url_for('index'))` se utiliza para crear una respuesta HTTP de redirección que redirige al usuario a la ruta asociada con la función `index` que es el archivo index.html, cuando la direccion original condice a un error 404. La función `url_for` se utiliza para generar una URL que apunta a la ruta asociada con la función `index`.


"""

from flask import Flask, render_template, request, url_for, redirect
import os

app = Flask(__name__)

@app.before_request
def before_request():
    print('Antes de la petición ...')
    
@app.after_request
def after_request(response):
    print('Después de la petición ...')
    return response

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param1'))
    print(request.args.get('param2'))
    return 'ok'

def pagina_no_encontrada(error):
    # return render_template('404.html'), 404
    return redirect(url_for('index'))
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug = True, port = 5000)

"""
Los decoradores @app.before_request y @app.after_request son funcionalidades proporcionadas por el framework Flask en Python. Estos decoradores te permiten ejecutar ciertas funciones antes y después de que se maneje una solicitud HTTP.

@app.before_request: Este decorador se utiliza para registrar una función que se ejecutará antes de que Flask maneje la solicitud. Puede ser útil para realizar tareas de inicialización, autenticación, verificación de permisos u otros tipos de tareas de preparación antes de que se maneje la solicitud.

@app.after_request: Este decorador se utiliza para registrar una función que se ejecutará después de que Flask maneje la solicitud y antes de que se envíe la respuesta al cliente. Puede ser útil para realizar tareas de limpieza, ajuste de encabezados de respuesta u otros tipos de tareas posteriores a la solicitud.

Veamos el ejemplo, vamos a complementar el archivo app.py, agregando, las funciones before_request y after_request:
"""


from flask import Flask, render_template, request, url_for, redirect
import os

app = Flask(__name__)

@app.before_request
def before_request():
    print('Antes de la petición ...')
    
@app.after_request
def after_request(response):
    print('Después de la petición ...')
    return response

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param1'))
    print(request.args.get('param2'))
    return 'ok'

def pagina_no_encontrada(error):
    return redirect(url_for('index'))
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug = True, port = 5000)
    
"""
Al correr la pagina http://127.0.0.1:5000/, vemos la siguiente respuesta en el terminal del editor:

Antes de la petición ...
Antes de la petición ...
Después de la petición ...
Después de la petición ...
127.0.0.1 - - [26/Oct/2023 17:57:41] "GET /static/css/layaut.css HTTP/1.1" 304 -
127.0.0.1 - - [26/Oct/2023 17:57:41] "GET /static/js/layaut.js HTTP/1.1" 304 -

Si se requiere conexion a una base de datos como MySQL, se debe instalar el paquete mysql-connector-python, para ello, vamos a la terminal del editor y ejecutamos el siguiente comando:

pip install flask_mysqldb

Complementamos el archivo app.py, agregando, la funcion mysql y agrgando lineas para aceder y obtener la base de datos.
"""

from flask import Flask, render_template, request, url_for, redirect, jsonify
from flask_mysqldb import MySQL

app = Flask(__name__)

# Configuración de la conexión a la base de datos
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'usuario'
app.config['MYSQL_PASSWORD'] = 'contraseña'
app.config['MYSQL_DB'] = 'basedatos'

# Creación de una instancia de la clase MySQL
conexion = MySQL(app)

@app.before_request
def before_request():
    print('Antes de la petición ...')
    
@app.after_request
def after_request(response):
    print('Después de la petición ...')
    return response

@app.route('/')
def index():
    cursos = ['Python', 'Flask', 'Django', 'Ruby', 'Java']
    data = {'titulo': 'Index123', 'Bienvenida': '¡Saludos!',
            'cursos': cursos, 'numero_cursos': len(cursos)}
    return render_template('index.html', data=data)

@app.route('/contacto/<nombre>/<int:edad>')
def contacto(nombre,edad):
    data={'titulo': 'Contacto', 'nombre': nombre, 'edad': edad}
    return render_template('contacto.html', data=data)

def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param1'))
    print(request.args.get('param2'))
    return 'ok'

@app.route('/cursos')
def listar_cursos():
    data={}
    try:
        cursor=conexion.connection.cursor()
        sql='SELECT codigo, nombre, creditos FROM curso ORDER BY nombre ASC'
        cursor.execute(sql)
        cursos=cursor.fetchall()
        print(cursos)
        data['cursos']=cursos
        data['mensaje'] = 'Exito'
    except Exception as ex:
        data['mensaje'] = 'Error al listar los cursos'
    return jsonify(data)

def pagina_no_encontrada(error):
    # return render_template('404.html'), 404
    return redirect(url_for('index'))
    
if __name__ == '__main__':
    app.add_url_rule('/query_string', view_func=query_string)
    app.register_error_handler(404, pagina_no_encontrada)
    app.run(debug = True, port = 5000)

"""
El código configura la conexión a una base de datos MySQL en una aplicación Flask utilizando la biblioteca `flask_mysqldb`.

El código utiliza la variable `app` para configurar la conexión a la base de datos. Las variables `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD` y `MYSQL_DB` se utilizan para especificar la dirección del servidor de la base de datos, el nombre de usuario y la contraseña para la conexión, y el nombre de la base de datos a la que se desea conectar.

Después de configurar la conexión a la base de datos, se crea una instancia de la clase `MySQL` utilizando la variable `app`. La instancia de la clase `MySQL` se almacena en la variable `conexion` y se utiliza para realizar operaciones de base de datos en la aplicación Flask.

La función se llama `listar_cursos` y se utiliza para manejar una solicitud HTTP GET a la ruta `/cursos`.

Dentro de la función `listar_cursos`, se define un diccionario llamado `data` que se utiliza para almacenar los datos que se devolverán como respuesta a la solicitud HTTP. El diccionario `data` tiene dos claves: `cursos` y `mensaje`.

Dentro del bloque `try`, se utiliza la variable `cursor` para crear un cursor de base de datos y ejecutar una consulta SQL para seleccionar todos los cursos de la tabla `curso`. La función `cursor.fetchall()` se utiliza para obtener todas las filas del resultado de la consulta y almacenarlas en la variable `cursos`.

La variable `cursos` se agrega al diccionario `data` con la clave `cursos`. La clave `mensaje` se establece en el valor `'Exito'`.

Si se produce una excepción dentro del bloque `try`, se establece la clave `mensaje` en el valor `'Error al listar los cursos'`.

Finalmente, la función `jsonify` de Flask se utiliza para convertir el diccionario `data` en una respuesta HTTP JSON y devolverla como respuesta a la solicitud HTTP.
"""