"""
Configuración de entornos virtuales
Para empezar con el proyecto, sería mejor disponer de un entorno virtual. El entorno virtual puede ayudarnos a crear un entorno aislado o separado. Esto nos ayudará a evitar conflictos en las dependencias entre proyectos. Si escribimos pip freeze en nuestro terminal veremos todos los paquetes instalados en nuestro ordenador. Si usamos virtualenv, accederemos sólo a los paquetes que son específicos para ese proyecto. Abre tu terminal e instala virtualenv.

pip install virtualenv

Vaya al terminal y escriba frente al directorio raiz el nuevo entorno virtual env, ejecutelo.:

virtualenv venv

Se crearon los siguientes archivos dentro de evn.
Lib
Scripts
.gitignore
pyvenv.cfg

Lib: Este directorio contiene las bibliotecas de Python y los paquetes instalados en el entorno virtual. Cada vez que instale un paquete en su entorno virtual, sus archivos se almacenan aquí. Esto asegura que las bibliotecas y paquetes en su entorno virtual sean independientes de los de tu sistema global de Python. Por lo tanto, puede tener diferentes versiones de bibliotecas en entornos virtuales separados sin que interfieran entre sí.

Scripts: Este directorio contiene los scripts y ejecutables necesarios para administrar y trabajar en el entorno virtual. Aquí se encuentra, por ejemplo, el script de activación (activate en sistemas Unix o activate.bat en Windows), que se utiliza para activar el entorno virtual en su terminal o consola. También puede contener otros scripts relacionados con la gestión del entorno virtual.

.gitignore: Este archivo es parte de la configuración de Git y se utiliza para especificar qué archivos o directorios no se deben incluir en el repositorio de Git. Esto es útil cuando está trabajando en un proyecto de Python y no desee que los archivos de su entorno virtual se incluyan en el control de versiones. Por lo general, se agrega una línea en .gitignore para excluir el directorio del entorno virtual y otros archivos específicos del proyecto.

pyvenv.cfg: Este archivo contiene la configuración del entorno virtual. Puede incluir información sobre la versión de Python utilizada en el entorno virtual y otras configuraciones específicas. La existencia y el contenido de este archivo pueden variar dependiendo de la versión de Python y el sistema operativo en el que se creó el entorno virtual.

Vamos a terminal y escribimos y ejecutamos frente al directorio raiz:

Vamos, en mi caso, a la terminal al directorio raiz C:\Users\alvan\OneDrive\Documentos\Python\Curso Python> y ejecutamos el comando:

venv\Scripts\activate

Se observara la siguiente respuesta, indicando que el entorno virtual está activo:

(venv) PS C:\Users\alvan\OneDrive\Documentos\Python\Curso Python> 

Para desactivar el entorno virtual ejecutamos el comando en el terminal:

deactivate

Se observara la siguiente respuesta, indicando que el entorno virtual está desactivado:

PS C:\Users\alvan\OneDrive\Documentos\Python\Curso Python>

Usted no deberia ver ningun paquete instalado en el entorno virtual activado. Compruebelo ejecuntando cualquiera de estos dos comandos en la terminal:

pip freeze
pip list
"""

