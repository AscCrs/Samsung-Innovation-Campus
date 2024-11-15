![Banner SIC](./Notas%20de%20clase/Imagenes/SamsungUDEM.png)
<h1 style="text-align: center;">Samsung Innovation Campus AI</h1>

## ¿Qué es Samsung Innovation Campus?
Es una iniciativa global de Samsung dirigida a estudiantes universitarios de instituciones públicas, con el propósito de capacitar y empoderar a jóvenes adultos para integrarse al mercado laboral impulsado por la tecnología.

En esta edición, la Universidad de Monterrey (UDEM) colabora en el programa, ofreciendo formación en Inteligencia Artificial y Liderazgo a los participantes. 

> [!NOTE] El presente repositorio, fue creado con la intención de llevar un registro del avance obtenido durante el curso por cuenta propia.
> - Unicamente se llevará registro de las notas tomadas por el estudiante.
> - Las prácticas igualmente presentadas, también son hechas en su totalidad por el estudiante.
## Ejecución de las prácticas
El desarrollo de las prácticas se llevo a cabo utilizando `Python 3`, por lo que resulta indispensable la instalación del mismo para poder ejecutar las prácticas.

Los paquetes utilizados de `Python` son los siguientes:
- `Numpy`
- `Matplotlib`
- `Pandas`
- `Jupyter`

Primero se debe de crear un **entorno virtual** para no generar conflictos con dependencias previamente instaladas, por lo que se debe de ejecutar el siguiente comando en la terminal:

``` Bash
python -m venv <Nombre del entorno virtual>
```

Una vez creado el entorno virtual, se debe inicializar. Para ello se ingresan cualquiera de los siguientes dos comandos, dependiendo si estas en **Windows** o **Linux**.

### Windows
Tomando en consideración que nos encontramos fuera de la carpeta donde esta el enorno
``` Powershell
.\<Nombre del entorno>\Scripts\activate
```
### Linux
Tomando en consideración que nos encontramos fuera de la carpeta donde esta el enorno
``` bash
source env/bin/activate.<Extension de tu terminal>
```
Con `extensión de tu terminal` se refiere a el tipo de terminal que estes utilizando, ya sea **fish**, **bash**, etc.

### Instalación de los paquetes
Para poder instalar los paquetes utilizados para poder ejecutar las prácticas, haremos uso del manejador de paquetes de python `pip`.

Para poder instalar `Numpy`, ejecutaremos en la terminal
``` Python
pip install numpy
```

Para instalar `Matplotlib`, ejecutaremos el siguiente comando:
``` Python
pip install matplotlib
```

Para instalar `Pandas`, ejecutaremos el siguiente comando:
``` Python
pip install pandas
```

Para poder instalar `Jupyter Nootebook`, requeriremos los siguientes paquetes:

`jupyter` permitira trabajar con libretas de jupyter en visual studio code
``` Python
pip install jupyter
```
`ipykernel` contiene las instrucciones necesarias para poder ejecutar las instrucciones de jupyter en nuestro equipo
``` Python
pip install ipykernel
```
`nbconvert` permite realizar la conversión de archivos de `ipynb` a `pdf`, `HTML`
``` Python
pip install nbconvert
```
