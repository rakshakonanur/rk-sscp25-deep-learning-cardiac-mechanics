# Example code for SSCP 2025 project for caridac mechanics

This respotitory contains example code for the SSCP 2025 project on cardiac mechanics. 
The code uses

## Installation
The code depend on [fenicsx-pulse](https://github.com/finsberg/fenicsx-pulse) and [caridac-geometriesx](https://github.com/ComputationalPhysiology/cardiac-geometriesx).


### conda
You can install the dependencies by using the provided environment file `environment.yml` in the root of the repository. First clone the repository and then run the following command in the root directory of the repository:

```bash
conda env create -f environment.yml
```

After the environment is created, you can activate it using:

```bash
conda activate sscp25-deep-learning-cardiac-mechanics
```

### Docker
Alternatively, you can use the fenicsx docker image and install the dependencies using pip. In the root directory of the repository, run the following commands:

```bash
docker run --name sscp25-deep-learning-cardiac-mechanics -w /home/shared -v $PWD:/home/shared -it ghcr.io/fenics/dolfinx/dolfinx:stable
```
Then install the dependencies using pip:

If you are running docker on a Mac you should run
```bash
python3 -m pip install -r requirements-aarch64.txt
```
otherwise run
```bash
python3 -m pip install -r requirements.txt
```

## Getting started
Create a geometry with
```
python3 create_geometry.py
```

Run a simulation with
```
python3 run_simulation.py
```

Post-process the results with
```
python3 postprocess.py
```


## License
MIT License
