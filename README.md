# Donkey-car-Attack-Defence

## Guidance
### Install donkey car on windows-software

 - Change to a dir you would like to use as the head of your projects.
 
```
mkdir projects
cd projects
```
---

 - Get the donkey from Github.
```
git clone https://github.com/wroscoe/donkey
cd donkey
```
 - Create the python Anaconda environment
```
conda env create -f envs\windows.yml
activate donkey
```
  Note : when you get "SpecNotFound: Can't process without a name", you need change your work path to ~/donkey
 - Install donkey source and create your local working dir:
```
pip install -e .
donkey createcar --path ~/d2
```
## Un-target attack algorithms
### Hardware-camera
### Software-picture

---

## Target attack algorithms

### Software-picture

---

## Defence algorithms


