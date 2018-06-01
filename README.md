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
  **Note** : when you get "SpecNotFound: Can't process without a name", you need change your work path to ~/donkey
 - Install donkey source and create your local working dir:
```
pip install -e .
donkey createcar --path ~/d2
```
### Start driving (On windows10)

 - Start a new Anaconda prompt
 - Change default environment
 ```
 activate donkey
 ```
 - Change to local dir for managing donkey:
 ```
 cd user\yourname\d2
 ```
  **Note:**Different between "/" in linux and "\" in windows.
 - Connect to our pi (via ssh):You must have set a wifi "ssid" and "password" already, then PI will connect to your network automaticly.
 - On the PI's operation system(Linux), start your car
 ```
 cd ~/d2
 python manage.py drive
 ```
 - This script will start the drive loop in your car which includes a part that is a web server for you to control your car. You can now control your car from a web browser at the URL: **<your car's IP's address>:8887**
 ![pic](http://docs.donkeycar.com/assets/drive_UI.png)
 - Keyboard control: (:space:) (:A:) (:i:) (:j:) (:k:) (:l:)
## Un-target attack algorithms
### Hardware-camera
### Software-picture

---

## Target attack algorithms

### Software-picture

---

## Defence algorithms


## Reference
[Donkey documentation](http://docs.donkeycar.com/)
