import subprocess,os

env_name = 'S_UI'
#conda create
command = "conda create --name "+env_name+" python=3.9"
process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
process.communicate(input=b'y\n')
os.system('conda list')

command = "activate "+env_name+" && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
os.system(command)

command = "activate "+env_name+" && pip install h5py"
os.system(command)

command = "activate "+env_name+" && pip install scipy"
os.system(command)
 
command = "activate "+env_name+" && pip install matplotlib"
os.system(command)

command = "activate "+env_name+" && pip install pandas"
os.system(command)

command = "activate "+env_name+" && pip install scikit-learn"
os.system(command)
os.system('conda list')
command = "activate "+env_name+" && pip install tensorboard"
os.system(command)

command = "activate "+env_name+" && pip install torchmetrics"
os.system(command)

command = "activate "+env_name+" && pip install tqdm"
os.system(command)


os.system("activate "+env_name+" && python train_ui.py")