import subprocess,os
#conda remove --name <虚拟环境名称> --all
def del_env(project_name):
    command = "conda env remove --name "+ str(project_name)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)
    process.communicate(input=b'y\n')
    os.system('conda list')

os.system('conda info --envs')
# del_env('ASDF')
# del_env('ASDFASDF')
# del_env('PD_GS12345 ')
# del_env('PD_GSS   ')
# del_env('base1236789 ')
# del_env('base123 ')
