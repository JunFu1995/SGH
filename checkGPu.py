import pynvml
import time 
import subprocess
pynvml.nvmlInit()



def taskGenerate():
    cmd_list = []
    for m in ['HyperIQA']:
        for ds in ['CVIU']:#[ 'Waterloo', 'QADS', 'CVIU']:
            cmd = 'nohup python train_test_IQA.py --dataset %s --netFile %s --gpuid %%d --batch_size 64> ./log/%s_%s.log &' % (ds, m, m, ds)
            #print(cmd)
            cmd_list.append(cmd)
    return cmd_list


cmd = taskGenerate()

while cmd:
    for i in range(3):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memoinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if (memoinfo.used/1e6 < 100):
            c = cmd.pop(0)
            c = c % i 
            print(c)
            subprocess.call(c, shell=True)
    time.sleep(90)
