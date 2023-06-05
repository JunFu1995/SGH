


for md in ['CNNIQA','JCSAN','Resnet50','DeepSRQ', 'HyperIQA']:
    dp = []

    for ds in [ 'Waterloo', 'QADS', 'CVIU']:
        with open('./log/%s_%s.log' % (md, ds), 'r') as f:
            lines = f.readlines()
            line = lines[-1]
            line = line.strip().split(' ')
            #print(line)
            srcc = line[3].split(',')[0]
            plcc = line[5].split(',')[0]
            krcc = line[-1]
            dp += [srcc, plcc, krcc]

    print('&'.join(dp))




