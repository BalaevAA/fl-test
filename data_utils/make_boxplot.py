import matplotlib.pyplot as plt
import copy
import numpy as np

def create_boxplot(args, train_local_loss):
    m = []
    for i in train_local_loss.keys():
        m.append(train_local_loss[i])
    
    arr_lens = []
    for i in range(50):
        arr_lens.append(len(m[i]))
    
    max_ep = max(arr_lens)
    
    
    m_c = copy.deepcopy(m)
    for i in range(args.num_users):
        if len(m_c[i]) < max_ep:
            for j in range(max_ep - len(m_c[i])):
                m_c[i].append(0)
    
    
    bp = []
    for i in range(max_ep):
        tmp = []
        for j in range(args.num_users):
            tmp.append(m_c[j][i])
        bp.append(tmp)
    
    
    for i in range(len(bp)):
        bp[i] = [j for j in bp[i] if j != 0]
    
    
    data = []
    temp = np.arange(0, 44, 3)
    for i in range(0, 44):
        if i in temp:
            data.append(bp[i])
        else:
            data.append([])
            
    
    
    fig = plt.figure(figsize =(15, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(bp)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(f'boxplot_epoch_{args.epochs}_numUser_{args.num_users}_dataDist_{args.data_dist}_dataset_{args.dataset}')
    # plt.show()
    plt.close()


train_local_loss = {2: [4.779181494257757,
  3.026107703872736,
  2.405161153787894,
  2.4243805856130924,
  2.4,
  2.009007349424837,
  1.5016178604972807,
  1.6795316244556695],
 4: [4.22435865309632],
 7: [3.9970805602414266, 3.4094285610176267, 2.8406616619655063],
 1: [3.809470191834465, 3.2910188181059703, 2.6324863508343697],
 6: [3.6203159103548623,
  3.214839448531469,
  2.889386916911699,
  2.6882358717966857],
 5: [3.5331490357562263, 3.135544874092948, 2.989422511033975],
 0: [3.2610006382068, 2.8045901814623484, 2.823920658538266],
 3: [3.2253085409747007, 2.9649695277712835],
 9: [2.9906185671718366, 2.8695143873586595],
 8: [3.0403976554188055]}

m = []
for i in train_local_loss.keys():
    m.append(train_local_loss[i])

arr_lens = []
for i in range(10):
    arr_lens.append(len(m[i]))

max_ep = max(arr_lens)


m_c = copy.deepcopy(m)
for i in range(10):
    if len(m_c[i]) < max_ep:
        for j in range(max_ep - len(m_c[i])):
            m_c[i].append(0)


bp = []
for i in range(max_ep):
    tmp = []
    for j in range(10):
        tmp.append(m_c[j][i])
    bp.append(tmp)


for i in range(len(bp)):
    bp[i] = [j for j in bp[i] if j != 0]


data = []
temp = np.arange(0, max_ep, 3)
for i in range(0, max_ep):
    # if i in temp:
    #     data.append(bp[i])
    # else:
    #     data.append([])
    data.append(bp[i])

print(data)
a = [[4.779181494257757, 4.22435865309632, 3.9970805602414266, 3.809470191834465, 3.6203159103548623, 3.5331490357562263, 3.2610006382068, 3.2253085409747007, 2.9906185671718366, 3.0403976554188055], 
     [3.026107703872736, 3.4094285610176267, 3.2910188181059703, 3.214839448531469, 3.135544874092948, 2.8045901814623484, 2.9649695277712835, 2.8695143873586595], 
     [2.405161153787894, 2.8406616619655063, 2.6324863508343697, 2.889386916911699, 2.989422511033975, 2.823920658538266], 
     [2.4243805856130924, 2.6882358717966857], 
     [2.5, 2.1], 
     [2.009007349424837, 1.7], 
     [1.5016178604972807], 
     [1.1795316244556695]]
        


fig = plt.figure(figsize =(15, 10))
ax = fig.add_axes([0, 0, 1, 1])
bp = ax.boxplot(a)
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.savefig(f'boxplot_epoch_{args.epochs}_numUser_{args.num_users}_dataDist_{args.data_dist}_dataset_{args.dataset}')
plt.show()
plt.close()