from taker import Model
import json
import torch
from tqdm import tqdm
import os.path
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LabeledTokenizer:
    def __init__(self, model, labels):
        self.model: Model = model
        self.lab = labels

    def tokenize(self, data, max_char, indice, function):

        self.model.eval()
        input_list = []
        label_list = []
        breaking = -1

        name = datase+'_'+function+'_'+mode
        print(name)

        for i, chunck in enumerate(tqdm(data)):

            text = chunck['text'][:int(max_char)]
            label = chunck['label']

            if len(text)>0:

                    try:
                        with torch.no_grad():
                            if function == 'residual':
                                residual = self.model.get_residual(text) #[layers, n_token, dmodel]
                            else:
                                residual = self.model.get_activations(text) #[layers, n_token, dmodel]
                            mean = torch.mean(residual.to(device), dim=1, keepdim=True).cpu() #[layers,1,288]
                            
                        input_list.append(mean)
                        del residual #to free memory (don't know if it works)
                        label_list.append(self.lab.index(label))

                    except torch.cuda.OutOfMemoryError:
                        breaking = i #cuda OOM, we leave the try/except
                        print('on break la', breaking)
                        break

        if breaking == 0: #oom at the first element
            print('OOM at the first element')
        elif breaking == i: #cuda OOM, we leave the loop
            print('OOM at the '+str(i)+'_th element', breaking, indice)
            breaking += indice
        

        if i == len(data)-1 and breaking == -1: #the dataset is finish and we didn't OOM
            print('dataset over')
            breaking=0

        print('len(label_list):', len(label_list))

        if len(label_list) != 0: #if we did something (either OOM or the dataset is finish)
            print('we did something', breaking)
            input_torch = torch.cat(input_list, dim=1)
            input_torch = input_torch.permute(1, 0, 2) #[layers, samples, 4096] => [samples, layers, 4096]
            label_torch = torch.tensor(label_list)

            if os.path.exists('/root/workspace/taker/eloise/'+name+'.pt'):
                [a, b] = torch.load(name+'.pt')
                c = torch.cat((a, input_torch))
                d = torch.cat((b, label_torch))
            else:
                c = input_torch
                d = label_torch
            torch.save([c, d], name+'.pt')

        f = open('temp.txt', "w")
        print('in temp, we write:', breaking)
        f.write(str(breaking))
        f.close()



indice = int(sys.stdin.read().strip())
#indice = 0

datase = 'CORE'

#function = 'residual'
function = 'activation'

mode = 'model'
#mode = 'rand'


f = open('CORE_edited.json')
data = json.load(f)

labels = ['OP', 'SR', 'NE', 'IN', 'PB', 'IP', 'ID', 'HI']


data = sorted(data, key=lambda x: len(x["text"]))
data = data[indice:]

limit=10000000000000000

if mode == 'model':
    modele = 'mistralai/Mistral-7B-Instruct-v0.2'
else:
    modele = "nickypro/mistral-7b-rand"

m = Model(modele, dtype="int8", limit=limit, output_device="cpu", collect_midlayers=False)

labeled_tokenizer = LabeledTokenizer(m, labels)

labeled_tokenizer.tokenize(data, limit*5.1, indice, function)