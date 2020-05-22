import torch
import sys
sys.path.append("../../")
from model import ConvLSTM
import numpy as np
import os
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
model_dir = './weights' # ConvLstm models weights

params_ls = {1:[[128, 64, 64, 32, 32],(5, 5), False],
             2:[[128, 64, 64, 32, 32],(5, 5), False]}
model_name = 'ConvLstm_struct1_2020-03-13-15-50-01.pt'

if use_cuda:
    print('use CUDA')
else:
    print('use CPU')

class prediction(object):
    def __init__(self, data_pth, model_name):
        self.input_data = np.loadtxt(data_pth).astype(np.float32).reshape(-1,1,17,2)#(seq_l, 1, 17, 2)
        self.model_name = os.path.join(model_dir, model_name)
        structre_num = int(model_name.split('_')[1][6:])
        self.model = ConvLSTM(input_size=(17, 2),
                             input_dim=1,
                             hidden_dim=params_ls[structre_num][0],
                             kernel_size=params_ls[structre_num][1],
                             num_layers=len(params_ls[structre_num][0]),
                             num_classes=2,
                             batch_size=2,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False,
                             attention=params_ls[structre_num][2]).cuda()
        self.model.load_state_dict(torch.load(self.model_name))
        self.model.eval()
    def get_input_data(self, input_data):
        data = torch.from_numpy(input_data)
        data = data.unsqueeze(0).to(device=device)
        
        return data#(1, 30, 1, 17, 2)
    def predict_pre_second(self, data):
        output = self.model(data)
        #print('output:',output)
        pred = output.data.max(1, keepdim=True)[1]
        #print('pred:',pred)
        return pred
    def predict(self):
        preds = []
        for i in range(0, self.input_data.shape[0], 30):
            data = self.get_input_data(self.input_data[i:i+30,:,:,:])
            if data.size(1)<30:
                break
            pred = self.predict_pre_second(data)
            print('pred:',pred)
            preds.append(pred)
        return pred


if __name__ == '__main__':
    input_pth = './demo/AlphaPose_john.txt'
    prediction = prediction(input_pth, model_name)
    prediction.predict()
    
    