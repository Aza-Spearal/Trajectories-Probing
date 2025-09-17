import torch
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



[inp, target] = torch.load('CORE_activation_model_19667.pt') #[samples, layers, 4096]
print(inp.shape, target.shape)
x_train, x_test, y_train, y_test = train_test_split(inp, target.to(device), test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
torch.save([x_train, x_test, y_train, y_test], 'CORE_activation_model_19667X.pt')

[a, b, c, d] = torch.load('CORE_activation_model_19667X.pt')
print(a.shape, b.shape, c.shape, d.shape)



[inp, target] = torch.load('CORE_activation_rand_21118.pt') #[samples, layers, 4096]
print(inp.shape, target.shape)
x_train, x_test, y_train, y_test = train_test_split(inp, target.to(device), test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
torch.save([x_train, x_test, y_train, y_test], 'CORE_activation_rand_21118X.pt')
[a, b, c, d] = torch.load('CORE_activation_rand_21118X.pt')
print(a.shape, b.shape, c.shape, d.shape)

[a, b, c, d] = torch.load('CORE_activation_rand_21118X.pt')
print(a.shape, b.shape, c.shape, d.shape)