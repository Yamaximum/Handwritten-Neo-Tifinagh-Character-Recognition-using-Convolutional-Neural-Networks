import scipy.io
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

    
# Load the MATLAB file
mat_data = scipy.io.loadmat('BDR.mat')

# Extract the image data and labels from the MATLAB file
images = mat_data['Data_Set2']  
labels = mat_data['Data_Label']  
neo_tifinagh_alphabet_list = ["ⴰ", "ⴱ", "ⵛ", "ⴷ", "ⴹ", "ⴺ", "ⴻ", "ⴼ", "ⴳ", "ⵄ", "ⵀ", "ⵃ", "ⵉ", "ⵊ", "ⴽ", "ⵍ", "ⵎ", "ⵏ", "ⵒ", "ⵇ", "ⵔ", "ⵕ", "ⵙ", "ⵚ", "ⵜ", "ⵟ", "ⵓ", "ⵖ", "ⵡ", "ⵅ", "ⵢ", "ⵣ", "ⵥ"]                                            

# Subtract 1 from each label to convert them from 1-33 to 0-32
labels = labels - 1

# Remove extra dimensions
labels = labels.squeeze() 

# Reshape the images to (45102, 50, 50) if needed
images = np.moveaxis(images, -1, 0)  # Move the last dimension to the first

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.11, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Create DataLoader objects for training and testing
batch_size = 32   
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(inputs[i],cmap='gray')
    
# Function to count the number of parameters
def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

# Initialize empty lists to store training and testing loss and accuracy
train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []

def train(epoch, model):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
            
        data = data.view(-1, 1, 50, 50)
        
        # Perform training steps and calculate metrics
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if batch_idx % 100 == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tPerte : {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
        # Update total loss and accuracy    
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    # Calculate and store epoch-level metrics    
    total_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    train_loss_list.append(total_loss)
    train_accuracy_list.append(accuracy)
    print('Epoch {}: Train Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch, total_loss, accuracy))

def test(model, perm=torch.arange(0, 50*50).long()):
    
    # Evaluate the model's performance on the test dataset
    model.eval()
    test_loss = 0
    correct = 0
    y_preds = []

    for data, target in test_loader:
        
        data = data.view(-1, 1, 50, 50)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()                                                                
        
        # Calculate and accumulate the number of correct predictions
        pred = output.data.max(1, keepdim=True)[1]                                                             
        y_preds.append(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
    
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)    
    
    # Calculate the average test loss and accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_loss_list.append(test_loss)
    test_accuracy_list.append(accuracy)
    
    # Print the evaluation results 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return y_pred_tensor
    
class CNN(nn.Module):
    def __init__(self, input_size, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_feature, kernel_size=5)
        self.conv2 = nn.Conv2d(n_feature, n_feature, kernel_size=5)
        self.fc1 = nn.Linear(n_feature*9*9, 100)
        self.fc2 = nn.Linear(100, 33)
        
    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature*9*9)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Training parameters
n_features = 24 # Number of feature maps

model_cnn = CNN(50*50, n_features, 33)
optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.85)
print('Number of parameters : {}'.format(get_n_params(model_cnn)))

for epoch in range(0, 10):
    train(epoch, model_cnn)
    test(model_cnn)
    
# Plot loss and accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(test_accuracy_list, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()

# Get the predictions
y_preds = test(model_cnn)  

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_preds)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[f'{i}' for i in neo_tifinagh_alphabet_list], yticklabels=[f'{i}' for i in neo_tifinagh_alphabet_list])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")



