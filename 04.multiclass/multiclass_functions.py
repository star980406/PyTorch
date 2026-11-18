
import torch
import numpy as np
import matplotlib.pyplot as plt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 학습
def Train(model, train_DL, criterion, optimizer, EPOCH):
  loss_history=[]
  NoT = len(train_DL.dataset)

  model.train()
  for ep in range(EPOCH):
    rloss = 0
    for x_batch, y_batch in train_DL:
      x_batch = x_batch.to(DEVICE)
      y_batch = y_batch.to(DEVICE)

      y_hat = model(x_batch)

      loss = criterion(y_hat, y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      loss_b = loss.item() * x_batch.shape[0]
      rloss += loss_b

    loss_e = rloss/ NoT
    loss_history += [loss_e]
    print(f'Epoch : {ep+1}, train loss : {loss_e:.3f}')
    print('-' * 20)
    
  return loss_history

# 테스트 함수
def Test(model, test_DL):
  model.eval()

  with torch.no_grad():
    rcorrect = 0    
    for x_batch, y_batch in test_DL :
      x_batch = x_batch.to(DEVICE)
      y_batch = y_batch.to(DEVICE)

      y_hat = model(x_batch)

      pred = y_hat.argmax(dim=1)
      print(y_hat.shape)
      print(pred)

      corrects_b = torch.sum(pred == y_batch).item()
      rcorrect += corrects_b

    accuracy_e = rcorrect/len(test_DL.dataset) * 100

  print(f'Test accuracy : {rcorrect} / {len(test_DL.dataset)} ({accuracy_e:.1f} %)')
  
  return round(accuracy_e, 1)

# 테스트 그래프
def Test_plot(model, test_DL):
  model.eval()
  with torch.no_grad():
    x_batch, y_batch = next(iter(test_DL))
    x_batch = x_batch.to(DEVICE)
    y_hat = model(x_batch)
    pred = y_hat.argmax(dim=1)

  x_batch = x_batch.to('cpu')

  plt.figure(figsize=(8,4))
  for idx in range(6):
    plt.subplot(2, 3, idx+1, xticks=[], yticks=[])
    plt.imshow(x_batch[idx].permute(1,2,0), cmap='gray')
    pred_class = test_DL.dataset.classes[pred[idx]]
    true_class = test_DL.dataset.classes[y_batch[idx]]
    plt.title(f'{pred_class} ({true_class})', color='g' if pred_class == true_class else 'r')

# 파라미터 개수얻어오기
def count_params(model):
  num = sum([p.numel() for p in model.parameters() if p.requires_grad])
  return num

# 혼동행렬
def get_conf(model, test_DL):
  N = len(test_DL.dataset.classes)
  model.eval()
  with torch.no_grad():
    confusion = torch.zeros(N, N)
    for x_batch, y_batch in test_DL :
      x_batch = x_batch.to(DEVICE)
      y_batch = y_batch.to(DEVICE)
      y_hat = model(x_batch)
      pred = y_hat.argmax(dim=1)

      confusion += torch.bincount(N * y_batch + pred.cpu(), minlength=N**2).reshape(N,N)
  
  confusion = confusion.numpy()
  return confusion 

# 혼동행렬 그래프
def plot_confusion_matrix(confusion, classes=None):
  N = confusion.shape[0]
  accuracy = confusion / np.sum(confusion) * 100

  plt.figure(figsize=(10, 7))
  plt.imshow(confusion, cmap="Blues")
  plt.title('confusion matrix')
  plt.colorbar()

  for i in range(N) :
    for j in range(N):
        plt.text(j, i, confusion[i,j], ha='center', va='center',
                color='white' 
                if confusion[i,j] > confusion.max() / 1.5 
                else 'black'
                )
  if classes is not None:
    plt.xticks(range(N), classes)
    plt.xticks(range(N), classes)    
  else:
    plt.xticks(range(N))
    plt.yticks(range(N))

  plt.xlabel(f'Perdicted label \n accuracy = {accuracy:.1f} %')
  plt.ylabel('True label')
