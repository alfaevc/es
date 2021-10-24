import numpy as np


"""### AT vs FD
"""

def vanilla_gradient(theta, F, sigma=1, N=100):
  epsilons = np.random.standard_normal(size=(N, theta.size))
  fn = lambda x: F(theta + sigma * x) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def FD_gradient(theta, F, sigma=1, N=100):
  # epsilons = np.random.standard_normal(size=(N, theta.size))
  epsilons=orthogonal_epsilons(theta)
  fn = lambda x: (F(theta + sigma * x) - F(theta)) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def AT_gradient(theta, F, sigma=1, N=100):
  #epsilons = np.random.standard_normal(size=(N, theta.size))
  epsilons=orthogonal_epsilons(theta)
  fn = lambda x: (F(theta + sigma * x) - F(theta - sigma * x)) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def orthogonal_epsilons(theta):
    epsilons = np.random.standard_normal(size=(theta.size, theta.size))
    Q, _ = np.linalg.qr(epsilons)#orthogonalize epsilons
    Q_normalize=np.copy(Q)
    fn = lambda x, y: np.linalg.norm(x) * y
    #renormalize rows of Q by multiplying it by length of corresponding row of epsilons
    Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
    #for i in range(theta.size):
    #  norm=np.linalg.norm(epsilons[i])
    #Q_normalize[i]=Q_normalize[i]*norm
    return Q_normalize@Q

def hessian_gaussian_smoothing(theta, F, sigma=1, N=5):
  epsilons = np.random.standard_normal(size=(N, theta.size))
  fn = lambda x: F(theta + sigma * x) 
  second_term=np.mean(np.array(list(map(fn, epsilons))), axis=0)/(sigma**2)
  fn = lambda x: F(theta + sigma * x)*np.outer(x,x)/(N*sigma**2)
  hessian = np.sum(np.array(list(map(fn, epsilons))), axis = 0) - np.identity(theta.size)*second_term
  # hessian=np.zeros((theta.size,theta.size))
  # for i in range(N):
  #   hessian+=F(theta + sigma * epsilons[i])*np.outer(epsilons[i],epsilons[i])/(N*sigma**2)
  # hessian -=np.identity(theta.size)*second_term
  return hessian

def choose_covariate(theta,F,sigma=1,N=5):
    grad=vanilla_gradient(theta, F, sigma=1, N=5)
    hessian=hessian_gaussian_smoothing(theta, F, sigma=1, N=5)
    MSE_AT=((theta.size+2)/N-1)*np.linalg.norm(grad)**2
    MSE_FD=((theta.size+4)/N)*np.sum(np.outer(grad,grad))
    MSE_FD+=((theta.size*2+8)/N-1)*np.linalg.norm(grad)**2
    MSE_FD+=((theta.size+4)*sigma**2/(4*N))*np.linalg.norm(hessian, ord='fro')**2
    MSE_FD+=(5*sigma**2/2)*np.diagonal(hessian)@np.diagonal(hessian)
    choice='AT'
    if (2*N/(N+1))*MSE_AT>MSE_FD:
        choice='FD'
    return choice,MSE_FD,MSE_AT
    
    
def gradascent(theta0, F, method=None, eta=1e-2, max_epoch=200, N=100):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  choice, MSE_FD, MSE_AT=choose_covariate(theta,F,sigma=1,N=theta.size)
  print('best method is ',choice,', MSE of FD is ',MSE_FD,', MSE OF AT is ', MSE_AT)
  for i in range(max_epoch):
    accum_rewards[i] = F(theta)
    if i%10==0:
      print("The return for episode {0} is {1}".format(i, accum_rewards[i]))
    if method == "FD":
      theta += eta * FD_gradient(theta, F, N=N)
    elif method == "AT":
      theta += eta * AT_gradient(theta, F, N=N)
    else: #vanilla
      theta += eta * vanilla_gradient(theta, F, N=N)
  return theta, accum_rewards




