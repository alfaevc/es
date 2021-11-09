import numpy as np


"""### AT vs FD
"""

def vanilla_gradient(theta, policy, sigma=1, N=100):
  epsilons=orthogonal_epsilons(N,theta.size)
  fn = lambda x: policy.F(theta + sigma * x) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def FD_gradient(theta, policy, sigma=1, N=100):
  # epsilons = np.random.standard_normal(size=(N, theta.size))
  epsilons=orthogonal_epsilons(N,theta.size)
  G = policy.F(theta)
  fn = lambda x: (policy.F(theta + sigma * x) - G) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma

def AT_gradient(theta, policy, sigma=1, N=100):
  #epsilons = np.random.standard_normal(size=(N, theta.size))
  epsilons=orthogonal_epsilons(N,theta.size)
  fn = lambda x: (policy.F(theta + sigma * x) - policy.F(theta - sigma * x)) * x
  return np.mean(np.array(list(map(fn, epsilons))), axis=0)/sigma/2

def orthogonal_epsilons(N,dim):
    #assume input N is a multiple of dim. 
    epsilons_N=np.zeros((N,dim))    
    for i in range(0,round(N/dim)):
      epsilons = np.random.standard_normal(size=(dim, dim))
      Q, _ = np.linalg.qr(epsilons)#orthogonalize epsilons
      Q_normalize=np.copy(Q)
      fn = lambda x, y: np.linalg.norm(x) * y
      #renormalize rows of Q by multiplying it by length of corresponding row of epsilons
      Q_normalize = np.array(list(map(fn, epsilons, Q_normalize)))
      epsilons_N[i*dim:(i+1)*dim] = Q_normalize@Q
    #for i in range(theta.size):
    #  norm=np.linalg.norm(epsilons[i])
    #  Q_normalize[i]=Q_normalize[i]*norm
    return epsilons_N

def hessian_gaussian_smoothing(theta, policy, sigma=1, N=100):
  epsilons = orthogonal_epsilons(N,theta.size)
  fn = lambda x: (np.outer(x,x)- np.identity(theta.size))*policy.F(theta + sigma * x)/(sigma**2)
  hessian = np.mean(np.array(list(map(fn, epsilons))), axis=0) 
  return hessian

def choose_covariate(theta,policy,sigma=1,N=100):
    grad=AT_gradient(theta, policy, sigma=sigma, N=2*N)
    hessian=hessian_gaussian_smoothing(theta, policy, sigma=sigma, N=N)
    MSE_AT=(np.linalg.norm(grad)**2)/N
    MSE_FD=np.copy(MSE_AT)
    MSE_FD+=((N+4)*sigma**4/(4*N))*np.linalg.norm(hessian, ord='fro')**2
    diag_hess = np.diagonal(hessian)
    MSE_FD+=(2.5*sigma**4/N) * diag_hess @ diag_hess
    choice = "AT" if (2*N/(N+1))*MSE_AT > MSE_FD else "FD"
    return choice, MSE_FD, MSE_AT
    
    
def gradascent_autoSwitch(theta0, policy, method=None, sigma=0.1, eta=1e-2, max_epoch=200, N=100):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  for i in range(max_epoch):
    accum_rewards[i] = policy.evaluate(theta)
    print("The return for episode {0} is {1}".format(i, accum_rewards[i]))
    if i%10==0:#update method every 20 iterations
      choice, MSE_FD, MSE_AT = choose_covariate(theta,policy,sigma,N=theta.size*5)
      method=choice
      print("method updated to: ", method,', MSE of FD is ', MSE_FD,', MSE OF AT is ', MSE_AT)    
    
    if method == "AT":
      theta += eta * AT_gradient(theta, policy, sigma, N=N)
    else:
      theta += eta * FD_gradient(theta, policy, sigma, N=2*N)#make # of queries for FD and AT the same   

  return theta, accum_rewards, method

def gradascent(theta0, policy, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  for i in range(max_epoch): 
    accum_rewards[i] = policy.evaluate(theta)
    if i%1==0:
      print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))
    
    if method == "AT":
      theta += eta * AT_gradient(theta, policy, sigma, N=N)
    elif method == "FD":
      theta += eta * FD_gradient(theta, policy, sigma, N=N)
    else: #vanilla
      theta += eta * vanilla_gradient(theta, policy, N=N)
  return theta, accum_rewards


def theta2nnparams(theta, input_dim, output_dim, layers=[16,16]):
    params = []
    end_index = input_dim * layers[0]
    params.append(theta[:end_index].reshape((input_dim, layers[0])))
    start_index = end_index
    for i in range(len(layers)-1):
      end_index += layers[i]
      params.append(theta[start_index:end_index])
      start_index = end_index
      end_index += layers[layers[i] * layers[i+1]]
      params.append(theta[start_index:end_index].reshape(layers[i], layers[i+1]))
      start_index = end_index
    
    end_index += layers[-1]
    params.append(theta[start_index:end_index])
    start_index = end_index
    end_index += layers[layers[-1] * output_dim]
    params.append(theta[start_index:end_index].reshape(layers[-1], output_dim))
    params.append(theta[-output_dim:])

    assert theta.size == end_index + output_dim

    return params

def nnparams2theta(params, theta_dim):
    theta = []
    for p in params:
      theta.append(p.reshape(-1))
    theta = np.concatenate(tuple(theta))
    assert theta.size == theta_dim
    return theta


