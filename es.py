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

def choose_covariate(theta, policy, sigma=1, N=100):
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
    accum_rewards[i] = policy.eval(theta)
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

def gradascent(theta0, policy, filename, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100):
  theta = np.copy(theta0)
  accum_rewards = np.zeros(max_epoch)
  for i in range(max_epoch): 
    accum_rewards[i] = policy.eval(theta)
    if i%1==0:
      print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))    
      with open(filename, "a") as f:
        f.write("%.d %.2f \n" % (i, accum_rewards[i]))
    
    if method == "AT":
      theta += eta * AT_gradient(theta, policy, sigma, N=N)
    elif method == "FD":
      theta += eta * FD_gradient(theta, policy, sigma, N=N)
    else: #vanilla
      theta += eta * vanilla_gradient(theta, policy, N=N)
  return theta, accum_rewards

def nn_gradascent(actor, policy, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100):
    accum_rewards = np.zeros(max_epoch)
    theta = actor.nnparams2theta()
    for i in range(max_epoch):
      if method == "AT":
        theta += eta * AT_gradient(theta, policy, sigma, N=N)
      elif method == "FD":
        theta += eta * FD_gradient(theta, policy, sigma, N=N)
      else: #vanilla
        theta += eta * vanilla_gradient(theta, policy, N=N)
      if i%1==0:
        new_params = actor.theta2nnparams(theta, policy.input_dim, policy.nn.output_dim)
        actor.update_params(new_params)
        accum_rewards[i] = policy.eval(actor)
        print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))

    return actor, accum_rewards

def nn_twin_gradascent(actor, critic, policy, filename, method=None, sigma=1, eta=1e-3, max_epoch=200, N=100):
    accum_rewards = np.zeros(max_epoch)

    # print(actor.nnparams2theta().size)
    # actor.print_params()
    # print(critic.nnparams2theta().size)
    # critic.print_params()
    theta = np.concatenate((actor.nnparams2theta(), critic.nnparams2theta()))
    # print(theta.size)
    for i in range(max_epoch):
      if method == "AT":
        theta += eta * AT_gradient(theta, policy, sigma, N=N)
      elif method == "FD":
        theta += eta * FD_gradient(theta, policy, sigma, N=N)
      else: #vanilla
        theta += eta * vanilla_gradient(theta, policy, N=N)
      if i%1==0:
        theta_action = theta[:policy.actor_theta_len]
        theta_state = theta[policy.actor_theta_len:]
        act_params = actor.theta2nnparams(theta_action, policy.nA, policy.nA)
        actor.update_params(act_params)
        critic_params = critic.theta2nnparams(theta_state, policy.state_dim, policy.nA)
        critic.update_params(critic_params)
        accum_rewards[i] = policy.eval(actor, critic)
        print("The return for epoch {0} is {1}".format(i, accum_rewards[i]))
        with open(filename, "a") as f:
          f.write("%.d %.2f \n" % (i, accum_rewards[i]))

    return actor, accum_rewards

    



