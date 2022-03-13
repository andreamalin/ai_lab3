from matplotlib import pyplot as plt
import numpy as np

# Inciso a
def f_a(inp):
  return (inp[0]**4) + (inp[1]**4) - (4*inp[0]*inp[1]) + (0.5*inp[1]) + 1

def df_a(inp):
  df = (4*(inp[0]**3) - 4*inp[1])*np.array([1., 0.]) + (4*(inp[1]**3) - 4*inp[0] + 0.5)*np.array([0., 1.])
  return df

def f_b(inp):
  x, y = inp
  return 100*(y - x**2)**2 + (1 - x)**2

def df_b(inp):
  x, y = inp
  df = (400*x**3 - 400*x*y + 2*x - 2)*np.array([1., 0.]) + (200*(y - x**2))*np.array([0., 1.])
  return df

def f_c(inp):
  sum = 0
  for i in range(9):
    sum += f_b([inp[i], inp[i+1]])
  return sum

def df_c(inp):
  gradient = []

  gradient.append(df_b([inp[0], inp[1]])[0])
  for i in range(1, 9):
    x, y, z = inp[i-1], inp[i], inp[i+1]
    di = 200*(z - y**2)*(-2*y) - 2*(1 - y) + 200*(y - x**2)
    gradient.append(di)
  gradient.append(200*(inp[9] - inp[8]**2))

  return np.array(gradient)

def maximum_descent(f, df, p0, alpha=0.1, maxI=500, tole=1e-4):
  point_vals = [p0]
  fun_vals = [f(p0)]
  errs = [np.linalg.norm(p0, 1)]
  num_iter = 0
  convergent = 0

  # Continuar con el while
  finish = 0
  point = p0.copy()
  while finish == 0 and num_iter < maxI:
    prev_point = point
    point = prev_point - alpha*df(prev_point) # Se obtiene la direccion

    # Se calcula el error
    error = np.linalg.norm(prev_point - point, 2)
    if (error < tole):
      convergent = 1
      finish = 1

    # Se agregan estos valores para que 
    point_vals.append(point)
    fun_vals.append(f(point))
    errs.append(error)

    num_iter += 1 # Al final se le suma una iteracion
  
  return [fun_vals[-1], errs, point_vals, fun_vals, num_iter, convergent]

def plot_errs(error, label):
  plt.figure()
  plt.plot(error[1:len(error)], label=label)
  plt.xlabel('step')
  plt.ylabel('error')
  plt.show()

if __name__ == '__main__':
  best, errs, point_vals, fun_vals, num_iter, convergent = maximum_descent(f_a, df_a, np.array([0., 0.]), alpha=0.1, maxI=5000)
  best2, errs2, point_vals2, fun_vals2, num_iter2, convergent2 = maximum_descent(f_b, df_b, np.array([2., 2.]), alpha=0.001, maxI=20)
  best3, errs3, point_vals3, fun_vals3, num_iter3, convergent3 = maximum_descent(f_c, df_c, np.array([2., 2., 2., 1., 1., 1., 0., 0., 0., 1.]), alpha=0.001, maxI=20)

  plot_errs(errs, 'a) ')
  plot_errs(errs2, 'b) ')
  plot_errs(errs3, 'c) ')