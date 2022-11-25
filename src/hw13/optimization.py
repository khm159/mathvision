import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from numpy.linalg import norm
import math 
from numpy import exp
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

def get_hessian_matrix(x,y):
    """return hessian matrix"""
    return np.array([
        [dxdx(x,y), dxdy(x,y)],
        [dydx(x,y), dydy(x,y)]
    ])
def get_hessian_matrix2(x,y):
    """return hessian matrix"""
    return np.array([
        [dxdx2(x,y), dxdy2(x,y)],
        [dydx2(x,y), dydy2(x,y)]
    ])

def show_2d_heatmap(x,y,z):
    #HEATMAP OUTPUT   
    z_min = np.min(z)
    z_max = np.max(z)
    plt.pcolormesh(
        x,y,z, 
        vmin=z_min, 
        vmax=z_max,
        cmap=cm.coolwarm
    )
    plt.colorbar(
        cmap=cm.coolwarm)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

def show_2d_heatmap_history(x,y,z, moving):
    #HEATMAP OUTPUT   
    z_min = np.min(z)
    z_max = np.max(z)
    plt.title("{} iter : x {} y {} z {}".format(
        len(moving),
        np.round(moving[-1][0],4),
        np.round(moving[-1][1],4),
        np.round(f2(moving[-1][0],moving[-1][1]),4)
    ))
    plt.pcolormesh(
        x,y,z, 
        vmin=z_min, 
        vmax=z_max,
        cmap=cm.coolwarm
    )
    for pt in moving:
        print("pt", pt)
        plt.scatter(
            pt[0], pt[1], s=10, c="#FF5733"
        )

    plt.colorbar(
        cmap=cm.coolwarm)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

def show_3d_surface(x,y,z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
    #ax.set_xlim(-1,1.5)
    #ax.set_ylim(-1.2,0.2)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def show_3d_surface_history(x,y,z, moving):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    ax.set_title("{} iter : x {} y {} z {}".format(
        len(moving),
        np.round(moving[-1][0],4),
        np.round(moving[-1][1],4),
        np.round(f2(moving[-1][0],moving[-1][1]),4)
    ))

    for elem in moving:
        plt.scatter(elem[0], elem[1],s=10,c="#FF5733")

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def show_2d_surface_critical_pt(x,y,z,cris,maxs,mins):
    z_min = np.min(z)
    z_max = np.max(z)
    plt.pcolormesh(
        x,y,z, 
        vmin=z_min, 
        vmax=z_max,
        cmap=cm.coolwarm
    )
    for elem in cris:
        plt.scatter(elem[0], elem[1],s=30,color="green",label="saddle")
    for elem in maxs:
        plt.scatter(elem[0], elem[1],s=30,color="red",label="local_maxima")
    for elem in mins:
        plt.scatter(elem[0], elem[1],s=30,color="blue",label="local_minima")

    plt.colorbar(
        cmap=cm.coolwarm)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

def show_2d_surface_pt_grad_vec(x,y,z,tgt_pt):
    #HEATMAP OUTPUT   
    z_min = np.min(z)
    z_max = np.max(z)
    plt.pcolormesh(
        x,y,z, 
        vmin=z_min, 
        vmax=z_max,
        cmap=cm.coolwarm
    )
    plt.scatter(tgt_pt[0], tgt_pt[1],s=10)
    plt.colorbar(
        cmap=cm.coolwarm)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

def show_3d_surface_pt_grad_vec(x,y,z,tgt_pt):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
                       
    ax.scatter(tgt_pt[0],tgt_pt[1],tgt_pt[2], c='r', marker='o',s=100)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlim(-1,1.5)
    ax.set_ylim(-1.2,0.2)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def show_3d_surface_critical(x,y,z,cris,maxs=None,mins=None):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) 
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)
                       
    for c_pt in cris:
        ax.scatter(c_pt[0],c_pt[1],f(c_pt[0],c_pt[1]), c='r', marker='o',s=100)
    if maxs is not None:
        for max_pt in maxs:
            ax.scatter(c_pt[0],c_pt[1],f(max_pt[0],max_pt[1]), c='g', marker='o',s=100)
    if mins is not None:
        for min_pt in mins:
            ax.scatter(c_pt[0],c_pt[1],f(min_pt[0],min_pt[1]), c='b', marker='o',s=100)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlim(-1,1.5)
    ax.set_ylim(-1.2,0.2)

    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def dxdx(x,y):
    """dxdx"""
    # 2y^2 + 2y
    return (2*y*y) + 2*y

def dydy(x,y):
    """dydy"""
    # 6xy + 2x + 2x^2
    return (6*x*y) + 2*x + (2*x*x)

def dxdy(x,y):
    """dxdy"""
    # 3y^2 + 2y + 4xy + 2x
    return (3*y*y) + 2*y + 4*x*y + 2*x

def dydx(x,y):
    """dydx"""
    # 4xy + 2x + 3y^2 + 2y
    return (4*x*y) + 2*x + (3*y*y) + 2*y

#==========
def dxdx2(x,y):
    """dxdx2"""
    # -sin(x+y-1) +2
    return -np.sin(x+y-1)+2

def dydy2(x,y):
    """dydy2"""
    # -sin(x+y-1) +2
    return -np.sin(x+y-1)+2

def dxdy2(x,y):
    """dxdy2"""
    # -sin(x+y-1) -2
    return -np.sin(x+y-1)-2

def dydx2(x,y):
    """dydx2"""
    # -sin(x+y-1) -2
    return -np.sin(x+y-1)-2
#===========


def partial_derivative_x(x,y):
    """pre-calculated gradient_x"""
    # delta f_x : 
    # 2y^2x + 2yx + y^3 + y^2
    return (2*y*y*x) + (2*y*x) + (y*y*y) + (y*y)

def partial_derivative_y(x,y):
    """pre-calculated gradient_y"""
    # delta f_y : 
    # 3xy^2 + 2xy + 2x^2y + x^2
    return (3*x*y*y) + (2*x*y) + (2*x*x*y) + x*x

def partial_derivative_x2(x,y):
    """pre-calculated gradient_x"""
    # delta f2_x:
    # cos(x+y-1) + (2x-2y-2) - 1.5
    return np.cos(x+y-1) + (2*x-2*y-2) - 1.5

def partial_derivative_y2(x,y):
    """pre-calculated gradient_y"""
    # delta f2_y:
    # cos(x+y-1) + (-2x+2y+2) + 2.5
    return np.cos(x+y-1)+(-2*x+2*y+2) + 2.5

def gradf2(x, y):
    return np.array(
        [partial_derivative_x2(x, y), 
        partial_derivative_y2(x, y)]
    )   

def f(x, y):
    return (x*x*y)+(x*x*y*y)+(x*y*y)+(x*y*y*y)

def f2(x,y):
    term_a = np.sin(x+y-1)
    term_b = (x-y-1)*(x-y-1)
    term_c = -1.5*x + 2.5*y+1
    return term_a+term_b+term_c

def classify_critical_point(w):
    negative = 0
    positive = 0
    for elem in w:
        if elem>0:
            positive +=1
        elif elem<0:
            negative +=1
    if negative==len(w):
        return "local_min"
    elif positive==len(w):
        return "local_max"
    else:
        return "mixture"

def min_max_scaling(z):
    return(z-np.min(z))/(np.max(z)-np.min(z))

def uniform_initialization_2d(x_range, y_range):
    import random
    x = random.uniform(x_range[0], x_range[1])
    y = random.uniform(y_range[0], y_range[1])
    return np.array([x,y])

def grad_descent(f, gradf, init_t, alpha):
    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()

    max_iter = 1000
    iter = 0
    pts = [] 
    pts.append(init_t)
    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*gradf(t[0], t[1])
        pts.append(t.copy())
        print (t, f(t[0], t[1]), gradf(t[0], t[1]))
        iter += 1
    return pts

def newton_method(f, gradf,init_t):

    EPS = 1e-5
    prev_t = init_t-10*EPS
    t = init_t.copy()

    max_iter = 1000
    iter = 0
    pts = [] 
    pts.append(init_t)

    while norm(t - prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        H_f = get_hessian_matrix2(t[0], t[1])
        inv_h_f = np.linalg.inv(H_f)
        t -= np.matmul(inv_h_f,gradf(t[0], t[1]))
        pts.append(t.copy())
        print (t, f(t[0], t[1]), gradf(t[0], t[1]))
        iter += 1
    return pts


    
if __name__ =="__main__":
    # [Q1.1] =============================
    # -1<=x<=1.5, -1.2<=y<=0.2 구간에서 
    # 이 함수의 그래프를 도시하시오.

    # 1. Define function range
    X = np.arange(-1,   1.5, 0.01)
    Y = np.arange(-1.2, 0.2, 0.01)
    X, Y = np.meshgrid(X, Y)

    # 2. get z-values 
    z = f(X,Y)   
    
    # 3. visualization the searching-space
    show_3d_surface(X,Y,z)
    show_2d_heatmap(X,Y,z)

    # [Q1.2] =============================
    # (1, 0)에서 f의 gradient를 구한 후, 구한 gradient가 
    # 함수의 최대 증가 방향과 일치함을 그래프를 통해 확인하시오
    g_x = partial_derivative_x(1,0)
    g_y = partial_derivative_y(1,0)
    gradient = [g_x, g_y] # <-- [sin theta, cos theta]
    theta = np.arctan2(g_x, g_y)
    tgt_point = [1,0,f(1,0)]
    print("Gradient : ", gradient)
    print("target point", tgt_point)
    show_3d_surface_pt_grad_vec(
        x=X,
        y=Y,
        z=z,
        tgt_pt=tgt_point
    )
    show_2d_surface_pt_grad_vec(
        x=X,
        y=Y,
        z=z,
        tgt_pt=tgt_point
    )

    # [Q1.3] =============================
    # Get critical points using the Hessian-matrix
    # and Eigen decomposition 

    # Grid-Search 
    x_grid = np.around(np.arange(-1,   1.5, 0.1),2)
    y_grid = np.around(np.arange(-1.2, 0.2, 0.1),2)

    maxs = [] 
    mins = []
    cris = [] 
    for x in x_grid:
        for y in y_grid:
            # 1. gradient calculation
            g_x = partial_derivative_x(x,y)
            g_y = partial_derivative_y(x,y)
            g_norm = math.sqrt(g_x*g_x+g_y*g_y)
            # is critical points? 
            if g_norm ==0:
                # 2. get hessian matrix H_f
                H_f = get_hessian_matrix(x,y)
                w,v = np.linalg.eig(H_f)
                res = classify_critical_point(w)
                print(" critical point : ",[x,y],w)
                if res == "local_min":
                    print("    --> local minima")
                    mins.append([x,y])
                elif res =="local_max":
                    print("    --> local maxima")
                    maxs.append([x,y])
                else:
                    cris.append([x,y])
    show_3d_surface_critical(
        x = X,
        y = Y,
        z = z,
        cris=cris,
        maxs=maxs,
        mins=mins
    )
    show_2d_surface_critical_pt(
        x=X,
        y=Y,
        z=z,
        cris=cris,
        maxs=maxs,
        mins=mins
    )

    # checking 
    eps = 0.0001
    print("PT1 :", 0, -1, f(0,-1))
    print("PT1+x : ", 0+eps, -1, f(0+eps,-1))
    print("PT1-x : ", 0-eps, -1, f(0-eps,-1))
    print("PT1+y : ", 0, -1+eps, f(0,-1+eps))
    print("PT1-y : ", 0, -1-eps, f(0,-1-eps))
    
    # [Q3.1] =============================
    # 그래프 도시 
    # 1. Define function range
    X = np.arange(-1, 5, 0.01)
    Y = np.arange(-3, 4, 0.01)
    X, Y = np.meshgrid(X, Y)

    # 2. get z-values 
    z = f2(X,Y)   
    
    # 3. visualization the searching-space
    show_3d_surface(X,Y,z)
    show_2d_heatmap(X,Y,z)

    # [Q3.2] =============================
    # 그래디언트 디센트 방법 

    # 1. initialization 
    # start = uniform_initialization_2d(
    #     x_range=[-1,5],
    #     y_range=[-3,4]
    # )

    #start = np.array([2.5, -0.5])
    start = np.array([-0.5, 3.3])
    

    # 2. searching 
    moving_history = grad_descent(
        f=f2,
        gradf=gradf2,
        init_t=start,
        alpha=0.01
    )

    show_2d_heatmap_history(X, Y, z, moving_history)
    show_3d_surface_history(X, Y, z, moving_history)

    # [Q3.3] =============================
    # 뉴턴 방법 

    # 2. searching 
    moving_history = newton_method(
        f=f2,
        gradf=gradf2,
        init_t=start
    )
    show_2d_heatmap_history(X, Y, z, moving_history)
    show_3d_surface_history(X, Y, z, moving_history)


    

                

            
    

    

    







    
