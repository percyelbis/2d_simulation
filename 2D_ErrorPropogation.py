
import sympy as sp
import numpy as np
import networkx as nx
import math as mth
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# inputs
Y = float(input('Coordena de Inicio Y: '))
X = float(input('Coordena de Inicio X: '))
longitud = float(input('Distancia de cada tramos en metros: '))
num_point = int(input('Numero de Puntos: '))
sl = float(input('Presicion en distancias en mm: '))
sa = float(input('Presicion en angulos en segundos: '))
factor = float(input('Factor de escala para las elispes de error: '))

#-------------------------------------------------------------------------------
sl = sl/1000
sa = sa/3600
fmes = open("simulacion.csv", 'a')
fmes.truncate(0)
fmes.writelines("from;to;distance;direction;distvar;dirvar\n")
l = 0
for i in range(num_point - 1):
    l = longitud + longitud*i
    fmes.writelines("P"+str(i+1)+";"+"P"+str(i+2) + ";" + str(l) + ";" + str(0)+ ";" + str(sl) + ";" + str(sa)+ "\n")
fmes.close()
med = "simulacion.csv"
f = open(med,"r")
first = f.readline()

G =nx.DiGraph()

start =[]
end =[]
obs ={}

"""reads files and adds edge data to networkx dictionary"""

for line in f:
    
    spl = line.split(";")
    From = spl[0]
    To = spl [1]
    Dist = float(spl[2])
    Dir = float(spl[3])
    Sig_Dist = float(spl[4])
    Sig_Dir = float(spl[5])
    G.add_edge(From, To, Distance = Dist, Direction = Dir, Sigma_Dist = Sig_Dist, Sigma_Dir = Sig_Dir)
    start =[From]
    end =[To]
    
    obs['s'+From+To] = Dist
    obs['d'+From+To] = Dir
    obs['sigS'+From+To] = Sig_Dist
    obs['sigD'+From+To] = Sig_Dir




def edge_S(s,e):
    return G.get_edge_data(s,e)['Distance']
    
def edge_D(s,e):
    return G.get_edge_data(s,e)['Direction']

def edge_sS(s,e):
    return G.get_edge_data(s,e)['Sigma Dist']

def edge_sD(s,e):
    return G.get_edge_data(s,e)['Sigma Dir']
    
"""polar for symbolic and polar1 for numerical values"""

def polar(xs,ys,dist,dirt):
    xn = xs + dist*sp.cos(dirt)
    yn = ys + dist*sp.sin(dirt)
    return xn,yn

def polar1(xs,ys,dist,dirt):
    xn = xs + dist*mth.cos(dirt)
    yn = ys + dist*mth.sin(dirt)
    return xn,yn
    
sort = list(nx.topological_sort(G))
root = sort[0]

symlib = {}
symlib_f = []
symlist =[]
varlist = []
prov = {}
prov_f =[]
orderP = []

"""Main loop to find node links, compute polars and create symbols based on calculations used"""

for i in sort:
    
    kids = G.successors(i)
    if i is root:
        
        for j in kids:
            s = sp.symbols('s'+i+j)
            d = sp.symbols('d'+i+j)
            xP1 = sp.symbols('xP1')
            yP1 = sp.symbols('yP1')
            p = polar(xP1,yP1,s,d)           
            xp = p[0]
            yp = p[1]
            symlib[j] = xp,yp
            
            symlib_f.append(xp)
            symlib_f.append(yp)
            sigS = sp.symbols('sigS'+i+j)
            sigD = sp.symbols('sigD'+i+j)
            symlist.append(sigD)
            symlist.append(sigS)
            varlist.append(s)
            varlist.append(d)
            
            p = polar1(X,Y,edge_S(root,j),edge_D(root,j))
            prov[j]={'x':p[0],'y':p[1]}
            prov_f.append(p[0])
            prov_f.append(p[1])
            orderP.append(j)
        
    else:
        if not kids:
            continue
        else:
            for j in kids:
                s = sp.symbols('s'+i+j)
                d = sp.symbols('d'+i+j)
                
                p = polar(symlib[i][0],symlib[i][0],s,d)
                xp = p[0]
                yp = p[1]
                
                symlib[j] = xp,yp

                symlib_f.append(xp)
                symlib_f.append(yp)
                sigS = sp.symbols('sigS'+i+j)
                sigD = sp.symbols('sigD'+i+j)
                symlist.append(sigD)
                symlist.append(sigS)
                varlist.append(s)
                varlist.append(d)
                
                p = polar1(prov[i]['x'],prov[i]['y'],edge_S(i,j),edge_D(i,j))
                prov[j]={'x':p[0],'y':p[1]}
                prov_f.append(p[0])
                prov_f.append(p[1])
                orderP.append(j)

prov[root]={'x':X,'y':Y}
symM = sp.diag(*symlist)
A = sp.Matrix(symlib_f)
J = A.jacobian(varlist)
covV = J*symM*(J.T)   
    
newM = sp.zeros(J.shape[0],J.shape[0])
for i in range(len(covV)):
    newM[i] = covV[i].evalf(subs=obs)
    
"""correlation = sp.eye(J.shape[0])
correval = sp.eye(J.shape[0])

for i in range(J.shape[0]):
    for j in range(J.shape[0]):
        if i<j:
            correlation[i,j] = covV[i,j]/sp.sqrt(covV[i,i]*covV[j,j])
            correlation[j,i] = correlation[i,j]
            correval[i,j] = newM[i,j]/sp.sqrt(newM[i,i]*newM[j,j])
            correval[j,i] = correval[i,j]"""

elips = []
semiA = []
semiB = []
"""Index covariance matrix to calculate semi-major, semi-minor axis and orientation of ellipse"""
for i in range(J.shape[0]):
    for j in range(J.shape[0]):
        if i==j and i%2==0:
            varx = newM[i,j]
            varxy= newM[i,j+1]
            vary= newM[i+1,j+1]
            if varx<0:
                varx = varx**2
            if vary<0:
                vary = vary**2
            if varxy<0:
                varxy = varxy**2
            #print(varx,varxy,vary)
            a = 'a'+str(i)+str(j)
            b = 'b'+str(i)+str(j)
            alp = 'alpha'+str(i)+str(j)
            
            semiA.append((varx+vary+(((varx-vary)**2)+4*varxy)**0.5)/2)
            semiB.append((varx+vary-(((varx-vary)**2)+4*varxy)**0.5)/2)
            elips.append(mth.atan((-2*varxy)/(varx-vary))/2)

nx.draw_networkx(G)
pos = {}
for i in prov.keys():
    pos[i] = prov[i]['x'],prov[i]['y'] 


 
plt.figure()
ax = plt.gca()   
"""Plots error ellipse and draws network of nodes"""
for i in range(len(semiA)):
    ells = Ellipse(xy=[prov[orderP[i]]['x'],prov[orderP[i]]['y']], width= semiB[i]*factor, height= semiA[i]*factor, angle= elips[i]*(180/mth.pi)+270, edgecolor='b', fc='red', lw=2 )
    ax.add_patch(ells)

nx.draw(G,pos, node_size = 30, node_color = 'blue', with_labels = True)
plt.show()