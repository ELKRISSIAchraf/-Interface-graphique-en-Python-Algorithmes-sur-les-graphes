import string
from PyQt5 import QtCore, QtGui, QtWidgets 
from acceuil import Ui_Form
from matriceadjacent import Ui_Form as matriceadjacente
from graphe import Ui_page3 as Graphe
from bfs import Ui_BFS 
from coût import Ui_cout 
import numpy as np
from bellmanford import Ui_BellmanFord
from dijikstra import Ui_Dijikstra 
from dfs import Ui_Form as dfs
from algorithme import Ui_algorithme as algo
import sys
import numpy as np
from pyvis.network import Network
from IPython.core.display import display, HTML
# -----------------------------------------------------debut du code ----------------------------------------
typegraphe=False
visiteddfs=[]
cas=0
#-------------------------------------------------------page matrice------------------------------------------
def pagematrice():
    if (ui.oriente.isChecked()): 
        global typegraphe
        typegraphe=True
    if (ui.nonoriente.isChecked()) :
        typegraphe=False
    if(ui.oriente.isChecked()  or ui.nonoriente.isChecked()):
        ordre=int(ui.ordegraphe.text())
        uimat.matrice.setColumnCount(ordre)
        uimat.matrice.setRowCount(ordre)
        i=0
        while(i<ordre):
            item = QtWidgets.QTableWidgetItem()
            uimat.matrice.setVerticalHeaderItem(i, item)
            uimat.matrice.setColumnWidth(i,50)
            item = QtWidgets.QTableWidgetItem()
            uimat.matrice.setHorizontalHeaderItem(i, item) 
            uimat.matrice.setRowHeight(i,50) 
            item = QtWidgets.QTableWidgetItem()
            uimat.matrice.setItem(i,i,item)
            i+=1
        i=0
        _translate = QtCore.QCoreApplication.translate
        while(i<ordre):
           item = uimat.matrice.verticalHeaderItem(i)
           item.setText(_translate("Form", chr(65+i)))
           item = uimat.matrice.horizontalHeaderItem(i)
           item.setText(_translate("Form", chr(65+i)))
           item = uimat.matrice.item(i, i)
           item.setText(_translate("Form", "0"))
           i+=1
        mat.show()
        princ.close()
    else :
        princ.close()
        princ.show()
#-----------------------------------------------------les icones-----------------------------------------------
def iconrevenir():
    princ.show()
    mat.close()
def iconavant():
    princ.show()
    mat.close()
#---------------------------------------------------page d'infos-------------------------------------------------
def recuperer() :
    r = uimat.matrice.rowCount();
    taile=0
    i=0
    matrix=[[0]*r for _ in range(r)]
    while(i<r):
      j=0
      while(j<r):
          
             item = uimat.matrice.item(i,j); 
             if(int(item.text())==1) : taile+=1
             matrix[i][j]=int(item.text());
             j+=1
      i+=1
    if(typegraphe==True) :uigraph.typegraphe.setText("Orienté")
    else : uigraph.typegraphe.setText("Non Orienté")
    uigraph.ordregraphe.setText(str(len(matrix)))
    if(typegraphe==True) : 
        uigraph.taillegraphe.setText(str(int(taile)))
    else : 
        uigraph.taillegraphe.setText(str(int(taile/2))) 
    if(len(matrix)==1) : uigraph.densite.setText(str("calcul imposible"))
    else :   uigraph.densite.setText(str(float (2*taile)/(len(matrix)*(len(matrix)-1))))
    i=0
    som=""
    while(i<r): 
        som+=f" {chr(65+i)} "
        i+=1
    uigraph.sommets.setText(som)
    #complet
    if(len(matrix)*(len(matrix)-1)==(taile)) : uigraph.complet.setText("Oui")
    else : uigraph.complet.setText("Non")
   #connexe
    m=[[0]*r for _ in range(r)]
    i=1
    while(i<=r):
        m+= np.linalg.matrix_power(matrix, i)
        i+=1
    i=0
    con=True
    while(i<r):
      j=0
      while(j<r):
             if(m[i][j]==0) : con=False
             j+=1
      i+=1
    
    if(con==True) :  uigraph.connexe.setText("Oui")
    else : uigraph.connexe.setText("Non")
    #reguliere
    v=np.ones(r)
    valp, vecp = np.linalg.eig(matrix)
    et=False
    i=0
    while(i<len(valp)) :
      if(np.allclose(np.matmul(matrix,v),(valp[i]*v))) : et=True
      i+=1
    if(et==True): uigraph.reguliere.setText("Oui")
    else : uigraph.reguliere.setText("Non")
    page3.show()
def typegraphe() :
    if(ui.oriente.isChecked):
       ordre=int(ui.ordegraphe.text())
    elif(ui.nonoriente.isChecked):
       print(ordre)
       princ.close()
    else :
        princ.close()
        princ.show()
def oriente():
    #if(ui.nonoriente.isChecked()):
      ui.nonoriente.setChecked(False)
def nonoriente():
    #if(ui.oriente.isChecked()):
     ui.oriente.setChecked(False)
#-------------------------------------------affichage du graphe--------------------------------------------------
def voirgraphe() :
    if(typegraphe==True) :  net2= Network(directed= True)
    else : net2= Network(directed= False)
    nodes=[]
    labels=[]
    edges=[]
    colors=[]
    r = uimat.matrice.rowCount()
    for index in range(r):
      nodes.append(index+1)
      labels.append(chr(65+index))
      colors.append('#E15B01')
    net2.add_nodes(nodes, label=labels,color=colors)
    i=0
    matrix=[[0]*r for _ in range(r)]
    while(i<r):
      j=0
      while(j<r):
             item = uimat.matrice.item(i,j); 
             matrix[i][j]=int(item.text());
             j+=1
      i+=1
    i=1
    print(matrix)
    for row in matrix:
       j=1
       for column in row:
           if column==1:
             edges.append((i,j))
           j=j+1
       i=i+1
    net2.add_edges(edges)
    display(HTML("edges.html"))
    net2.show('edges.html')
def rvalgor():
    page3.show()
    algorithme.close()
    #----------------------------------------------algorithmes--------------------------------------------------
def algorithmes():
    uialgo.flow.setText("")
    uialgo.source.setText("")
    uialgo.puits.setText("")
    algorithme.show()
#------------------------------------------------Warshall-------------------------------------------------------
def warshall():
    if(typegraphe==True) :  net2= Network(directed= True)
    else : net2= Network(directed= False)
    nodes=[]
    labels=[]
    edges=[]
    colors=[]
    r = uimat.matrice.rowCount()
    for index in range(r):
      nodes.append(index+1)
      labels.append(chr(65+index))
      colors.append('#E15B01')
    net2.add_nodes(nodes, label=labels,color=colors)
    i=0
    matrix=[[0]*r for _ in range(r)]
    while(i<r):
      j=0
      while(j<r):
             item = uimat.matrice.item(i,j); 
             matrix[i][j]=int(item.text());
             j+=1
      i+=1
    i=1
    for row in matrix:
       j=1
       for column in row:
           if column==1:
             edges.append((i,j))
           j=j+1
       i=i+1
    net2.add_edges(edges)
    for y in nodes:
      for z in nodes:
        if (y!=z) and ((y,z) in edges):
            for x in nodes:
                if (x != y ) and ((x,y) in edges):
                   #colors=colors.append('#01E13D')
                   #net2.add_nodes(color=colors)
                   if(not (x,z) in edges) :   net2.add_edge(x,z,color="#05FB47")
                  
    display(HTML("edges.html"))
    net2.show('edges.html')
#---------------------------------------------------BFS---------------------------------------------------------
def BFS(starting_node):
    nodes=[]
    edges=[]
    labels=[]
    r = uimat.matrice.rowCount()
    for index in range(r):
      nodes.append(index+1)
      labels.append(index+1)
    i=0
    matrix=[[0]*r for _ in range(r)]
    while(i<r):
      j=0
      while(j<r):
             item = uimat.matrice.item(i,j); 
             matrix[i][j]=int(item.text());
             j+=1
      i+=1
    i=1
    for row in matrix:
       j=1
       for column in row:
           if column==1:
             edges.append((i,j))
           j=j+1
       i=i+1
    visited = []
    queue = [starting_node]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            for edge in edges:
              if(typegraphe==True):
                if edge[0] == node:
                    queue.append(edge[1])
              else :
                if edge[0] == node:
                    queue.append(edge[1])
                elif edge[1] == node:
                    queue.append(edge[0])
    return visited
def afficherbfs():
    r = uimat.matrice.rowCount()
    uibfs.tableWidget.setRowCount(r)
    i=0
    while(i<r) :
      item = QtWidgets.QTableWidgetItem()
      font = QtGui.QFont()
      font.setPointSize(10)
      font.setBold(True)
      font.setWeight(75)
      item.setFont(font)
      uibfs.tableWidget.setVerticalHeaderItem(i, item) 
      #uibfs.tableWidget.setRowHeight(i,50) 
      item = QtWidgets.QTableWidgetItem()
      uibfs.tableWidget.setItem(i,0,item)
      i+=1
    i=0
    _translate = QtCore.QCoreApplication.translate
    while(i<r):
      item = uibfs.tableWidget.verticalHeaderItem(i)
      item.setText(_translate("BFS",chr(65+i)))
      item = uibfs.tableWidget.item(i,0)
      list=BFS(i+1)
      if(len(list)==1): item.setText(_translate("BFS","Ø"))
      else :
       A=[]
       res=""
       for node in list :
        j=1
        while(j<=r) :
          if(node==j): A.append(chr(65+(j-1)))
          j+=1
       for sommet in A :
        res+=" {} ".format(sommet)
       item.setText(_translate("BFS",res))
      i+=1 
    bfs.show()
#-------------------------------------------------------DFS---------------------------------------------------
def DFs(starting_node):
    nodes=[]
    edges=[]
    labels=[]
    r = uimat.matrice.rowCount()
    for index in range(r):
      nodes.append(index+1)
      labels.append(index+1)
    i=0
    matrix=[[0]*r for _ in range(r)]
    while(i<r):
      j=0
      while(j<r):
             item = uimat.matrice.item(i,j); 
             matrix[i][j]=int(item.text());
             j+=1
      i+=1
    i=1
    for row in matrix:
       j=1
       for column in row:
           if column==1:
             edges.append((i,j))
           j=j+1
       i=i+1
    def explorer(starting_node):
     global visiteddfs
     visiteddfs.append(starting_node)
     for edge in edges:
        if edge[0]==starting_node:
            next=edge[1]
            if next not in visiteddfs:
                explorer(next)
    
    explorer(starting_node)
def afficherdfs() :
    r = uimat.matrice.rowCount()
    uidfs.tableWidget.setRowCount(r)
    i=0
    while(i<r) :
      item = QtWidgets.QTableWidgetItem()
      font = QtGui.QFont()
      font.setPointSize(10)
      font.setBold(True)
      font.setWeight(75)
      item.setFont(font)
      uidfs.tableWidget.setVerticalHeaderItem(i, item) 
      #uibfs.tableWidget.setRowHeight(i,50) 
      item = QtWidgets.QTableWidgetItem()
      uidfs.tableWidget.setItem(i,0,item)
      i+=1
    i=0
    _translate = QtCore.QCoreApplication.translate
    while(i<r):
      item = uidfs.tableWidget.verticalHeaderItem(i)
      item.setText(_translate("Form",chr(65+i)))
      item = uidfs.tableWidget.item(i,0)
      DFs(i+1)
      if(len(visiteddfs)==1): item.setText(_translate("Form","Ø"))
      else :
       A=[]
       res=""
       for node in visiteddfs :
        j=1
        while(j<=r) :
          if(node==j): A.append(chr(65+(j-1)))
          j+=1
       for sommet in A :
        res+=" {} ".format(sommet)
       item.setText(_translate("Form",res))
      i+=1 
      visiteddfs.clear()
    DFS.show()
    #---------------------------------------------recuperer les couts----------------------------------
def Cout() :
      #(uialgo..clicked()) : n=2
      r = uimat.matrice.rowCount()
      uicout.tableWidget.setColumnCount(r)
      uicout.tableWidget.setRowCount(r)
      i=0
      i=0
      matrix=[[0]*r for _ in range(r)]
      while(i<r):
        j=0
        while(j<r):
             item = uimat.matrice.item(i,j); 
             matrix[i][j]=int(item.text());
             j+=1
        i+=1
      i=0
      while(i<r):
            item = QtWidgets.QTableWidgetItem()
            uicout.tableWidget.setVerticalHeaderItem(i, item)
            uicout.tableWidget.setColumnWidth(i,50)
            item = QtWidgets.QTableWidgetItem()
            uicout.tableWidget.setHorizontalHeaderItem(i, item) 
            uicout.tableWidget.setRowHeight(i,50) 
            """if (typegraphe!=True) :
             j=0
             while(j<i): 
               item = QtWidgets.QTableWidgetItem()
               uicout.tableWidget.setItem(i,j,item)
               j+=1
              """
            j=0
            while(j<r):
              if(matrix[i][j]==0) :
                item = QtWidgets.QTableWidgetItem()
                uicout.tableWidget.setItem(i,j,item)
              j+=1
            i+=1
      i=0
      _translate = QtCore.QCoreApplication.translate
      while(i<r):
           item = uicout.tableWidget.verticalHeaderItem(i)
           item.setText(_translate("cout", chr(65+i)))
           item = uicout.tableWidget.horizontalHeaderItem(i)
           item.setText(_translate("cout", chr(65+i)))
           """
           if (typegraphe!=True) :
             j=0
             while(j<i): 
                item1 = uicout.tableWidget.item(j,i);
                item = uicout.tableWidget.item(i, j)
                item.setText(_translate("cout", ))
                j+=1
           
           """
           j=0
           while(j<r):
              if(matrix[i][j]==0) :
                item = uicout.tableWidget.item(i, j)
                item.setText(_translate("cout", "------"))
              j+=1
           i+=1
      


      COUT.show()
      #-----------------------------------------KRUSKAL-------------------------------------------------------
def KRUSKAL() :
    if(typegraphe==True) :  net3= Network(directed= True)
    else : net3= Network(directed= False)
    nodes=[]
    labels=[]
    listt=[]
    n=[]
    colors=[]
    labelsEdges=[]   
    r = uimat.matrice.rowCount()
    for index in range(r):
      nodes.append(index+1)
      labels.append(chr(index+65))
      colors.append('#02FC45')
    net3.add_nodes(nodes,label=labels,color=colors)
    i=0
    matrix=[[0]*r for _ in range(r)]
    while(i<r):
      j=0
      while(j<r):
             item = uicout.tableWidget.item(i,j); 
             matrix[i][j]=item.text();
             j+=1
      i+=1
    i=0
    while(i<r):
      j=0
      while(j<r):
             if(matrix[i][j]!="------"):
                 #edges.append((i+1,j+1))
                 #item = uicout.tableWidget.item(i,j); 
                 labelsEdges.append(int(matrix[i][j]));
                 listt.append([i+1,j+1,int(matrix[i][j])])
             j+=1
      i+=1
      """
      i=1 
    for row in matrix:
       j=1
       for column in row:
           if (column == "1"):
             edges.append((i,j))
             item = uicout.tableWidget.item(i,j); 
             labelsEdges.append(int(item.text()));
             listt.append([i,j,item.text()])
           j=j+1
       i=i+1
       """
    def hasCycle(src,dest,edges5):
     if len(edges5)==1:
        n.append(0)
        return 0
     if src==dest:
        n.append(1)
        return 1
     else:
        for edge in edges5:
            if edge[0]==dest:
                next=edge[1]
                hasCycle(src,next,edges5)
    def Kruskal():
     labelsEdges.sort()
     edges2 = []
     list1=[]
     if(typegraphe==True):
       for label in labelsEdges:
        for ls in listt:
            if ls[2]==label:
                edges2.append((ls[0], ls[1]))
                hasCycle(ls[0], ls[1],edges2)
                #print(n)
                try:
                    x = n.pop(0)
                except:
                    x=0
                if x==1:
                    edges2.pop()
                else:
                    net3.add_edge(ls[0],ls[1],label=ls[2])
     else :
      for label in labelsEdges:
        for ls in listt:
            if ls[2]==label:
                edges2.append((ls[0], ls[1]))
                hasCycle(ls[0], ls[1],edges2)
                #print(n)
                try:
                    x = n.pop(0)
                except:
                    x=0
                if x==1:
                    list1.append(edges2.pop())
      for edge in edges2:
        if (edge[0],edge[1]) not in list1:
          for ls in listt:
              if ls[0]==edge[0] and ls[1]==edge[1]:
                net3.add_edge(edge[0],edge[1],label=ls[2])  
     net3.show('edges.html')
     display(HTML("edges.html"))
    #print(labelsEdges)
    Kruskal()
    #print(labelsEdges)
    listt.clear()
    labelsEdges.clear()
    #---------------------------------------------------PRIM----------------------------------------------
def PRIM() :
    if(typegraphe==True) :  net3= Network(directed= True)
    else : net3= Network(directed= False)
    INF = 9999999
    nodes=[]
    labels=[]
    colors=[]
    # number of vertices in graph
    V = uimat.matrice.rowCount()
    for index in range(V):
      nodes.append(index+1)
      labels.append(chr(index+65))
      colors.append('#02FC45')
    net3.add_nodes(nodes,label=labels,color=colors)
    # create a 2d array of size 5x5
    # for adjacency matrix to represent graph
   #G = [[0, 9, 75, 0, 0],
         #[9, 0, 95, 19, 42],
         #[75, 95, 0, 51, 66],
         #[0, 19, 51, 0, 31],
         #[0, 42, 66, 31, 0]]
    # create a array to track selected vertex
    # selected will become true otherwise false
    selected =[]
    G=[[0]*V for _ in range(V)]
    for i in range(V):
        selected.append(0)
    # set number of edge to 0
    no_edge = 0
    # the number of egde in minimum spanning tree will be
    # always less than(V - 1), where V is number of vertices in
    # graph
    # choose 0th vertex and make it true
    selected[0] = True
    # print for edge and weight
    print("Edge : Weight\n")
    while (no_edge < V - 1):
        # For every vertex in the set S, find the all adjacent vertices
        # , calculate the distance from the vertex selected at step 1.
        # if the vertex is already in the set S, discard it otherwise
        # choose another vertex nearest to selected vertex  at step 1.
        minimum = INF
        x = 0
        y = 0
        i=0
        while(i<V) :
          j=0
          while(j<V) :
             item = uicout.tableWidget.item(i,j); 
             if(item.text()!='------') : G[i][j]=int(item.text())
             else :  G[i][j]=0 
             j+=1
          i+=1
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if ((not selected[j]) and G[i][j]):
                        # not in selected and there is an edge
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        #print(str(x+1) + "-" + str(y+1) + ":" + str(G[x][y]))
        net3.add_edge(x+1, y+1, label=G[x][y])
        selected[y] = True
        no_edge += 1
    net3.show('edges.html')
    display(HTML("edges.html"))
    #----------------------------------------------BellmanFord-----------------------------------------------
def Bellmanford() :
  _translate = QtCore.QCoreApplication.translate
  uibelman.tableWidget.setRowCount(2)
  item = QtWidgets.QTableWidgetItem()
  font = QtGui.QFont()
  font.setPointSize(7)
  font.setBold(True)
  item.setFont(font)
  uibelman.tableWidget.setVerticalHeaderItem(0, item)
  item = QtWidgets.QTableWidgetItem()
  font = QtGui.QFont()
  font.setPointSize(7)
  font.setBold(True)
  item.setFont(font)
  uibelman.tableWidget.setVerticalHeaderItem(1, item)     
  item = uibelman.tableWidget.verticalHeaderItem(0)
  item.setText(_translate("BellmanFord", "L"))
  item = uibelman.tableWidget.verticalHeaderItem(1)
  item.setText(_translate("BellmanFord", "P"))  
  r = uimat.matrice.rowCount()
  uibelman.tableWidget.setColumnCount(r)
  i=0
  while(i<r):
            item = QtWidgets.QTableWidgetItem()
            uibelman.tableWidget.setHorizontalHeaderItem(i, item) 
            uibelman.tableWidget.setColumnWidth(i,50)
            i+=1 
  i=0
  while(i<r):
           item = uibelman.tableWidget.horizontalHeaderItem(i)
           item.setText(_translate("BellmanFord", chr(65+i)))
           i+=1
  matrix=[[0]*r for _ in range(r)]
  i=0
  while(i<r):
      j=0
      while(j<r):
             item = uicout.tableWidget.item(i,j); 
             matrix[i][j]=item.text();
             j+=1
      i+=1
  def bellmanFord(graph, sommetDepart):
    distances = {} 
    predecesseurs = {}
    for sommet in graph:
        distances[sommet] = np.inf
        predecesseurs[sommet] = None
    distances[sommetDepart] = 0
    for i in range(len(graph)-1):
        for j in graph:
            for k in graph[j]: 
                if distances[k] > distances[j] + graph[j][k]:
                    distances[k]  = distances[j] + graph[j][k]
                    predecesseurs[k] = j
    for i in graph:
        for j in graph[i]:
          if(distances[j] > distances[i] + graph[i][j]) : return distances, predecesseurs,True
    return distances, predecesseurs,False
  g={}
  i=0
  while(i<len(matrix)) :
    j=0
    go={}
    while(j<len(matrix)) :
      if(matrix[i][j]!='------') :
        go[chr(65+j)]=int(matrix[i][j])
      j+=1
    g[chr(65+i)]=go
    i+=1
  BLM.show()
  def bl() :
     sommet=uibelman.sommet.text()
     L,P,cbe=bellmanFord(g,sommet)
     print(cbe)
     if(cbe==True): uibelman.inter.setText(_translate("BellmanFord", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\"> Attention !! détection du circuit absorbant</span></p></body></html>"))
     elif(len(BFS(ord(sommet)-64))==1): 
       uibelman.inter.setText(_translate("BellmanFord", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\"> Attention !! ce sommet n'as pas de succeseur</span></p></body></html>"))
       
       uibelman.textBrowser.clear()
       i=0
       while(i<r) :
        j=0
        while(j<r):
              item = QtWidgets.QTableWidgetItem()
              uibelman.tableWidget.setItem(i,j,item)
              j+=1
        i+=1
     
       i=0
       while(i<r) :
          item = uibelman.tableWidget.item(0,i)
          item.setText(_translate("BellmanFord",""))
          item = uibelman.tableWidget.item(1,i)
          item.setText(_translate("BellmanFord",""))
          i+=1
     else :
      i=0
      while(i<r) :
        j=0
        while(j<r):
              item = QtWidgets.QTableWidgetItem()
              uibelman.tableWidget.setItem(i,j,item)
              j+=1
        i+=1
     
      i=0
      while(i<r) :
          item = uibelman.tableWidget.item(0,i)
          item.setText(_translate("BellmanFord",str(L[chr(65+i)])))
          item = uibelman.tableWidget.item(1,i)
          item.setText(_translate("BellmanFord",str(P[chr(65+i)])))
          i+=1
      uibelman.inter.setText(_translate("BellmanFord", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Interpretation d'algorithme de BellmanFord :</span></p></body></html>"))
      uibelman.textBrowser.clear()
      
      i=0
      while(i<r) :
       if(chr(65+i)!=sommet) :
        t=[]
        ch="{}".format(sommet)
        t.append(chr(65+i))
        p=chr(65+i)
        while(P[p]!=sommet and P[p] is not None) :
          p=P[p]
          t.append(p)
        t.reverse()
        j=0
        while(j<len(t)) :
          ch+="->{}".format(t[j])
          j+=1
        if(L[chr(65+i)] == np.inf) : ch=" chemin entre {} et {}  n'existe pas !!!".format(sommet,chr(65+i))
        else: ch+=" : {}".format(L[chr(65+i)])
        #print(ch)
        html = """<p>{code}</p>""".format(code=ch)
        uibelman.textBrowser.append(_translate("BellmanFord",html))
       i+=1
       
     #-------------------------
     L.clear() 
     P.clear()
  uibelman.sommetbutton.clicked.connect(bl)
  #-----------------------------------------------DIJIKSTRA--------------------------------------------------
def DIJIKSTRA() :

  _translate = QtCore.QCoreApplication.translate
  uidij.tableWidget.setRowCount(2)
  item = QtWidgets.QTableWidgetItem()
  font = QtGui.QFont()
  font.setPointSize(7)
  font.setBold(True)
  item.setFont(font)
  uidij.tableWidget.setVerticalHeaderItem(0, item)
  item = QtWidgets.QTableWidgetItem()
  font = QtGui.QFont()
  font.setPointSize(7)
  font.setBold(True)
  item.setFont(font)
  uidij.tableWidget.setVerticalHeaderItem(1, item)     
  item = uidij.tableWidget.verticalHeaderItem(0)
  item.setText(_translate("Dijikstra", "L"))
  item = uidij.tableWidget.verticalHeaderItem(1)
  item.setText(_translate("Dijikstra", "P"))  
  r = uimat.matrice.rowCount()
  uidij.tableWidget.setColumnCount(r)
  i=0
  while(i<r):
            item = QtWidgets.QTableWidgetItem()
            uidij.tableWidget.setHorizontalHeaderItem(i, item) 
            uidij.tableWidget.setColumnWidth(i,50)
            i+=1 
  i=0
  while(i<r):
           item = uidij.tableWidget.horizontalHeaderItem(i)
           item.setText(_translate("Dijikstra", chr(65+i)))
           i+=1
  matrix=[[0]*r for _ in range(r)]
  i=0
  while(i<r):
      j=0
      while(j<r):
             item = uicout.tableWidget.item(i,j); 
             matrix[i][j]=item.text();
             j+=1
      i+=1
  def bellmanFord(graph, sommetDepart):
    distances = {} 
    predecesseurs = {}
    for sommet in graph:
        distances[sommet] = np.inf
        predecesseurs[sommet] = None
    distances[sommetDepart] = 0
    for i in range(len(graph)-1):
        for j in graph:
            for k in graph[j]: 
                if distances[k] > distances[j] + graph[j][k]:
                    distances[k]  = distances[j] + graph[j][k]
                    predecesseurs[k] = j
    """                
    for i in graph:
        for j in graph[i]:
          if(distances[j] > distances[i] + graph[i][j]) : return distances, predecesseurs,True
    """ 
    return distances, predecesseurs
  g={}
  i=0
  while(i<len(matrix)) :
    j=0
    go={}
    while(j<len(matrix)) :
      if(matrix[i][j]!='------') :
        go[chr(65+j)]=int(matrix[i][j])
      j+=1
    g[chr(65+i)]=go
    i+=1
  Dijikstra.show()
  def bl() :
     sommet=uidij.sommet.text()
     L,P=bellmanFord(g,sommet)
     
     
     if(len(BFS(ord(sommet)-64))==1): 
       uidij.inter.setText(_translate("Dijikstra", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\"> Attention !! ce sommet n'as pas de succeseur</span></p></body></html>"))
       
       uidij.textBrowser.clear()
       i=0
       while(i<r) :
        j=0
        while(j<r):
              item = QtWidgets.QTableWidgetItem()
              uidij.tableWidget.setItem(i,j,item)
              j+=1
        i+=1
     
       i=0
       while(i<r) :
          item = uidij.tableWidget.item(0,i)
          item.setText(_translate("Dijikstra",""))
          item = uidij.tableWidget.item(1,i)
          item.setText(_translate("Dijikstra",""))
          i+=1
     else :
      i=0
      while(i<r) :
        j=0
        while(j<r):
              item = QtWidgets.QTableWidgetItem()
              uidij.tableWidget.setItem(i,j,item)
              j+=1
        i+=1
     
      i=0
      while(i<r) :
          item = uidij.tableWidget.item(0,i)
          item.setText(_translate("Dijikstra",str(L[chr(65+i)])))
          item = uidij.tableWidget.item(1,i)
          item.setText(_translate("Dijikstra",str(P[chr(65+i)])))
          i+=1
      uidij.inter.setText(_translate("Dijikstra", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Interpretation d'algorithme de Dijikstra :</span></p></body></html>"))
      uidij.textBrowser.clear()
      
      i=0
      while(i<r) :
       if(chr(65+i)!=sommet) :
        t=[]
        ch="{}".format(sommet)
        t.append(chr(65+i))
        p=chr(65+i)
        while(P[p]!=sommet and P[p] is not None) :
          p=P[p]
          t.append(p)
        t.reverse()
        j=0
        while(j<len(t)) :
          ch+="->{}".format(t[j])
          j+=1
        if(L[chr(65+i)] == np.inf) : ch=" chemin entre {} et {}  n'existe pas !!!".format(sommet,chr(65+i))
        else: ch+=" : {}".format(L[chr(65+i)])
        #print(ch)
        html = """<p>{code}</p>""".format(code=ch)
        uidij.textBrowser.append(_translate("Dijikstra",html))
       i+=1
       
     #-------------------------
     L.clear() 
     P.clear()
  uidij.sommetbutton.clicked.connect(bl)
  #--------------------------------------------COLORIAGE------------------------------------------------------
def coloriage():
 if(typegraphe==True) :  net4= Network(directed= True)
 else : net4= Network(directed= False)
 r = uimat.matrice.rowCount()
 Matrice=[[0]*r for _ in range(r)]
 
 edges=[]
 i=0
 while(i<r):
      j=0
      while(j<r):
          
             item = uimat.matrice.item(i,j); 
             Matrice[i][j]=int(item.text());
             if(Matrice[i][j]!=0): edges.append((i+1,j+1))
             j+=1
      i+=1
 l1=[]
 temp=[]
 color=0
 i=1
 list1=[]
 list=[]
 for row in Matrice:
    j=1
    val=0
    for column in row:
        if column==1:
            val=val+1
        j=j+1
    list1.append([val,i])
    i=i+1
 for l11 in sorted(list1,key=lambda x:(-x[0],x[1])):
    list.append(l11[1])
 for node in list:
    for l in l1:
        temp.append(l[0])
    if node not in temp:
        color=color+1
        for node1 in list:
            if node!=node1 and (node,node1) not in edges and (node1 not in temp):
                l1.append([node1,color])
        l1.append([node, color])
    temp.clear()
 for sommet_color in l1:
    if sommet_color[1]==1:
        net4.add_node(sommet_color[0], label=chr(65+sommet_color[0]-1), color='#228B22')
    if sommet_color[1]==2:
        net4.add_node(sommet_color[0], label=chr(65+sommet_color[0]-1), color='#EE2C2C')
    if sommet_color[1]==3:
        net4.add_node(sommet_color[0], label=chr(65+sommet_color[0]-1), color='#0000FF')
    if sommet_color[1]==4:
        net4.add_node(sommet_color[0], label=chr(65+sommet_color[0]-1), color='#E3CF57')
 for edge in edges:
    net4.add_edge(edge[0],edge[1])
 net4.show('edges.html')
 display(HTML("edges.html"))
 #-----------------------------------------------ford--------------------------------------------------------
def Ford(): 
  r = uimat.matrice.rowCount()
  i=0
  edges=[]
  G=[[0]*r for _ in range(r)]
  while(i<r):
      j=0
      while(j<r):
             item = uicout.tableWidget.item(i,j); 
             if(item.text()=='------') : G[i][j]==0
             else : 
               G[i][j]=int(item.text())
               edges.append((i+1,j+1))
             j+=1
      i+=1
  def BFSS(G, s, t, parent):
    # Return True if there is node that has not iterated.
    visited = [False] * len(G)
    queue = []
    queue.append(s)
    visited[s] = True
    while queue:
        u = queue.pop(0)
        for ind in range(len(G[u])):
            if visited[ind] is False and G[u][ind] > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
    return True if visited[t] else False
  def FordFulkerson(G, source, sink):
    # This array is filled by BFS and to store path
    parent = [-1] * (len(G))
    max_flow = 0
    while BFSS(G, source, sink, parent):
        path_flow = float("Inf")
        s = sink
        while s != source:
            # Find the minimum value in select path
            path_flow = min(path_flow, G[parent[s]][s])
            s = parent[s]

        max_flow += path_flow
        v = sink
        while v != source:
            u = parent[v]
            G[u][v] -= path_flow
            G[v][u] += path_flow
            v = parent[v]
    return max_flow
  source=uialgo.source.text()
  puits=uialgo.puits.text()
  sourc, sink = ord(source)-64, ord(puits)-64
  uialgo.flow.setText("FlowMax = {}".format(FordFulkerson(G, sourc-1, sink-1)))
  #--------------------------------FIN des algos--------------------------------------------------
def traiter() :
  if(cas==1) : KRUSKAL()
  elif(cas==2) : PRIM()
  elif(cas==3) :Bellmanford()
  elif(cas==4) :DIJIKSTRA()
  elif(cas==5) :Ford()
  else : print("noooooooooooooooooooooooon")
def coutkruskal():
  global cas 
  cas=1
  Cout()
def coutbelman():
  global cas 
  cas=3
  Cout()
def coutprim():
  global cas 
  cas=2
  Cout()
def coutdijikstra():
  global cas 
  cas=4
  Cout()
def coutford():
  global cas 
  cas=5
  i=0
  edges=[]
  r = uimat.matrice.rowCount()
  while(i<r):
      j=0
      while(j<r):
             item = uimat.matrice.item(i,j); 
             if(int(item.text())!=0) :edges.append((i+1,j+1))
             j+=1
      i+=1
  s=True
  a=True
  source=uialgo.source.text()
  puits=uialgo.puits.text()
  sourc, sink = ord(source)-64, ord(puits)-64
  print(sourc)
  print(sink)
  lis=BFS(sourc)
  for edge in edges:
        if edge[1]==sourc:
          s=False  
        if edge[0]==sink:
          a=False
  if(typegraphe==False ) :uialgo.flow.setText("Impossible Graphe n'est pas orienté ")
  elif(s==False) :  uialgo.flow.setText("Impossible Source a un predecesseur")
  elif(a==False) :  uialgo.flow.setText("Impossible Puits a un successeur")
  elif(sink not in lis) : uialgo.flow.setText("Impossible chemin n'existe pas")
  #elif(  uialgo.source.selectedText()==string.empty or  uialgo.puits.selectedText()==string.empty ) : uialgo.flow.setText("Entrer d'abord la source et le puits")
  else:Cout()
def RV() :
  COUT.close()
def quitterdfs() :
  DFS.close()
def quitterbfs() :
  bfs.close()
def quitterbellman():
  BLM.close()
def quitterdijikstra():
  Dijikstra.close()
import sys
app = QtWidgets.QApplication(sys.argv)
princ = QtWidgets.QWidget()
ui = Ui_Form()
ui.setupUi(princ)
#creation de la matrice d'adjacent
mat = QtWidgets.QWidget()
uimat = matriceadjacente()
uimat.setupUi(mat)
#creation du graphe
page3 = QtWidgets.QWidget()
uigraph = Graphe()
uigraph.setupUi(page3)
#algorithmes :
algorithme = QtWidgets.QWidget()
uialgo = algo()
uialgo.setupUi(algorithme)
#bfs
bfs = QtWidgets.QWidget()
uibfs = Ui_BFS()
uibfs.setupUi(bfs)

#dfs
DFS = QtWidgets.QWidget()
uidfs = dfs()
uidfs.setupUi(DFS)
#cout
COUT = QtWidgets.QWidget()
uicout = Ui_cout()
uicout.setupUi(COUT)
#belmanford
BLM = QtWidgets.QWidget()
uibelman = Ui_BellmanFord()
uibelman.setupUi(BLM)
#dijikstra
Dijikstra = QtWidgets.QWidget()
uidij = Ui_Dijikstra()
uidij.setupUi(Dijikstra)
princ.show()
# les signaux 
ui.star.clicked.connect(pagematrice)
ui.oriente.clicked.connect(oriente)
ui.nonoriente.clicked.connect(nonoriente)
uimat.iconrevenir.clicked.connect(iconrevenir)
uimat.iconavant.clicked.connect(recuperer)
uigraph.voirgraphe.clicked.connect(voirgraphe)
uigraph.algorithmes.clicked.connect(algorithmes)
uialgo.rvalgo.clicked.connect(rvalgor)
uialgo.warshall.clicked.connect(warshall)
uialgo.bfs.clicked.connect(afficherbfs)
uialgo.dfs.clicked.connect(afficherdfs)
uialgo.kruskal.clicked.connect(coutkruskal) 
uialgo.prim.clicked.connect(coutprim)
uialgo.bellmanford.clicked.connect(coutbelman)
uialgo.dijkstra.clicked.connect(coutdijikstra)
uialgo.fordfulkerson.clicked.connect(coutford)
uialgo.coloriage.clicked.connect(coloriage)
uicout.click2cout.clicked.connect(traiter)
uicout.click1cout.clicked.connect(RV)
uibfs.bfsrevenir.clicked.connect(quitterbfs)
uidfs.dfsrevenir.clicked.connect(quitterdfs)
uibelman.bellmanrevenir.clicked.connect(quitterbellman)
uidij.commandLinkButton.clicked.connect(quitterdijikstra)
sys.exit(app.exec_())





