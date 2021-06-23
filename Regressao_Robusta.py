class RegressaoRobust:
    def __init__(self):
        self.coeficiente_angular = None
        self.coeficiente_linear = None
        self.multiplica_partes = None
        self.X = None
        self.y = None
        self.Matriz_W = None
        self.XT = None
        self.X_Alterado = None
        self.nova_matriz_X_para_WLLS = None
        self.nova_matriz_y_para_WLLS = None
        self.XT_WLLS = None
        self.multiplica_partes_WLLS = None
        self.valor_para_predizer = None
    

    def Transposta(self,matriz):
        rez = [[matriz[j][i] for j in range(len(matriz))] for i in range(len(matriz[0]))]
        return rez

    def MultiplicacaoMatrizes(self, matrizA, matrizB):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*matrizA)] for A_row in matrizB]

    def fit(self, X, y):
        self.X = X
        self.y = y
        X = self.AlteraLista(X)
        self.XT = self.Transposta(X)
        XT_vezes_X = self.MultiplicacaoMatrizes(self.XT,X)
        XT_vezes_y = self.MultiplicacaoMatrizes(self.XT,[y])
        XT_vezes_X_inverso = self.inv3(XT_vezes_X)
        self.multiplica_partes = self.MultiplicacaoMatrizes(XT_vezes_X_inverso,XT_vezes_y)
        return self.multiplica_partes

    def previsao(self,X):
        self.valor_para_predizer = X
        a = X[0]*self.multiplica_partes[0][1] + X[1]*self.multiplica_partes[0][2] + self.multiplica_partes[0][0]
        return a

    def AlteraLista(self, X):
        lista_UM = []
        for x in range(len(X[0])):
            lista_UM.append(1)
        h =  [lista_UM, X[0], X[1]]
        self.X_Alterado = h
        return h

    def MatrizQuadrado(self,X):
        return [x*x for x in X]

    def alteraListaPredict(self, X):
     return [[1,X, X*X]]

    def det2(self,A):
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]

    def inv2(self,A):
        d = self.det2(A)
        return [[A[1][1]/d, -A[0][1]/d], [-A[1][0]/d, A[0][0]/d]]
    
    def det3(self,B):
        ret = 0
        for i in range(3):
            pos = 1
            neg = 1
            for j in range(3):
                pos *= B[j][(i+j) % 3]
                neg *= B[j][(i-j) % 3]
            ret += (pos - neg)
        return ret

    def inv3(self,B):
        ret = [3*[None] for _i in range(3)]
        det = self.det3(B)
        for i in range(3):
            for j in range(3):
                adj = [[n for ii, n in enumerate(row) if ii != i] 
                        for jj, row in enumerate(B) if jj != j]
                d = self.det2(adj)
                sgn = (-1)**(i+j)
                if(det != 0):
                    ret[i][j] = sgn * d / det
                else:
                    ret[i][j] = 0
        return ret

    def Calcula_W(self,X,y):
        matriz_W = []
        for x in range(len(X[0])):
            prev  = self.previsao([X[0][x],X[1][x]])
            w = 1/(y[x]-prev)
            if w < 0:
                w = w * -1
            matriz_W.append(w)
        
        self.Matriz_W = matriz_W

        return matriz_W

    def Transforma_X_para_WLLS(self):

        primeira_parte_X = []
        segunda_parte_X =[]

        primeira_parte_y = []
        segunda_parte_y =[]

        for item in range(len(self.X[0])):
            valor_W1_X = self.X[0][item] * self.Matriz_W[item]
            valor_W2_X = self.X[1][item] * self.Matriz_W[item]
            
            valor_W1_y = self.y[item] * self.Matriz_W[item]


            primeira_parte_X.append(valor_W1_X)
            segunda_parte_X.append(valor_W2_X)

            primeira_parte_y.append(valor_W1_y)


        self.nova_matriz_X_para_WLLS = [primeira_parte_X, segunda_parte_X]
        self.nova_matriz_y_para_WLLS = primeira_parte_y

    def fit_WLLS(self):
        X = self.AlteraLista(self.nova_matriz_X_para_WLLS)
        self.XT_WLLS = self.Transposta(X)
        XT_vezes_X = self.MultiplicacaoMatrizes(self.XT_WLLS,X)
        XT_vezes_y = self.MultiplicacaoMatrizes(self.XT,[self.nova_matriz_y_para_WLLS])
        XT_vezes_X_inverso = self.inv3(XT_vezes_X)
        self.multiplica_partes_WLLS = self.MultiplicacaoMatrizes(XT_vezes_X_inverso,XT_vezes_y)
        return self.multiplica_partes_WLLS

    def previsao_WLLS(self, valor_predicao):
        a = valor_predicao[0]*self.multiplica_partes_WLLS[0][1] + valor_predicao[1]*self.multiplica_partes_WLLS[0][2] + self.multiplica_partes_WLLS[0][0]
        #h = self.MultiplicacaoMatrizes(self.Transposta(self.multiplica_partes),self.alteraListaPredict(X))
        return a

        

    def WLLS(self,X):

        self.Transforma_X_para_WLLS()
        self.fit_WLLS()
        retorno = self.previsao_WLLS(X)
        return retorno

X = [[0,1,1,0,2,4,4,1,4,3,0,2,1,4,1,0,1,3,0,1,4,4,0,2,3,1,0,3,3,2,2,3,2,2,3,4,4,3,1,2,0],[9,15,15,10,16,10,20,11,20,15,15,8,13,18,10,8,10,16,11,19,12,11,19,15,15,20,6,15,19,14,13,17,20,11,20,20,20,9,8,16,10]]
y = [45,57,57,45,51,65,88,44,87,89,59,66,65,56,47,66,41,56,37,45,58,47,64,97,55,51,61,69,79,71,62,87,54,43,92,83,94,60,56,88,62]


A = RegressaoRobust()
A.fit(X,y)
print('Regressão Linear: ', A.previsao([0,9]))
matriz_W = A.Calcula_W(X,y)
#print(matriz_W)
print('Regressão Linear Robusta: ', A.WLLS([0,9]))