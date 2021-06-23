import matplotlib.pyplot as plt

class RegressaoQuadratica:
    def __init__(self):
        self.coeficiente_angular = None
        self.coeficiente_linear = None
        self.multiplica_partes = None
        self.X = None
        self.y = None

    def Transposta(self,matriz):
        rez = [[matriz[j][i] for j in range(len(matriz))] for i in range(len(matriz[0]))]
        return rez

    def MultiplicacaoMatrizes(self, matrizA, matrizB):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*matrizA)] for A_row in matrizB]

    def fit(self, X, y):
        self.X = X
        self.y = y

        X = self.AlteraLista(X)
        XT = self.Transposta(X)
        XT_vezes_X = self.MultiplicacaoMatrizes(XT,X)
        XT_vezes_y = self.MultiplicacaoMatrizes(XT,[y])
        XT_vezes_X_inverso = self.inv3(XT_vezes_X)
        self.multiplica_partes = self.MultiplicacaoMatrizes(XT_vezes_X_inverso,XT_vezes_y)
        #self.coeficiente_angular =  self.multiplica_partes[0][1]
        #self.coeficiente_linear =  self.multiplica_partes[0][0]
        return self.multiplica_partes

    def previsao(self,X):
        h = self.MultiplicacaoMatrizes(self.Transposta(self.multiplica_partes),self.alteraListaPredict(X))
        #print("O resultado da predição é: ", h[0][0])
        return h
        #return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*self.Transposta(self.multiplica_partes))] for A_row in X]

    def AlteraLista(self, X):
        lista_UM = []
        for x in range(len(X)):
            lista_UM.append(1)
        h =  [lista_UM, X, self.MatrizQuadrado(X)]
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
                ret[i][j] = sgn * d / det
        return ret

    def mostraGrafico(self, X,y):
        plt.figure(figsize=(5,4), dpi= 80) #criando a fig do gráfico
        plt.scatter(X,y, marker= 'o', s=50, alpha=0.8) #plot dados
        plt.title('Regressão linear')
        plt.xlabel('Feature value (x)')
        plt.ylabel('Target value (y)')
        plt.show()

    def SQtot(self):
        SQtot = 0
        for x in range(len(self.y)):
            SQtot += (self.y[x] - (sum(self.y)/len(self.y))) ** 2
        return SQtot

    def SQres(self):
        SQres = 0
        for x in range(len(self.y)):
            valor = self.y[x]
            prev = self.previsao(self.X[x])[0][0]
            SQres += ((valor - prev) ** 2)
        return SQres

    def R2(self):
        return (1 - (self.SQres()/self.SQtot()))

X_Census = [1900,1910,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000]
y_Census = [759950,919720,919720,1057110,1232030,1316690,1506970,1793230,2032120,2265050,2496330,2814220]

A = RegressaoQuadratica()
A.fit(X_Census,y_Census)
print('A previsão de 2010 no conjunto de dados "US Census Dataset" é: ', A.previsao(2010)[0][0])
A.mostraGrafico(X_Census,y_Census)
print('O conjunto de dados "US Census Dataset" está com R² de: ', A.R2(), '\n')