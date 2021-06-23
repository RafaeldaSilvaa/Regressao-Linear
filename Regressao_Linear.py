import matplotlib.pyplot as plt

class RegressaoLinear:
    def __init__(self):
        self.coeficiente_angular = None
        self.coeficiente_linear = None
        self.multiplica_partes = None
        self.X = None
        self.y = None

    def Transposta(self,matriz):
        rez = [[matriz[j][i] for j in range(len(matriz))] for i in range(len(matriz[0]))]
        return rez

    def transposeMatrix(self, matriz):
        return map(list,zip(*matriz))

    def getMatrixMinor(self, matriz,i,j):
        return [row[:j] + row[j+1:] for row in (matriz[:i]+matriz[i+1:])]

    def getMatrixDeternminant(self, matriz):
        #base case for 2x2 matrix
        if len(matriz) == 2:
            return matriz[0][0]*matriz[1][1]-matriz[0][1]*matriz[1][0]

        determinant = 0
        for c in range(len(matriz)):
            determinant += ((-1)**c)*matriz[0][c]*self.getMatrixDeternminant(self.getMatrixMinor(matriz,0,c))
        return determinant

    def getMatrixInverse(self, matriz):
        determinant = self.getMatrixDeternminant(matriz)
        #special case for 2x2 matrix:
        if len(matriz) == 2:
            return [[matriz[1][1]/determinant, -1*matriz[0][1]/determinant],
                    [-1*matriz[1][0]/determinant, matriz[0][0]/determinant]]

        #find matrix of cofactors
        cofactors = []
        for r in range(len(matriz)):
            cofactorRow = []
            for c in range(len(matriz)):
                minor = self.getMatrixMinor(matriz,r,c)
                cofactorRow.append(((-1)**(r+c)) * self.getMatrixDeternminant(minor))
            cofactors.append(cofactorRow)
        cofactors = self.transposeMatrix(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c]/determinant
        return cofactors

    def MultiplicacaoMatrizes(self, matrizA, matrizB):
        return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*matrizA)] for A_row in matrizB]

    def fit(self, X, y):
        self.X = X
        self.y = y

        X = self.AlteraLista(X)
        XT = self.Transposta(X)
        XT_vezes_X = self.MultiplicacaoMatrizes(XT,X)
        XT_vezes_y = self.MultiplicacaoMatrizes(XT,[y])
        XT_vezes_X_inverso = self.getMatrixInverse(XT_vezes_X)
        self.multiplica_partes = self.MultiplicacaoMatrizes(XT_vezes_X_inverso,XT_vezes_y)
        self.coeficiente_angular =  self.multiplica_partes[0][1]
        self.coeficiente_linear =  self.multiplica_partes[0][0]
        return self.multiplica_partes

    def mostraGrafico(self, X,y):
        plt.figure(figsize=(5,4), dpi= 80) #criando a fig do gráfico
        plt.scatter(X,y, marker= 'o', s=50, alpha=0.8) #plot dados
        plt.title('Regressão linear')
        plt.xlabel('Feature value (x)')
        plt.ylabel('Target value (y)')
        plt.show()


    def previsao(self,X):
        retorno = self.MultiplicacaoMatrizes(self.Transposta(self.multiplica_partes),self.alteraListaPredict(X))
        #print("O resultado da predição é: ", retorno[0][0])
        return retorno

    def AlteraLista(self, X):
        lista_UM = []
        for x in range(len(X)):
            lista_UM.append(1)
        h =  [lista_UM, X]
        return h

    def alteraListaPredict(self, X):
        return [[1,X]]

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

X_slides =  [69,67,71,65,72,68,74,65,66,72]
y_slides = [9.5,8.5,11.5,10.5,11,7.5,12,7,7.5,13]

X_Water = [194.5,194.3,194.3,197.9,198.4,199.4,199.9,200.9,201.1,201.4,201.3,203.6,204.6,209.5,208.6,210.7,211.9,212.2]
y_Water = [20.79,20.79,20.79,22.4,22.67,23.15,23.35,23.89,23.99,24.02,24.01,25.14,26.57,28.49,27.76,29.04,29.88,30.06]

X_Census = [1900,1910,1910,1920,1930,1940,1950,1960,1970,1980,1990,2000]
y_Census = [759950,919720,919720,1057110,1232030,1316690,1506970,1793230,2032120,2265050,2496330,2814220]

A = RegressaoLinear()
A.fit(X_slides,y_slides)
print('A previsão de altura 70 no conjunto de dados "Height x Shoe Size" é: ', A.previsao(70)[0][0])
A.mostraGrafico(X_slides,y_slides)
print('O conjunto de dados "Height x Shoe Size" está com R² de: ', A.R2(), '\n')

B = RegressaoLinear()
B.fit(X_Water,y_Water)
print('A previsão de 194 no conjunto de dados "Alps Water" é: ', B.previsao(194)[0][0])
B.mostraGrafico(X_Water,y_Water)
print('O conjunto de dados "Alps Water" está com R² de: ', B.R2(), '\n')

C = RegressaoLinear()
C.fit(X_Census,y_Census)
print('A previsão de 2010 no conjunto de dados "US Census Dataset" é: ', C.previsao(2010)[0][0])
C.mostraGrafico(X_Census,y_Census)
print('O conjunto de dados "US Census Dataset" está com R² de: ', C.R2(), '\n')