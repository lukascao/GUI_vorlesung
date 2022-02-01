# -*- coding: cp936 -*-
"""
This is a Simple Code for compute the specific value of autocorrelation function
Compute Fomular: R(k) = E[(Xi - e)(X(i+k) - e)] / V, as E is Expection function,
e is Expection Value and V is Variance
e = E(Xi) = sum(Xi) / n, i = 1, 2,3 ...n
V = E[(Xi - e)^2]
The Line-Function of Two-point is y = Y0 + k(X - X0), k = (Y1 - Y0)/(X1 - X0)
"""

# Autorized by GY, date 2014-08-05
"""
��������������ͨ���������غ�����0��1���ĺ���ֵ��������ֱ��ͼ���ȡ0������ֵ������R(0.001)
"""
import sys


class File:
    def __init__(self):
        # open a file to read
        FileName = input('Please input the FileName: ')
        self.fs = open(FileName, 'rb')  # All attribute must be self-*

    def FRead(self):
        global buff
        buff = self.fs.read()
        # �����ļ���С
        global size
        size = self.fs.tell()
        print('size is ', size)
        self.fs.close()
        return size


class AR:
    # ��������
    def Expe(self, buff):
        print('Compute the Expection...')
        esum = 0.0
        for i in range(size):
            esum = esum + ord(buff[i])
        self.E = esum / size
        print('The Expection is ', self.E)
        return self.E

    # ���㷽��
    def Vari(self, buff):
        print('Compute the Variance...')
        vsum = 0.0
        for e in buff:
            vsum = vsum + pow((ord(e) - self.E), 2)
        self.V = vsum / size
        print('The Variance is ', self.V)
        return self.V

    # ��������1���ARֵ,��Ϊ��������غ�������AR(0) = 1
    def AR_P(self):
        print('compute the AR-Value...')
        self.tsum = 0.0
        for i in range(size - 1):
            self.tsum += (ord(buff[i]) - self.E) * (ord(buff[i + 1]) - self.E)
        print('self.tsum is ', self.tsum)
        self.ar = self.tsum / ((size - 1) * self.V)
        print('self.ar is :', self.ar)
        return self.ar

    # ��֪�������ֱ��ĳ���yֵ
    def lineP(self, x):
        print('compute the y value...')
        self.value = self.ar + (self.ar - 1) * (x - 1)
        print('AR(', x, ') is ', self.value)
        return self.value


print('This tool is to return AutoFunction(point)...')
fs = File()
if fs.FRead() <= 0:
    print('File Read Error!')
    sys.exit()
print('Now we begin build the AF value...')
ar = AR()
ar.Expe(buff)
print('')
ar.Vari(buff)
ar.AR_P()
x = input("Please input the wanted x value...")
print('The AR() value is ')
ar.lineP(x)
input('Enter for Exit...')