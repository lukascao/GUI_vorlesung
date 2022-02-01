# import os
import sys
import csv
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# import pandas as pd
# from pandas.plotting import lag_plot
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import *
# from scipy.stats.stats import pearsonr
from statsmodels.tsa.arima_process import arma_generate_sample
# from sklearn.metrics import mean_squared_error as MSE
import math
# -- -- #

import distribution_displays

# -- -- #

import functools
import inspect
from typing import Optional


# -- -- #
import Simulation_fit
# -- -- #
from ellipse import confidence_ellipse, get_correlated_dataset
from wave_g import createWave, saveWave, reafWave


class Example(QWidget):
    label: list[Optional[list[QLabel]]]
    le: list[Optional[list[QLineEdit]]]
    intline: list[Optional[list[QLineEdit]]]
    pb: list[Optional[list[QPushButton]]]
    combobox: list[Optional[QComboBox]]
    stack: list[Optional[QWidget]]
    sp: list[Optional[QSpinBox]]
    group: list[Optional[list[QGroupBox]]]

    def __init__(self):
        super().__init__()

        # -- -- #

        self.n_pages = 13

        # -- -- #

        self.tree = None
        self.tree_dict = None
        self.stackedWidget = None
        self.stack = [None] * self.n_pages
        self.le = [None] * self.n_pages
        self.var_dict = None
        self.groupbox = None
        self.link21 = None
        self.intline = [None] * self.n_pages
        self.sp = [None] * self.n_pages
        self.pb = [None] * self.n_pages
        self.icon = None
        self.label = [None] * self.n_pages
        self.filename = None
        self.group = [None] * self.n_pages
        self.combobox = [None] * self.n_pages

        # -- -- #

        self.initUI()

    'This is the main window,left is tree menu and right part are the stacked windows.'

    def initUI(self):

        self.setFixedSize(700, 450)
        self.setWindowTitle('Tool of SMNt')
        self.setStyleSheet("background-color:'silver'")
        hbox = QHBoxLayout(self)
        left = QFrame(self)
        left.setFixedSize(235, 450)
        right = QFrame(self)
        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.setSizes([35, ])

        splitter1.addWidget(left)
        splitter1.addWidget(right)
        hbox.addWidget(splitter1)
        self.setLayout(hbox)

        self.tree = QTreeWidget(left)

        list_00 = [self.tree.setMinimumSize, self.tree.setStyleSheet, self.tree.setAutoScroll,
                   self.tree.setEditTriggers, self.tree.setTextElideMode, self.tree.setRootIsDecorated,
                   self.tree.setUniformRowHeights, self.tree.setItemsExpandable, self.tree.setAnimated,
                   self.tree.setHeaderHidden, self.tree.setExpandsOnDoubleClick, self.tree.setObjectName]
        list_01 = [[35, 450], ["background-color:'silver';border:outset;color:seagreen;font:bold;font-size:15px"],
                   [True], [QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed], [Qt.ElideMiddle],
                   [True], [False], [True], [False], [True], [True], ["tree"]]
        for i, j in zip(list_00, list_01):
            i(*j)

        self.tree_dict = {}
        list_02 = ["root", "root1", "root2", "root3", "child11", "child12", "child13", "child14", "child21", "child22",
                   "child25", "child31", "child32", "child33"]
        list_03 = [None, None, None, None, "root1", "root1", "root1", "root1", "root2",
                   "root2", "root2", "root3", "root3", "root3"]
        list_04 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        list_05 = ['HomePage', 'Probability', 'Correlation Function', 'Estimation Function', 'Coin Flipping',
                   'Dice Throw', 'Distribution Simulation', 'Central Limit Theorem', 'Correlation Coefficient',
                   'Correlation Function', 'AR Model', 'Maximum likelihood', 'confidence ellipse', "waveforms"]
        for i, j, k, l in zip(list_02, list_03, list_04, list_05):
            if j is None:
                self.tree_dict[i] = QTreeWidgetItem(self.tree)
            else:
                self.tree_dict[i] = QTreeWidgetItem(self.tree_dict[j])
            self.tree_dict[i].setText(k, l)

        self.tree.addTopLevelItem(self.tree_dict["root"])

        self.stackedWidget = QStackedWidget(right)

        self.stack = [QWidget() for i in range(13)]

        for func in [self.stackUI0, self.stackUI1, self.stackUI2, self.stackUI3, self.stackUI4, self.stackUI5,
                     self.stackUI6, self.stackUI7, self.stackUI8, self.stackUI9, self.stackUI10, self.stackUI11,
                     self.stackUI12]:
            func()

        for i in range(len(self.stack)):
            self.stackedWidget.addWidget(self.stack[i])

        self.tree.clicked.connect(self.Display)

    'Change the stacked windows.'

    def Display(self):

        item = self.tree.currentItem()

        switcher = {
            "HomePage": 0,
            "Coin Flipping": 1,
            "Dice Throw": 2,
            "Distribution Simulation": 3,
            "Central Limit Theorem": 4,
            "Correlation Coefficient": 5,
            "Correlation Function": 6,
            "AR Model": 9,
            "Maximum likelihood": 10,
            "confidence ellipse": 11,
            "waveforms": 12
        }

        i = switcher.get(item.text(0), None)
        if i is not None:
            self.stackedWidget.setCurrentIndex(i)

    '----------------Homepage------------------------------'

    def stackUI0(self):

        layout = QVBoxLayout(self.stack[0])

        self.label[0] = [QLabel()]
        self.label[0][0].setText("Statistische Methoden \nder Nachrichtentechnik\nVer.1.0")
        self.label[0][0].setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        self.label[0][0].setAlignment(Qt.AlignCenter)
        self.label[0][0].setFont(QFont("Sanserif", 15, QFont.Bold))

        layout.addWidget(self.label[0][0])

    '---------------------Coin----------------------------'

    def stackUI1(self):
        vlayout = QVBoxLayout(self.stack[1])

        gridlayout = QGridLayout()
        grid = QWidget()
        grid.setLayout(gridlayout)
        vlayout.addWidget(grid)

        self.le[1] = [QLineEdit() for i in range(2)]

        self.var_dict = {}

        list_00 = ["pb1_2", "pb1_1", "help"]
        list_01 = ["Execute", "Clear", "Help"]
        list_02 = [self.coin, self.clear11, self.msg1]
        for i, j, k in zip(list_00, list_01, list_02):
            self.var_dict[i] = QPushButton(j)
            self.var_dict[i].clicked.connect(k)

        for i, j in zip(["label11", "label12"], ["Times:", "Probability:"]):
            self.var_dict[i] = QLabel()
            self.var_dict[i].setText(j)

        list_03 = [self.var_dict["help"], self.var_dict["label11"], self.var_dict["label12"], *self.le[1],
                   self.var_dict["pb1_2"], self.var_dict["pb1_1"]]
        list_04 = [1, 2, 3, 2, 3, 4, 5]
        list_05 = [2, 0, 0, 1, 1, 2, 2]
        for i in zip(list_03, list_04, list_05):
            gridlayout.addWidget(*i)

    def msg1(self):
        QMessageBox.about(self, "Help", "This function is a simulator of coin flipping.\n"
                                        "Input the number of flipping times and the probability of head.\n"
                                        "The simulator will generate a graphic of probability change polyline.\n"
                                        "Can generate multiple images simultaneously.")

    def coin(self):
        try:
            times = []
            frequency = []
            n_heads = 0
            n_instances = 0
            number = int(self.le[1][0].text())
            probability = float(self.le[1][1].text())
            'when number of trials is smaller than 100,we need a solid sampling interval'

            for flip_num in range(0, number):
                if random.random() <= probability:
                    n_heads += 1
                n_instances += 1
                frequency.append(n_heads / n_instances)
                times.append(n_instances)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(times, frequency, color='b', label='(actual) relative occurrence')
            ax.set_xlabel('Number of Trials')
            ax.set_ylabel('Frequency')
            ax.set_ylim([0, 1])
            ax.set_xlim([1, number])
            'red line is the probability of head'
            ax.plot([-0, number], [probability, probability], label='Theoretical Probability', color='r',
                    linewidth='1.5')
            ax.legend(loc=0)
            plt.title('{} times Flip with Probability p(head)= {} '.format(number, probability))
            plt.grid(axis='y')
            plt.show()
        except:
            self.Error()

    def clear11(self, index=1):
        for i in self.le[index]:
            i.clear()

    '---------------------Dice----------------------------'

    def stackUI2(self):

        vlayout = QVBoxLayout(self.stack[2])
        gridlayout = QGridLayout()
        gridlayout2 = QGridLayout()
        grid = QWidget()
        grid2 = QWidget()
        grid.setLayout(gridlayout)
        grid2.setLayout(gridlayout2)

        self.groupbox = QGroupBox('5 Platonic Solids', self)
        self.groupbox.setLayout(gridlayout)

        self.link21 = QLabel()
        self.link21.setOpenExternalLinks(True)
        self.link21.setText(u'<a href="https://en.wikipedia.org/wiki/Platonic_solid" style="color:#0000ff;">'
                            u'<b>Wikipedia</b></a>')
        self.link21.setStyleSheet('font-size: 11px')

        self.intline[2] = [QLineEdit() for i in range(3)]

        self.pb[2] = []
        self.icon = {}
        list_00 = ["1", "2", "3", "4", "5"]
        list_01 = ["4.png", "6.png", "8.png", "12.png", "20.png"]
        list_02 = ["4", "6", "8", "12", "20"]
        list_03 = ["Tetrahedron 4 faces", "Cube 6 faces", "Octahedron 8 faces", "Dodecahedron 12 faces",
                   "Icosahedron 20 faces"]

        for j, k, l, m in zip(list_00, list_01, list_02, list_03):
            self.pb[2].append(QPushButton())
            self.icon[j] = QIcon()
            self.icon[j].addPixmap(QPixmap(k), QIcon.Normal, QIcon.Off)
            self.pb[2][-1].setIcon(self.icon[j])
            self.pb[2][-1].setIconSize(QSize(50, 50))
            self.pb[2][-1].clicked.connect(functools.partial(self.intline[2][1].setText, l))
            self.pb[2][-1].setToolTip(m)

        for j, k in zip(["Help", "Execute", "Clear"], [self.msg2, self.dice_simulation, self.clear21]):
            self.pb[2].append(QPushButton(j))
            self.pb[2][-1].clicked.connect(k)

        self.label[2] = [QLabel(text) for text in ["Number of dice:", "Faces of dice:", "Throw times:"]]

        list_07 = [*self.pb[2][:5], self.link21, *self.pb[2][5:], *self.intline[2], *self.label[2]]
        list_08 = [2, 2, 2, 2, 2, 3, 1, 5, 6, 2, 3, 4, 2, 3, 4]
        list_09 = [0, 1, 2, 3, 4, 4, 2, 2, 2, 1, 1, 1, 0, 0, 0]
        for i in zip(list_07[:6], list_08[:6], list_09[:6]):
            gridlayout.addWidget(*i)
        for i in zip(list_07[6:], list_08[6:], list_09[6:]):
            gridlayout2.addWidget(*i)

        vlayout.addWidget(self.groupbox)
        vlayout.addWidget(grid2)

    def clear21(self, index=2):
        for i in self.intline[index]:
            i.clear()

    def msg2(self):
        QMessageBox.about(self, "Help", "This function is a simulator of dice throw .\n"
                                        "The simulator will generate a graphic of probability distribution.\n"
                                        "Can generate multiple images simultaneously.")

    def dice_simulation(self):
        try:
            number, face, time = [int(i.text()) for i in self.intline[2]]
            list_dice = np.arange(1, face + 1)

            # result = [sum(random.choices(list_dice, k=10)) for i in range(time)]
            result = [sum(random.choices(list_dice, k=number)) for i in range(time)]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            'the formula of bins makes the histogram looks more comfortable'
            # histo = ax.hist(result, color='RoyalBlue', bins=np.arange((face-1) * number + 4)-0.5, label='occurrence')
            histo = ax.hist(result, color='RoyalBlue', bins=np.arange(face * number + 2) - 0.5, label='occurrence')
            ax.set_xlim([number - 1, number * face + 1])
            ax.set_xlabel('Dice Points')
            ax.set_ylabel('total occurence')
            ax.legend(loc=1)
            plt.title('{} times Toss {} dice each has {} faces'.format(time, number, face))
            plt.grid()
            plt.show()

        except:
            self.Error()

    '-----------------Distribution-------------------------'

    def stackUI3(self):
        layout = QVBoxLayout()
        glayout = QGridLayout()
        gbox = QWidget()
        gbox.setLayout(glayout)

        self.combobox[3] = QComboBox()
        font31 = QFont()
        font31.setPointSize(16)
        self.combobox[3].setFont(font31)

        list_00 = ["Please select...", "Beta_Distribution", "Binomial_Distribution", "Cauchy_Distribution",
                   "Chi2_Distribution", "Expon_Distribution", "F_Distribution", "Gamma_Distribution",
                   "Geometric_Distribution", "Laplace_Distribution", "Logistic_Distribution", "Lomax_Distribution",
                   "Lognorm_Distribution", "Negative_Binomial_Distribution", "Normal_Distribution",
                   "Poisson_Distribution", "Rayleigh_Distribution", "T_Distribution", "Weibull_Distribution",
                   "Zipf_Distribution"]

        for i in list_00:
            self.combobox[3].addItem(i)

        self.combobox[3].currentIndexChanged.connect(self.Select_onChange31)

        self.le[3] = [QLineEdit() for i in range(2)]

        self.label[3] = [QLabel()] + [QLabel(text) for text in [" ", "1. parameter:", "2. parameter:"]]
        self.label[3][1].setFont(QFont('Sanserif', 15))
        self.label[3][1].setStyleSheet("font:bold")

        self.pb[3] = []
        for i, j in zip(['Execute', 'Clear', 'Help'], [self.Select_onChange32, self.clear31, self.msg3]):
            self.pb[3].append(QPushButton(i))
            self.pb[3][-1].clicked.connect(j)

        for i in [self.combobox[3], *self.label[3][:2], gbox]:
            layout.addWidget(i)

        list_00 = [*self.label[3][2:], *self.le[3], *self.pb[3]]
        list_01 = [1, 2, 1, 2, 2, 3, 1]
        list_02 = [0, 0, 1, 1, 2, 2, 2]
        for i in zip(list_00, list_01, list_02):
            glayout.addWidget(*i)

        self.stack[3].setLayout(layout)

    def clear31(self, index=3):
        for i in self.le[index]:
            i.clear()

    def msg3(self):
        QMessageBox.about(self, "Help", "This function is a simulator of general distributions.\n"
                                        "The simulator will generate a graphic of probability distribution.\n"
                                        "Inputboxes support multiple sets of parameters like '0.5,0.7,1.0'")

    def Select_onChange31(self):

        switcher = {
            "Binomial_Distribution": ["binomial.svg", [200, 60], 'n={}\n' 'p={}'],
            "Normal_Distribution": ["normal.svg", [200, 60], 'μ={}\n' 'σ²={}'],
            "Poisson_Distribution": ["poisson.svg", [250, 70], 'λ={}'],
            "Rayleigh_Distribution": ["rayleigh.svg", [250, 60], 'σ={}'],
            "Beta_Distribution": ["Beta.svg", [200, 60], 'α={}\n' 'β={}'],
            "F_Distribution": ["f.svg", [450, 350], 'd1={}\n' 'd2={}'],
            "Gamma_Distribution": ["gamma2.svg", [300, 50], 'k={} θ={}'],
            "Geometric_Distribution": ["geometric.svg", [290, 60], 'p={}'],
            "Lognorm_Distribution": ["lognorm.svg", [250, 60], 'μ={}\n' 'σ={}'],
            "Chi2_Distribution": ["chi2.svg", [300, 140], 'df={}'],
            "Cauchy_Distribution": ["cauchy.svg", [350, 80], 'x0={}\n' 'γ={}'],
            "Laplace_Distribution": ["laplace.svg", [200, 60], 'μ={}\n' 'λ={}'],
            "T_Distribution": ["t.svg", [300, 90], 'v={}'],
            "Expon_Distribution": ["exponential.svg", [200, 60], 'λ={}'],
            "Weibull_Distribution": ["weibull.svg", [350, 80], 'λ={}\n' 'a={}'],
            "Negative_Binomial_Distribution": ["negativ.svg", [300, 60], 'n={}\n' 'p={}'],
            "Lomax_Distribution": ["lomax.svg", [250, 60], 'λ={}\n' 'α={}'],
            "Logistic_Distribution": ["logistic.svg", [300, 170], 'μ={}\n' 's={}']
        }

        if self.combobox[3].currentText() == 'Please select...':
            self.label[3][0].setText(' ')
            self.label[3][1].setText(' ')
        elif self.combobox[3].currentText() == 'Zipf_Distribution':
            self.label[3][0].setText('No pic')
            self.label[3][0].setScaledContents(True)
            self.label[3][0].setMaximumSize(200, 60)
            self.label[3][1].setText('a={}')
        else:
            i = switcher.get(self.combobox[3].currentText(), None)
            self.label[3][0].setPixmap(QPixmap(i[0]))
            self.label[3][0].setScaledContents(True)
            self.label[3][0].setMaximumSize(*i[1])
            self.label[3][1].setText(i[2])

    def _decorator(self, func):
        def inner(*args, **kwargs):
            try:
                n_args = len(inspect.getfullargspec(func).args)
                sp = [self.le[3][i].text().split(',') for i in range(n_args)]

                if len(sp) > 1 and not all(len(sp[0]) == len(x) for x in sp[1:]):
                    QMessageBox.about(self, "Warning", "The length of the two rows is not the same.")
                else:
                    func(*sp, *args, **kwargs)
            except:
                self.Error()

        return inner

    def Select_onChange32(self):

        switcher = {
            "Binomial_Distribution": self._decorator(distribution_displays.Binomial_Distribution),
            "Normal_Distribution": self._decorator(distribution_displays.Normal_Distribution),
            "Poisson_Distribution": self._decorator(distribution_displays.Poisson_Distribution),
            "Rayleigh_Distribution": self._decorator(distribution_displays.Rayleigh_Distribution),
            "Beta_Distribution": self._decorator(distribution_displays.Beta_Distribution),
            "F_Distribution": self._decorator(distribution_displays.F_Distribution),
            "Gamma_Distribution": self._decorator(distribution_displays.Gamma_Distribution),
            "Geometric_Distribution": self._decorator(distribution_displays.Geometric_Distribution),
            "Lognorm_Distribution": self._decorator(distribution_displays.Lognorm_Distribution),
            # "Uniform_Distribution": self._decorator(distribution_displays.Uniform_Distribution),
            "Chi2_Distribution": self._decorator(distribution_displays.Chi2_Distribution),
            "Cauchy_Distribution": self._decorator(distribution_displays.Cauchy_Distribution),
            "Laplace_Distribution": self._decorator(distribution_displays.Laplace_Distribution),
            "T_Distribution": self._decorator(distribution_displays.T_Distribution),
            "Expon_Distribution": self._decorator(distribution_displays.Expon_Distribution),
            "Weibull_Distribution": self._decorator(distribution_displays.Weibull_Distribution),
            "Zipf_Distribution": self._decorator(distribution_displays.Zipf_Distribution),
            "Negative_Binomial_Distribution": self._decorator(distribution_displays.Negative_Binomial_Distribution),
            "Lomax_Distribution": self._decorator(distribution_displays.Lomax_Distribution),
            "Logistic_Distribution": self._decorator(distribution_displays.Logistic_Distribution)
        }

        switcher.get(self.combobox[3].currentText(), lambda: None)()

    def Error(self):
        QMessageBox.about(self, 'Warning', 'Error happened!\n'
                                           'please check parameters!')

    '----------------------Central Limit Theorem----------------------------'

    def stackUI4(self):
        layout = QVBoxLayout()
        layout1 = QGridLayout()
        hbox1 = QWidget()
        hbox1.setLayout(layout1)

        self.pb[4] = [QPushButton(text) for text in ['Execute', 'Help', 'Clear']]

        self.le[4] = [QLineEdit() for i in range(2)]

        self.label[4] = [QLabel(text) for text in ['Number of dice:', 'Number of times:', 'Central limit theorem']]

        self.label[4][2].setStyleSheet(
            "background-color:'RoyalBlue';border:outset;color:'yellow';font:bold;font-size:20px"
        )
        self.label[4][2].setAlignment(Qt.AlignCenter)
        self.label[4][2].setGeometry(20, 30, 600, 300)

        list_00 = [*self.label[4][:2], *self.le[4], *self.pb[4]]
        list_01 = [2, 3, 2, 3, 4, 1, 5]
        list_02 = [0, 0, 1, 1, 2, 2, 2]
        for i in zip(list_00, list_01, list_02):
            layout1.addWidget(*i)

        layout.addWidget(self.label[4][2])
        layout.addWidget(hbox1)

        for i, j in zip(self.pb[4], [self.CentralLimintTheorem, self.msg4, self.clear41]):
            i.clicked.connect(j)

        self.stack[4].setLayout(layout)

    def clear41(self, index=4):
        for i in self.le[index]:
            i.clear()

    def msg4(self):
        QMessageBox.about(self, 'Help', 'the more dice and the more loop times,makes the histgram more like approach'
                                        ' the Normal Distribution')

    def CentralLimintTheorem(self):
        try:
            number = int(self.le[4][0].text())
            times = int(self.le[4][1].text())
            mu = 3.5 * number
            sigma = (35 / 12) * number
            x = np.arange(1 * number - 1, 6 * number + 1, 0.1)
            y = norm.pdf(x, loc=mu, scale=math.sqrt(sigma))
            samples_sum = []
            for i in range(times):
                sample = np.random.randint(1, 7, size=number)
                sum = np.sum(sample)
                samples_sum.append(sum)

            fig = plt.figure(tight_layout=True)
            ax = fig.add_subplot(111)
            ax.hist(samples_sum, bins=np.arange(4 + 5 * number) - 0.5, density=True, color='RoyalBlue', alpha=0.9,
                    label='relative occurrence')
            ax.set_xlabel('Dice Points')
            ax.set_ylabel('relative occurence')
            ax.set_xlim([number - 2, number * 6 + 2])
            ax.legend(loc=2)
            ax2 = ax.twinx()
            ax2.plot(x, y, label='Theoretical Value', color='r')
            ax2.legend(loc=1)
            ax2.set_ylabel('Probability')
            ax2.set_ylim(ymin=0)
            plt.title('{} dice {} throws'.format(number, times))
            ax2.set_ylim(ax.get_ylim())
            plt.grid()
            plt.show()
        except:
            self.Error()

    '---------------------------Correlation example-------------------------------------------------'

    def stackUI5(self):
        layout = QVBoxLayout()
        layout_i = [QGridLayout() for i in range(2)]
        hlayout = QHBoxLayout()

        hbox = QWidget()
        hbox.setLayout(hlayout)

        self.label[5] = [QLabel(text) for text in ['Title precision:', 'x.xx']]
        self.sp[5] = QSpinBox()
        self.sp[5].setValue(2)
        self.sp[5].setMinimum(0)
        self.sp[5].setSingleStep(1)
        self.sp[5].valueChanged.connect(functools.partial(lambda x: x.label[5][1].setText("x." + x.sp[5].value() * "x"),
                                                          self))

        for i in [*self.label[5], self.sp[5]]:
            hlayout.addWidget(i)

        group = [QGroupBox(text, self) for text in ['Example', 'Input Data']]
        for i, j in zip(group, layout_i):
            i.setLayout(j)

        for i in [hbox, *group]:
            layout.addWidget(i)

        self.pb[5] = [QPushButton(text) for text in
                      ['Weak correlation', 'Strong correlation', 'Uncorrelated', 'AddFile']]

        list_00 = [0, 0, 0, 1]
        list_01 = [1, 1, 1, 1]
        list_02 = [0, 1, 2, 0]
        list_03 = [self.Weak_correlation, self.Strong_correlation, self.Uncorrelated, self.Openfile_coe]
        for i, j, *k, l in zip(list_00, self.pb[5], list_01, list_02, list_03):
            layout_i[i].addWidget(j, *k)
            j.clicked.connect(l)

        self.stack[5].setLayout(layout)

    def Valuechange51(self):  # In my opinion 3 levels are enough

        self.label[5][1].setText("x." + self.sp[5].value() * "x")

    def base_correlation(self, func):
        try:
            x = np.arange(1, 101)
            y = func(x)

            coefxy = np.corrcoef(x, y)
            pxy = coefxy[0, 1]

            res = linregress(x, y)
            y1 = res.intercept + np.multiply(res.slope, x)

            plt.plot(x, y, marker='o', linestyle='None')
            plt.plot(x, y1, c='r', label='fitted line')

            plt.xlabel('x')
            plt.ylabel('y')

            plt.title(f"r = {pxy:.{self.sp[5].value()}f} Fitted line: y = {res.intercept:.{self.sp[5].value()}f}"
                      f" + {res.slope:.{self.sp[5].value()}f} * x")

            plt.grid()
            plt.legend()
            plt.show()
        except:
            self.Error()

    def Weak_correlation(self):
        self.base_correlation(lambda x: np.random.randn(100) * 350 + np.random.randint(-10, 10, 1) * x)

    def Strong_correlation(self):
        self.base_correlation(lambda x: np.random.randn(100) * 50 + np.random.randint(-10, 10, 1) * x)

    def Uncorrelated(self):
        x0 = np.linspace(-1, 1, 200)  # Draw a circle in polar coordinates
        y0 = np.sqrt(1 - x0 ** 2)
        list1 = []
        list2 = []

        for i in range(200):
            a = (-1) ** random.randint(0, 1)
            b = random.random() * 0.1
            list1.append(a)
            list2.append(b)
        y1 = np.multiply(y0, list1)
        list3 = np.array(list2)
        y2 = y1 + list3

        coefxy = np.corrcoef(x0, y2)
        pxy = coefxy[0, 1]

        plt.plot(x0, y2, marker='o', linestyle='None')

        plt.xlabel('x')
        plt.ylabel('y')

        plt.title(f"r = {pxy:.{self.sp[5].value()}f} No fitted line")

        plt.grid()
        plt.show()

    def Openfile_coe(self):
        try:
            self.filename = QFileDialog.getOpenFileName(self, 'ChooseFile')[0]
            self.Openfile_coe2()
        except Exception as r:
            self.Error()

    def loader(self):
        with open(self.filename, 'r') as f:
            f_csv = csv.reader(f)
            csv_list = []
            for row in f_csv:
                csv_list.append(row)

            x = csv_list[0]
            x = list(map(int, x))

            y = csv_list[1]
            y = list(map(int, y))

        return x, y

    def Openfile_coe2(self):
        x, y = self.loader()

        cov = np.cov(*self.loader())
        result1 = cov[0, 0] * cov[1, 1]
        result2 = cov[0, 1] * cov[1, 0]

        if result1 != 0 and result2 != 0:  # Check whether the data is independent

            coefxy = np.corrcoef(x, y)
            pxy = coefxy[0, 1]

            res = linregress(x, y)
            y1 = res.intercept + np.multiply(res.slope, x)

            plt.plot(x, y, marker='o', linestyle='None')

            plt.xlabel('x')
            plt.ylabel('y')

            plt.plot(x, y1, c='r', label='fitted line')

            plt.title(f"r = {pxy:.{self.sp[5].value()}f} Fitted line: y = {res.intercept:.{self.sp[5].value()}f}"
                      f" + {res.slope:.{self.sp[5].value()}f} * x")

            plt.legend()
            plt.grid()
            plt.show()

        else:

            plt.plot(x, y, marker='o', linestyle='None')

            plt.xlabel('x')
            plt.ylabel('y')

            plt.title(f"r = {0:.{self.sp[5].value()}f} No fitted line")

            plt.grid()
            plt.show()

    '===================CCF========================='

    def stackUI6(self):
        layout = QVBoxLayout()
        layout_i = [QGridLayout() for i in range(2)]

        self.group[6] = [QGroupBox(text, self) for text in ['Example', 'Input Data']]

        for i, j in zip(self.group[6], layout_i):
            i.setLayout(j)
            layout.addWidget(i)

        self.pb[6] = [QPushButton(i) for i in ["CCF", "ACF", "ACF_Rxx", "AddFile1_CCF", "AddFile2_ACF",
                                               "AddFile3_Linear_Regression"]]

        list_00 = [1, 1, 1, 1, 1, 1]
        list_01 = [0, 1, 2, 0, 1, 2]
        list_02 = [self.ccf, self.acf, self.acf_Rxx, self.add_file_01, self.add_file_02, self.add_file_03]

        for i, *j, k in zip(self.pb[6][:3], list_00[:3], list_01[:3], list_02[:3]):
            layout_i[0].addWidget(i, *j)
            layout_i[0].addWidget(i)
            i.clicked.connect(k)

        for i, *j, k in zip(self.pb[6][3:], list_00[3:], list_01[3:], list_02[3:]):
            layout_i[1].addWidget(i, *j)
            layout_i[1].addWidget(i)
            i.clicked.connect(k)

        self.stack[6].setLayout(layout)

    def acf_Rxx(self):
        # x = np.array([50,47,60,88,20,19,12,57,49,33,42,10,99,22,58,67,90,56,33,74,23,62,90,29,74,10,29,74,57,15])
        x1 = np.arange(1, 50, 0.01)

        x2 = np.cos(x1)

        x3 = x2[0]
        n = len(x3)
        o = np.arange(1 - n, n)
        var = np.var(x3, ddof=1)

        mx = np.mean(x3)

        autocorrelation = np.correlate(x3, x3, 'full')
        plt.plot(o, autocorrelation, marker='o', linestyle='None')
        plt.plot([1 - n, n], [var + mx * 2, var + mx * 2], c='g', linestyle='--')
        plt.plot([1 - n, n], [mx * 2, mx * 2], c='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('R')
        plt.title('Autocorrelation function')
        plt.grid()
        plt.show()

    def ccf(self):
        x = np.array([50, 47, 60, 88, 20, 19, 12, 57, 49, 33, 42, 10, 99, 22, 58, 67, 90, 56, 33, 74, 23, 62, 90, 29,
                      74, 10, 29, 74, 57, 15])
        y = np.array([20, 70, 66, 40, 53, 22, 14, 68, 43, 89, 54, 55, 3, 78, 56, 4, 9, 41, 14, 24, 68, 64, 87, 45, 33,
                      67, 55, 22, 86, 45])

        n = len(x)
        o = np.arange(1 - n, n)
        crosscorrelation = np.correlate(x, y, 'full')

        plt.plot(o, crosscorrelation, marker='o', linestyle='None')

        plt.xlabel('Time')
        plt.ylabel('R')
        plt.title('Cross correlation function')

        plt.grid()
        plt.show()

    def acf(self):
        x = np.array([50, 47, 60, 88, 20, 19])
        n = len(x)
        o = np.arange(1 - n, n)

        autocorrelation = np.correlate(x, x, 'full')

        plt.plot(o, autocorrelation, marker='o', linestyle='None')
        plt.xlabel('Time')
        plt.ylabel('R')
        plt.title('Autocorrelation function')
        plt.grid()
        plt.show()
        '------------------------------------------------------'

    '-------------------------------------------------------'

    def stackUI7(self):
        pass

    def get_data(self, func, title):
        try:
            x, y = self.loader()
            n = len(x)
            o = np.arange(1 - n, n)
            correlation = func(x, y)

            plt.plot(o, correlation, marker='o')
            plt.xlabel('Time')
            plt.ylabel('R')
            plt.title(title)
            plt.show()
            plt.grid()
        except:
            self.Error()

    def add_file_01(self):
        try:
            self.filename = QFileDialog.getOpenFileName(self, 'ChooseFile')[0]
            self.self.get_data(lambda x, y: np.correlate(x, y, 'full'), 'Cross correlation function')
        except:
            self.Error()

    def add_file_02(self):
        try:
            self.filename = QFileDialog.getOpenFileName(self, 'ChooseFile')[0]
            self.self.get_data(lambda x, y: np.correlate(x, x, 'full'), 'Autocorrelation function')
        except:
            self.Error()

    def add_file_03(self):
        try:
            self.filename = QFileDialog.getOpenFileName(self, 'ChooseFile')[0]
            self.get_data3()
        except:
            self.Error()

    def get_data3(self):
        try:
            x, y = self.loader()

            res = linregress(x, y)
            y1 = res.intercept + np.multiply(res.slope, x)
            plt.plot(x, y, 'o', label='original data')
            plt.plot(x, y1, 'r', label='fitted line')
            plt.legend()
            plt.show()
        except:
            self.Error()
        '------------------AR-----------------------------'

    '-------------------------AR Model----------------------------------'

    def stackUI8(self):  # This part is the application of ARMA Model in the economic field, which is inconsistent
        # with the teaching purpose of this lecture
        # layout = QVBoxLayout()
        #
        # self.pb81 = QPushButton('Dataset')
        # self.pb82 = QPushButton('coefficient')
        # self.pb84 = QPushButton('Prediction')
        # self.pb81.clicked.connect(self.Dataset)
        # self.pb82.clicked.connect(self.coefficient)
        # self.pb84.clicked.connect(self.Prediction)
        #
        # self.df = pd.read_csv(
        # 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv',
        #                  index_col=0, parse_dates=True)
        #
        # layout.addWidget(self.pb81)
        # layout.addWidget(self.pb82)
        # layout.addWidget(self.pb84)
        # self.stack8.setLayout(layout)
        pass

    def Dataset(self):
        pass
        # self.df.plot()
        # plt.show()

    def coefficient(self):
        pass
        # a = self.df.Temp
        # b = self.df.Temp.shift(1)
        # coefficient = pearsonr(a[1:], b[1:])
        # lag_plot(self.df)
        # plt.title('Correlation coefficient:r = {:.2f}'.format(coefficient[0]))
        # plt.show()

    def ACF_PACF(self):
        pass
        # fig, axes = plt.subplots(2, 1)
        # plot_acf(self.df['Temp'], ax=axes[0])
        # plot_pacf(self.df['Temp'], ax=axes[1])
        #
        # plt.tight_layout()
        # plt.show()

    def Prediction(self):
        pass
        # x = self.df.values
        # train, test = x[:-7], x[-7:]
        # model_fit = AR(train).fit()
        # params = model_fit.params
        # p = model_fit.k_ar
        # # p = 1
        # history = train[-p:]
        # history = np.hstack(history).tolist()
        # test = np.hstack(test).tolist()
        #
        # predictions = []
        # for t in range(len(test)):
        #     lag = history[-p:]
        #     yhat = params[0]
        #     for i in range(p):
        #         yhat += params[i + 1] * lag[p - 1 - i]
        #     predictions.append(yhat)
        #     obs = test[t]
        #     history.append(obs)
        # print(np.mean((np.array(test) - np.array(predictions)) ** 2))  # 得到mean_squared_error, MSE
        # plt.plot(test, color='b',label='Reality')
        # plt.plot(predictions, color='r',label='Prediction')
        # plt.legend()
        # plt.show()

    def stackUI9(self):
        layout = QVBoxLayout()
        layout_i = [QGridLayout() for i in range(2)]

        self.group[9] = [QGroupBox(text, self) for text in ['Sample generate', 'Graphic']]
        for i, j in zip(self.group[9], layout_i):
            i.setLayout(j)
            layout.addWidget(i)

        self.label[9] = [QLabel(text) for text in ["b1:", "σw^2"]]
        self.le[9] = [QLineEdit() for i in range(2)]

        self.pb[9] = []
        for i, j in zip(["Execute", "Rxx", "rxx"], [self.sample_generate91, self.Rxx91, self.rxx91]):
            self.pb[9].append(QPushButton(i))
            self.pb[9][-1].clicked.connect(j)

        list_00 = [*self.label[9], *self.le[9], *self.pb[9]]
        list_01 = [2, 3, 2, 3, 4, 1, 2]
        list_02 = [0, 0, 1, 1, 2, 1, 1]
        for i in zip(list_00[:5], list_01[:5], list_02[:5]):
            layout_i[0].addWidget(*i)

        for i in zip(list_00[5:], list_01[5:], list_02[5:]):
            layout_i[1].addWidget(*i)

        self.stack[9].setLayout(layout)

    def sample_generate91(self):
        try:
            b1 = float(self.le[9][0].text())
            ar_coefs = [1]
            ar_coefs.append(b1)
            ma_coefs = [1, 0]
            sigma_s = float(self.le[9][1].text())
            max_lag = 15
            y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100)
            x = np.arange(100)
            plt.plot(x, y)
            plt.title('Time Series b1 = {} σw^2 = {}'.format(b1, sigma_s))
            plt.show()
        except:
            self.Error()

    def Rxx91(self):
        try:
            b1 = float(self.le[9][0].text())
            sigma_w_2 = float(self.le[9][1].text())
            sigma_x_2 = sigma_w_2 / (1 - b1 ** 2)

            Rxx = []
            for m in range(-10, 11, 1):
                y = b1 ** abs(m) * sigma_x_2
                Rxx.append(y)

            x = np.arange(-10, 11)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, Rxx, 'bo')
            ax.set_ylim([0, None])
            plt.title('Rxx b1 = {} σw^2 = {}'.format(b1, sigma_w_2))
            plt.grid()
            plt.show()
        except:
            self.Error()

    # coefficient
    def rxx91(self):
        try:
            b1 = float(self.le[9][0].text())
            sigma_w_2 = float(self.le[9][1].text())
            sigma_x_2 = sigma_w_2 / (1 - b1 ** 2)

            Rxx = []
            for m in range(0, 11, 1):
                y = b1 ** abs(m) * sigma_x_2
                Rxx.append(y)

            rxx = []
            for i in Rxx:
                i /= sigma_x_2
                rxx.append(i)

            x = np.arange(0, 11)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, rxx, 'bo')
            ax.set_ylim([0, None])
            plt.title('rxx b1 = {} σw^2 = {}'.format(b1, sigma_w_2))
            plt.grid()
            plt.show()
        except:
            self.Error()  # coeff

    '-------------------------Estimation Model----------------------------------'

    def stackUI10(self):
        layout = QVBoxLayout()
        glayout = QGridLayout()
        gbox = QWidget()
        gbox.setLayout(glayout)

        self.combobox[9] = QComboBox()
        self.combobox[10] = QComboBox()
        font31 = QFont()
        font31.setPointSize(16)
        self.combobox[9].setFont(font31)
        self.combobox[10].setFont(font31)

        list_00 = ["Please select Distribution", "Beta_Distribution",  "Cauchy_Distribution",
                   "F_Distribution", "Gamma_Distribution", "Laplace_Distribution", "Logistic_Distribution",
                   "Lomax_Distribution", "Lognorm_Distribution", "Normal_Distribution", "Rayleigh_Distribution"]
        list_001 = ["Please fit Distribution", "Beta_Distribution", "Binomial_Distribution", "Cauchy_Distribution",
                    "F_Distribution", "Gamma_Distribution", "Laplace_Distribution", "Logistic_Distribution",
                    "Lomax_Distribution", "Lognorm_Distribution", "Normal_Distribution", "Rayleigh_Distribution"]

        for i in list_00:
            self.combobox[10].addItem(i)
        for i in list_001:
            self.combobox[9].addItem(i)

        self.combobox[10].currentIndexChanged.connect(self.Select_onChange101)

        self.le[10] = [QLineEdit() for i in range(2)]

        self.label[10] = [QLabel()] + [QLabel(text) for text in [" ", "1. parameter:", "2. parameter:"]]
        self.label[10][1].setFont(QFont('Sanserif', 15))
        self.label[10][1].setStyleSheet("font:bold")

        self.pb[10] = []
        for i, j in zip(['Execute', 'Clear', 'Help', 'Fit'],
                        [self.Simulation, self.clear101, self.msg10, self.fit]):
            self.pb[10].append(QPushButton(i))
            self.pb[10][-1].clicked.connect(j)

        for i in [self.combobox[10], self.combobox[9], *self.label[10][:2], gbox]:
            layout.addWidget(i)

        list_00 = [*self.label[10][2:], *self.le[10], *self.pb[10]]
        list_01 = [1, 2, 1, 2, 2, 4, 1, 3]
        list_02 = [0, 0, 1, 1, 2, 2, 2, 2]
        for i in zip(list_00, list_01, list_02):
            glayout.addWidget(*i)

        self.stack[10].setLayout(layout)

    def clear101(self, index=10):
        for i in self.le[index]:
            i.clear()

    def Select_onChange101(self):

        switcher = {
            "Binomial_Distribution": ["binomial.svg", [200, 60], 'n={}\n' 'p={}'],
            "Normal_Distribution": ["normal.svg", [200, 60], 'μ={}\n' 'σ²={}'],
            "Poisson_Distribution": ["poisson.svg", [250, 70], 'λ={}'],
            "Rayleigh_Distribution": ["rayleigh.svg", [250, 60], 'σ={}'],
            "Beta_Distribution": ["Beta.svg", [200, 60], 'α={}\n' 'β={}'],
            "F_Distribution": ["f.svg", [450, 350], 'd1={}\n' 'd2={}'],
            "Gamma_Distribution": ["gamma2.svg", [300, 50], 'k={} θ={}'],
            "Geometric_Distribution": ["geometric.svg", [290, 60], 'p={}'],
            "Lognorm_Distribution": ["lognorm.svg", [250, 60], 'μ={}\n' 'σ={}'],
            "Chi2_Distribution": ["chi2.svg", [300, 140], 'df={}'],
            "Cauchy_Distribution": ["cauchy.svg", [350, 80], 'x0={}\n' 'γ={}'],
            "Laplace_Distribution": ["laplace.svg", [200, 60], 'μ={}\n' 'λ={}'],
            "T_Distribution": ["t.svg", [300, 90], 'v={}'],
            "Expon_Distribution": ["exponential.svg", [200, 60], 'λ={}'],
            "Weibull_Distribution": ["weibull.svg", [350, 80], 'λ={}\n' 'a={}'],
            "Negative_Binomial_Distribution": ["negativ.svg", [300, 60], 'n={}\n' 'p={}'],
            "Lomax_Distribution": ["lomax.svg", [250, 60], 'λ={}\n' 'α={}'],
            "Logistic_Distribution": ["logistic.svg", [300, 170], 'μ={}\n' 's={}']
        }

        if self.combobox[10].currentText() == 'Please select Distribution':
            self.label[10][0].setText(' ')
            self.label[10][1].setText(' ')
        elif self.combobox[10].currentText() == 'Zipf_Distribution':
            self.label[10][0].setText('No pic')
            self.label[10][0].setScaledContents(True)
            self.label[10][0].setMaximumSize(200, 60)
            self.label[10][1].setText('a={}')
        else:
            i = switcher.get(self.combobox[10].currentText(), None)
            self.label[10][0].setPixmap(QPixmap(i[0]))
            self.label[10][0].setScaledContents(True)
            self.label[10][0].setMaximumSize(*i[1])
            self.label[10][1].setText(i[2])

    def msg10(self):
        QMessageBox.about(self, "Help", "The simulator generates a probability distribution\n"
                                        "which is then fitted to estimate its parameters.\n"
                                        
                                        "Inputboxes support a sets of parameters like '4'")

    def Simulation(self):
        try:
            a = 0
            b = 0
            a = float(self.le[10][0].text())
            b = float(self.le[10][1].text())

            switcher = {
                # "Binomial_Distribution": Simulation_fit.getParament(a, b).binomial_P,
                "Normal_Distribution": Simulation_fit.getParament(a, b).normal_P,
                "Rayleigh_Distribution": Simulation_fit.getParament(a, b).rayleigh_P,
                "Beta_Distribution": Simulation_fit.getParament(a, b).beta_P,
                "F_Distribution": Simulation_fit.getParament(a, b).f_P,
                "Gamma_Distribution": Simulation_fit.getParament(a, b).gamma_P,
                "Lognorm_Distribution": Simulation_fit.getParament(a, b).lognorm_P,
                "Cauchy_Distribution": Simulation_fit.getParament(a, b).cauchy_P,
                "Laplace_Distribution": Simulation_fit.getParament(a, b).laplace_P,
                "Lomax_Distribution": Simulation_fit.getParament(a, b).lomax_P,
                "Logistic_Distribution": Simulation_fit.getParament(a, b).logistic_P
            }

            X = switcher.get(self.combobox[10].currentText(), lambda: None)()
            fig = plt.figure()
            Simulation_fit.fit_Funktion(X).Sim()
        except:
            self.Error()

    def fit(self):
        try:
            a = 0
            b = 0
            a = float(self.le[10][0].text())
            # print(a)
            b = float(self.le[10][1].text())
            # print(b)
            switcher = {
                # "Binomial_Distribution": Simulation_fit.getParament(a, b).binomial_P,
                "Normal_Distribution": Simulation_fit.getParament(a, b).normal_P,
                "Rayleigh_Distribution": Simulation_fit.getParament(a, b).rayleigh_P,
                "Beta_Distribution": Simulation_fit.getParament(a, b).beta_P,
                "F_Distribution": Simulation_fit.getParament(a, b).f_P,
                "Gamma_Distribution": Simulation_fit.getParament(a, b).gamma_P,
                "Lognorm_Distribution": Simulation_fit.getParament(a, b).lognorm_P,
                "Cauchy_Distribution": Simulation_fit.getParament(a, b).cauchy_P,
                "Laplace_Distribution": Simulation_fit.getParament(a, b).laplace_P,
                "Lomax_Distribution": Simulation_fit.getParament(a, b).lomax_P,
                "Logistic_Distribution": Simulation_fit.getParament(a, b).logistic_P
            }
            X = switcher.get(self.combobox[10].currentText(), lambda: None)()
            fig = plt.figure()
            switcher_new = {
                "Binomial_Distribution": Simulation_fit.getParament(a, b).binomial_P,
                "Normal_Distribution": Simulation_fit.fit_Funktion(X).normal_Fit,
                "Rayleigh_Distribution": Simulation_fit.fit_Funktion(X).rayleigh_Fit,
                "Beta_Distribution": Simulation_fit.fit_Funktion(X).beat_Fit,
                "F_Distribution": Simulation_fit.fit_Funktion(X).f_Fit,
                "Gamma_Distribution": Simulation_fit.fit_Funktion(X).gamma_Fit,
                "Lognorm_Distribution": Simulation_fit.fit_Funktion(X).lognorm_Fit,
                "Cauchy_Distribution": Simulation_fit.fit_Funktion(X).cauchy_Fit,
                "Laplace_Distribution": Simulation_fit.fit_Funktion(X).laplace_Fit,
                "Lomax_Distribution": Simulation_fit.fit_Funktion(X).lomax_Fit,
                "Logistic_Distribution": Simulation_fit.fit_Funktion(X).logistic_Fit
            }

            switcher_new.get(self.combobox[9].currentText(), lambda: None)()
        except:
            self.Error()
    '-------------------------confidence ellipse----------------------------------'
    def stackUI11(self):
        vlayout = QVBoxLayout(self.stack[11])
        gridlayout = QGridLayout()
        grid = QWidget()
        grid.setLayout(gridlayout)
        vlayout.addWidget(grid)

        self.le[11] = [QLineEdit() for i in range(4)]

        list_00 = ["pb1_1", "pb1_2", "pb1_3", "pb1_4", "help"]
        list_01 = ["positive correlation", "negative correlation", "Weak correlation", "Clear",  "Help"]
        list_02 = [self.positive_correlation, self.negative_correlation, self.Weak_correlation,
                   self.clear111, self.msg11]
        for i, j, k in zip(list_00, list_01, list_02):
            self.var_dict[i] = QPushButton(j)
            self.var_dict[i].clicked.connect(k)

        for i, j in zip(["label11", "label12", "label13", "label14"], ["quantity:", "mu:", "scale", "std"]):
            self.var_dict[i] = QLabel()
            self.var_dict[i].setText(j)

        list_03 = [self.var_dict["help"], self.var_dict["label11"], self.var_dict["label12"],
                   self.var_dict["label13"], self.var_dict["label14"], *self.le[11],
                   self.var_dict["pb1_1"], self.var_dict["pb1_2"], self.var_dict["pb1_3"], self.var_dict["pb1_4"]]
        list_04 = [1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5]
        list_05 = [2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        for i in zip(list_03, list_04, list_05):
            gridlayout.addWidget(*i)

    def positive_correlation(self):
        fig, ax_nstd = plt.subplots()

        dependency_nstd = [[0.85, 0.35],
                           [0.15, -0.65]]

        sp = [self.le[11][i].text().split(',') for i in range(4)]

        try:
            n = int(self.le[11][0].text())
            mu = list(map(int, sp[1]))
            scale = list(map(int, sp[2]))
            std = list(map(int, sp[3]))

            ax_nstd.axvline(c='grey', lw=1)
            x, y = get_correlated_dataset(n, dependency_nstd, mu, scale)
            ax_nstd.scatter(x, y, s=0.5)

            for i in range(len(std)):
                colors = ['r', 'g', 'b']
                confidence_ellipse(x, y, ax_nstd, n_std=std[i],
                                   label=r'$%s\sigma$' % std[i], edgecolor=colors[i])

            ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
            ax_nstd.set_title('Positive Correlation:Different standard deviations')
            ax_nstd.legend()
            plt.show()

        except:

            self.Error()

    def negative_correlation(self):
        fig, ax_nstd = plt.subplots()

        dependency_nstd = [[0.9, -0.4],
                           [0.1, -0.6]]

        sp = [self.le[11][i].text().split(',') for i in range(4)]

        try:
            n = int(self.le[11][0].text())
            mu = list(map(int, sp[1]))
            scale = list(map(int, sp[2]))
            std = list(map(int, sp[3]))

            ax_nstd.axvline(c='grey', lw=1)
            x, y = get_correlated_dataset(n, dependency_nstd, mu, scale)
            ax_nstd.scatter(x, y, s=0.5)

            for i in range(len(std)):
                colors = ['r', 'g', 'b']
                confidence_ellipse(x, y, ax_nstd, n_std=std[i],
                                   label=r'$%s\sigma$' % std[i], edgecolor=colors[i])

            ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
            ax_nstd.set_title('Negative Correlation:Different standard deviations')
            ax_nstd.legend()
            plt.show()

        except:

            self.Error()

    def Weak_correlation(self):
        fig, ax_nstd = plt.subplots()

        dependency_nstd = [[1, 0],
                           [0, 1]]

        sp = [self.le[11][i].text().split(',') for i in range(4)]

        try:
            n = int(self.le[11][0].text())
            mu = list(map(int, sp[1]))
            scale = list(map(int, sp[2]))
            std = list(map(int, sp[3]))

            ax_nstd.axvline(c='grey', lw=1)
            x, y = get_correlated_dataset(n, dependency_nstd, mu, scale)
            ax_nstd.scatter(x, y, s=0.5)

            for i in range(len(std)):
                colors = ['r', 'g', 'b']
                confidence_ellipse(x, y, ax_nstd, n_std=std[i],
                                   label=r'$%s\sigma$' % std[i], edgecolor=colors[i])

            ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
            ax_nstd.set_title('Weak Correlation:Different standard deviations')
            ax_nstd.legend()
            plt.show()

        except:

            self.Error()

    def clear111(self, index=11):
        for i in self.le[index]:
            i.clear()

    def msg11(self):
        QMessageBox.about(self, "Help", "The simulator generates confidence ellipses\n"
                                        
                                        "quantity support a sets of parameters like '500'\n"
                                        "Other inputboxes support multiple sets of parameters like '8,5'")

    '-------------------------Waveforms----------------------------------'

    def stackUI12(self):
        layout = QVBoxLayout(self.stack[12])
        gridlayout = QGridLayout()
        grid = QWidget()
        grid.setLayout(gridlayout)
        layout.addWidget(grid)

        self.le[12] = [QLineEdit() for i in range(3)]

        self.combobox[12] = QComboBox()
        font31 = QFont()
        font31.setPointSize(12)
        self.combobox[12].setFont(font31)

        list_00 = ["Please select mode", "square125", "square25", "square50", "square75", "triangle", "noise"]
        for i in list_00:
            self.combobox[12].addItem(i)
        layout.addWidget(self.combobox[12])

        list_00 = ["pb1_1", "pb1_2", "pb1_3", "pb1_4", "help"]
        list_01 = ["Execute", "Save", "Read", "Clear",  "Help"]
        list_02 = [self.wave_generation, self.save_wave, reafWave,
                   self.clear111, self.msg11]
        for i, j, k in zip(list_00, list_01, list_02):
            self.var_dict[i] = QPushButton(j)
            self.var_dict[i].clicked.connect(k)

        for i, j in zip(["label11", "label12", "label13"], ["Sample-rate:", "Frequency:", "Time-lengh"]):
            self.var_dict[i] = QLabel()
            self.var_dict[i].setText(j)

        list_03 = [self.var_dict["help"], self.var_dict["label11"], self.var_dict["label12"],
                   self.var_dict["label13"],  *self.le[12],
                   self.var_dict["pb1_1"], self.var_dict["pb1_2"], self.var_dict["pb1_3"], self.var_dict["pb1_4"]]
        list_04 = [1, 3, 4, 5,  3, 4, 5, 2, 3, 4, 5]
        list_05 = [2, 0, 0, 0,  1, 1, 1, 2, 2, 2, 2]
        for i in zip(list_03, list_04, list_05):
            gridlayout.addWidget(*i)

    def wave_generation(self):
        try:
            fig = plt.figure()
            sample_rate = int(self.le[12][0].text())
            fa = int(self.le[12][1].text())
            t_length = float(self.le[12][2].text())
            mode = str(self.combobox[12].currentText())

            y, t = createWave(sample_rate=sample_rate, fa=fa, t_length=t_length, mode=mode)
            plt.plot(t, y)
            plt.title("%s" % mode)
            plt.show()
        except:
            self.Error()

    def save_wave(self):
        try:
            fig = plt.figure()
            sample_rate = int(self.le[12][0].text())
            fa = int(self.le[12][1].text())
            t_length = float(self.le[12][2].text())
            mode = str(self.combobox[12].currentText())

            y, t = createWave(sample_rate=sample_rate, fa=fa, t_length=t_length, mode=mode)
            saveWave(y=y, sample_rate=sample_rate, path=r'wave.wav')
        except:
            self.Error()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())
