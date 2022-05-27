# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'acceuil.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.resize(500, 400)
        Form.setMinimumSize(QtCore.QSize(500, 400))
        Form.setMaximumSize(QtCore.QSize(500, 400))
        Form.setMouseTracking(True)
        Form.setTabletTracking(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("imgs/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setStyleSheet("background-color: rgb(219, 242, 244);\n"
"")
        Form.setWindowFilePath("")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(-60, 0, 561, 251))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("imgs/Graph-Theory.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.oriente = QtWidgets.QCheckBox(Form)
        self.oriente.setGeometry(QtCore.QRect(90, 260, 91, 18))
        self.oriente.setObjectName("oriente")
        self.nonoriente = QtWidgets.QCheckBox(Form)
        self.nonoriente.setGeometry(QtCore.QRect(270, 260, 121, 18))
        self.nonoriente.setObjectName("nonoriente")
        self.star = QtWidgets.QCommandLinkButton(Form)
        self.star.setEnabled(True)
        self.star.setGeometry(QtCore.QRect(430, 330, 61, 61))
        self.star.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.star.setAutoFillBackground(False)
        self.star.setStyleSheet("background-color: rgb(189, 223, 255);\n"
"background-color: rgb(137, 168, 255);")
        self.star.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("imgs/iconstar.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.star.setIcon(icon1)
        self.star.setIconSize(QtCore.QSize(40, 40))
        self.star.setCheckable(False)
        self.star.setChecked(False)
        self.star.setAutoExclusive(False)
        self.star.setAutoDefault(False)
        self.star.setObjectName("star")
        self.ordegraphe = QtWidgets.QSpinBox(Form)
        self.ordegraphe.setGeometry(QtCore.QRect(250, 310, 42, 22))
        self.ordegraphe.setObjectName("ordegraphe")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(120, 310, 91, 20))
        self.label_2.setObjectName("label_2")
        self.star.raise_()
        self.label.raise_()
        self.oriente.raise_()
        self.nonoriente.raise_()
        self.ordegraphe.raise_()
        self.label_2.raise_()

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "acceuil"))
        self.oriente.setText(_translate("Form", "Graphe orienté"))
        self.nonoriente.setText(_translate("Form", "Graphe non orienté"))
        self.label_2.setText(_translate("Form", "Ordre du graphe :"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
