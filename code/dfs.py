# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dfs.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 299)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("imgs/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setStyleSheet("background-color: rgb(219, 242, 244);")
        self.tableWidget = QtWidgets.QTableWidget(Form)
        self.tableWidget.setGeometry(QtCore.QRect(130, 90, 161, 192))
        self.tableWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.tableWidget.setHorizontalHeaderItem(0, item)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(100, 50, 231, 20))
        self.label.setObjectName("label")
        self.dfsrevenir = QtWidgets.QCommandLinkButton(Form)
        self.dfsrevenir.setGeometry(QtCore.QRect(10, 0, 51, 51))
        self.dfsrevenir.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("imgs/iconrevenir.webp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.dfsrevenir.setIcon(icon1)
        self.dfsrevenir.setIconSize(QtCore.QSize(40, 40))
        self.dfsrevenir.setObjectName("dfsrevenir")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "DFS"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "DFS(sommet)"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Resultat de l\'algorithme de DFS :</span></p><p><br/></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
