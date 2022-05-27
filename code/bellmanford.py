# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'bellmanford.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BellmanFord(object):
    def setupUi(self, BellmanFord):
        BellmanFord.setObjectName("BellmanFord")
        BellmanFord.resize(393, 370)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("imgs/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        BellmanFord.setWindowIcon(icon)
        BellmanFord.setStyleSheet("background-color: rgb(219, 242, 244);")
        self.label = QtWidgets.QLabel(BellmanFord)
        self.label.setGeometry(QtCore.QRect(90, 30, 61, 16))
        self.label.setObjectName("label")
        self.sommet = QtWidgets.QLineEdit(BellmanFord)
        self.sommet.setGeometry(QtCore.QRect(170, 30, 41, 21))
        self.sommet.setObjectName("sommet")
        self.bellmanrevenir = QtWidgets.QCommandLinkButton(BellmanFord)
        self.bellmanrevenir.setGeometry(QtCore.QRect(0, 0, 41, 41))
        self.bellmanrevenir.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("imgs/iconrevenir.webp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bellmanrevenir.setIcon(icon1)
        self.bellmanrevenir.setIconSize(QtCore.QSize(30, 30))
        self.bellmanrevenir.setObjectName("bellmanrevenir")
        self.tableWidget = QtWidgets.QTableWidget(BellmanFord)
        self.tableWidget.setGeometry(QtCore.QRect(30, 70, 351, 91))
        self.tableWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.sommetbutton = QtWidgets.QPushButton(BellmanFord)
        self.sommetbutton.setGeometry(QtCore.QRect(250, 30, 75, 23))
        self.sommetbutton.setStyleSheet("background-color: rgb(112, 186, 255);")
        self.sommetbutton.setObjectName("sommetbutton")
        self.inter = QtWidgets.QLabel(BellmanFord)
        self.inter.setGeometry(QtCore.QRect(70, 180, 271, 21))
        self.inter.setObjectName("inter")
        self.textBrowser = QtWidgets.QTextBrowser(BellmanFord)
        self.textBrowser.setGeometry(QtCore.QRect(80, 220, 256, 161))
        self.textBrowser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(BellmanFord)
        QtCore.QMetaObject.connectSlotsByName(BellmanFord)

    def retranslateUi(self, BellmanFord):
        _translate = QtCore.QCoreApplication.translate
        BellmanFord.setWindowTitle(_translate("BellmanFord", "bellmanford"))
        self.label.setText(_translate("BellmanFord", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Sommet :</span></p></body></html>"))
        self.sommetbutton.setText(_translate("BellmanFord", "valider"))
        self.inter.setText(_translate("BellmanFord", "<html><head/><body><p><br/></p></body></html>"))
        self.textBrowser.setHtml(_translate("BellmanFord", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    BellmanFord = QtWidgets.QWidget()
    ui = Ui_BellmanFord()
    ui.setupUi(BellmanFord)
    BellmanFord.show()
    sys.exit(app.exec_())
