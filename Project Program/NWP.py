from PyQt5 import QtWidgets, uic, QtGui
import os
import tkinter
from tkinter import filedialog
import sys
import webbrowser
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle

#from charset_normalizer import md__mypyc

from keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seq_len, seed_text):

    # Intial Seed Sequence
    input_text = seed_text    

    # Take the input text string and encode it to a sequence
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]

    # Pad sequences to our trained rate (25 words in the video)
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

    # Predict Class Probabilities for each word
    predict_x=model.predict(pad_encoded)
    pred_word_ind=np.argmax(predict_x,axis=1)[0]

    # Grab word
    pred_word = tokenizer.index_word[pred_word_ind]

    
    return pred_word

class NWP(QtWidgets.QMainWindow):

    model_cnn = None
    model_gru = None
    model_lstm = None
    tok_cnn = None
    tok_gru = None
    tok_lstm = None
    check = False

    def predictText(self):
        # self.predictedText.setText(self.textEdit.toPlainText())
        if self.check:
            if len(self.textEdit.toPlainText().strip()) == 0:
                self.predictedText.setText("Prediction is at its best after 5 words.")
            elif self.lstmradioButton.isChecked():
                self.predictedText.setText(generate_text(
                    self.model_lstm,
                    self.tok_lstm,
                    5,
                    self.textEdit.toPlainText()
                ))
            elif self.gruradioButton.isChecked():
                self.predictedText.setText(generate_text(
                    self.model_gru,
                    self.tok_gru,
                    5,
                    self.textEdit.toPlainText()
                ))
            elif self.cnnradioButton.isChecked():
                self.predictedText.setText(generate_text(
                    self.model_cnn,
                    self.tok_cnn,
                    5,
                    self.textEdit.toPlainText()
                ))


    
    def __init__(self):

        super(NWP,self).__init__()
        uic.loadUi('res/ui/NWP.ui',self)

        

        try:
            json_file = open('res\models\modelcnn.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_cnn = model_from_json(loaded_model_json)
            self.model_cnn.load_weights("res\models\cnn.h5")
            self.model_cnn.compile(
                optimizer = 'adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.tok_cnn = pickle.load(open('res\models\cnntok.sav','rb'))
            self.cnnradioButton.setChecked(True)
            self.check = True
        except:
            self.cnnradioButton.setEnabled(False)
        
        try:
            json_file = open('res\models\modelgru.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_gru = model_from_json(loaded_model_json)
            self.model_gru.load_weights("res\models\gru.h5")
            self.model_gru.compile(
                optimizer = 'adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.tok_gru = pickle.load(open('res\models\grutok.sav','rb'))
            self.gruradioButton.setChecked(True)
            self.check = True
        except:
            self.gruradioButton.setEnabled(False)

        try:
            json_file = open('res\models\modellstm.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model_lstm = model_from_json(loaded_model_json)
            self.model_lstm.load_weights("res\models\lstm.h5")
            self.model_lstm.compile(
                optimizer = 'adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.tok_lstm = pickle.load(open('res\models\lstmtok.sav','rb'))
            self.lstmradioButton.setChecked(True)
            self.check = True
        except:
            self.lstmradioButton.setEnabled(False)

        if not self.check:
            self.predictedText.setText("Please make sure model files are placed correctly.")
        else:
            self.predictedText.setText("Prediction is at its best after 5 words.")


        self.setWindowIcon(QtGui.QIcon('res/img.png'))
        self.setWindowTitle("Next Word Predictor")

        self.setFixedSize(1112, 858)
        self.textEdit.textChanged.connect(self.predictText)

        self.show()

        ##This is for finding functions using auto complete as they cannot be found with the item loaded in the .ui
        # self.x = QtWidgets.QRadioButton(self.centralwidget)
        # self.x.isChecked()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = NWP()
    app.exec_()