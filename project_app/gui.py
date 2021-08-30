from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys
import requests
import os
import time
from videoprops import get_video_properties
import zipfile


fn = ''
ofn = ''
opfn = ''

class GetProgress(QThread):
    
    progressUpdated = pyqtSignal(int)

    def run(self):
        time.sleep(5)
        progress = 0
        count = 1
        while True:
            try:
                #response = requests.get('https://progress-dot-sincere-cacao-318706.uc.r.appspot.com')
                response = requests.get('http://172.17.0.2:5000/progress')
                progress = int(response.text)
                
                if progress == -1:
                    time.sleep(5)
                    continue
                elif progress == 100:
                    if count == 1:
                      continue
                    self.progressUpdated.emit(progress)
                    print(progress)                         
                    break
                else:
                    self.progressUpdated.emit(progress)
                    print(progress)
                    count+=1
                time.sleep(5)
            except requests.ConnectionError:
                print("Error occurred")
            
class PostVideo(QThread):

    finished = pyqtSignal(str)

    def run(self):
        global ofn
        global opfn

        try:
            #upload file to the server
            file = {"File":open(fn,'rb')}
            #response = requests.post('https://sincere-cacao-318706.uc.r.appspot.com', files = file )
            response = requests.post('http://172.17.0.2:5000/', files = file )
            
            if int(response.status_code) == 200:
                output_dir = './output'
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                path_to_zip_file = os.path.join(output_dir,'output.zip')
                
                outFile = open(path_to_zip_file, 'wb')
                outFile.write(response.content)
                outFile.close()

                ofn = os.path.join(output_dir+"/output", os.path.basename(fn))
                video_name = os.path.basename(os.path.basename(fn)).split('.')[0]
                opfn = os.path.join(output_dir+"/output_preds", video_name+'_preds.mp4')

                with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                
                os.remove(path_to_zip_file)

                
                self.finished.emit(ofn)
            else:
                self.finished.emit('error')
        except requests.ConnectionError:
            self.finished.emit('error')
            print("Error occurred")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = '神の目 Kami no me'
        self.left = 10
        self.top = 10
        self.width = 900
        self.height = 500
        self.fileName = ''
        self.outputFileName = ''
        #self.apiURL = "https://sincere-cacao-318706.uc.r.appspot.com"
        self.apiURL = 'http://172.17.0.2:5000/'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(self.width, self.height)

        widget = QWidget(self)
        self.setCentralWidget(widget)

        glayout = QGridLayout()

        widget.setLayout(glayout)

        # menubar
        menubar = self.menuBar()
        file = menubar.addMenu("File")

        openFileAction = QAction('&Open Video File', self)
        openFileAction.setShortcut('Ctrl+O')
        openFileAction.triggered.connect(self.openFileNameDialog)

        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(qApp.quit)

        file.addAction(openFileAction)
        file.addSeparator()
        file.addAction(exitAction)

        # labels
        self.label1 = QLabel("Input video:")
        self.label2 = QLabel("Output video:")
        self.label3 = QLabel("Output graph video:")

        glayout.addWidget(self.label1, 3, 0)
        glayout.addWidget(self.label2, 3, 1)
        glayout.addWidget(self.label3, 3, 2)

        # mediaplayers
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer2 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer3 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        videoWidget2 = QVideoWidget()
        videoWidget3 = QVideoWidget()

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer2.setVideoOutput(videoWidget2)
        self.mediaPlayer3.setVideoOutput(videoWidget3)

        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

        self.mediaPlayer2.positionChanged.connect(self.positionChanged2)
        self.mediaPlayer2.durationChanged.connect(self.durationChanged2)

        self.mediaPlayer3.positionChanged.connect(self.positionChanged3)
        self.mediaPlayer3.durationChanged.connect(self.durationChanged3)


        glayout.addWidget(videoWidget, 0, 0, 2, 1)
        glayout.addWidget(videoWidget2, 0, 1, 2, 1)
        glayout.addWidget(videoWidget3, 0, 2, 2, 1)


        #sliders
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.setRange(0, 0)
        self.slider1.sliderMoved.connect(self.setPosition)
        
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setRange(0, 0)
        self.slider2.sliderMoved.connect(self.setPosition2)

        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setRange(0, 0)
        self.slider3.sliderMoved.connect(self.setPosition3)

        glayout.addWidget(self.slider1,2,0)
        glayout.addWidget(self.slider2,2,1)
        glayout.addWidget(self.slider3,2,2)


        #progress bar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        glayout.addWidget(self.progressbar,4,0,1,3)


        # buttons
        self.pbutton = QPushButton()
        self.pbutton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.pbutton.setToolTip("Play input video")
        self.pbutton.setDisabled(True)

        self.pbutton2 = QPushButton()
        self.pbutton2.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.pbutton2.setToolTip("Play output video")
        self.pbutton2.setDisabled(True)

        self.pbutton3 = QPushButton()
        self.pbutton3.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.pbutton3.setToolTip("Play output graph video")
        self.pbutton3.setDisabled(True)

        self.pbutton.clicked.connect(self.play)
        self.pbutton2.clicked.connect(self.play2)
        self.pbutton3.clicked.connect(self.play3)

        glayout.addWidget(self.pbutton, 5, 0)
        glayout.addWidget(self.pbutton2, 5, 1)
        glayout.addWidget(self.pbutton3, 5, 2)

    def openFileNameDialog(self):
        global fn
        options = QFileDialog.Options()
        self.fileName, _ = QFileDialog.getOpenFileName(
            self, "Select a video", "", "mp4 Files (*.mp4)", options=options)
        if self.fileName != '':
            fn = self.fileName
            
            props = get_video_properties(self.fileName)

            self.label1.setText(f'''
            Input video:
            
            Duration: {props['duration']} s
            Codec: {props['codec_name']}
            Resolution: {props['width']}×{props['height']}
            Aspect ratio: {props['display_aspect_ratio']}
            Frame rate: {props['avg_frame_rate']}
            ''')

            #These threads will post the video and get progress
            self.gp = GetProgress()
            self.pv = PostVideo()

            
                       
            if self.pv.isRunning() or self.gp.isRunning():
                pass
            else:
                self.pv.start()
                self.gp.start()
                self.gp.progressUpdated.connect(self.onProgressChange)
                self.pv.finished.connect(self.onRunFinish)

            self.pbutton.setDisabled(False)
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.fileName)))
            self.mediaPlayer.play()
            self.pbutton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.showdialog(QMessageBox.Information,"Choose File", "Select an input video file to proceed")

    def onRunFinish(self, value):
        if value != 'error':
            self.outputFileName = value
            if os.path.isfile(self.outputFileName):
                self.pbutton2.setDisabled(False)
                self.pbutton3.setDisabled(False)
                #self.mediaPlayer2.setMedia(QMediaContent(QUrl.fromLocalFile(self.outputFileName)))
                self.mediaPlayer2.setMedia(QMediaContent(QUrl.fromLocalFile(QFileInfo(self.outputFileName).absoluteFilePath())))
                self.mediaPlayer2.play()
                #self.mediaPlayer3.setMedia(QMediaContent(QUrl.fromLocalFile(opfn)))
                self.mediaPlayer3.setMedia(QMediaContent(QUrl.fromLocalFile(QFileInfo(opfn).absoluteFilePath())))
                self.mediaPlayer3.play()

                props = get_video_properties(self.outputFileName)
                prop = get_video_properties(opfn)

                self.label2.setText(f'''
                Output video:

                Duration: {props['duration']} s
                Codec: {props['codec_name']}
                Resolution: {props['width']}×{props['height']}
                Frame rate: {props['avg_frame_rate']}
                ''')

                self.label3.setText(f'''
                Output graph video:

                Duration: {prop['duration']} s
                Codec: {prop['codec_name']}
                Resolution: {prop['width']}×{props['height']}
                Frame rate: {prop['avg_frame_rate']}
                ''')
            else:
                self.gp.terminate()
                self.progressbar.setValue(0)
                self.showdialog(icon=QMessageBox.Warning,title="Output File Unavailable", text="Error occured during processing")
        else:
            self.gp.terminate()
            self.progressbar.setValue(0)
            self.showdialog(icon=QMessageBox.Warning,title="Output File Unavailable", text="Error occured during processing")

    def onProgressChange(self, value):
        self.progressbar.setValue(value)
            
    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.pbutton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.mediaPlayer.play()
            self.pbutton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def play2(self):
        if os.path.isfile(self.outputFileName):
            if self.mediaPlayer2.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer2.pause()
                self.pbutton2.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self.mediaPlayer2.play()
                self.pbutton2.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.showdialog(icon=QMessageBox.Warning,title="Output File Unavailable", text="Error occured during processing")

    def play3(self):
        if os.path.isfile(self.outputFileName):
            if self.mediaPlayer3.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer3.pause()
                self.pbutton3.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            else:
                self.mediaPlayer3.play()
                self.pbutton3.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def positionChanged(self, position):
        self.slider1.setValue(position)

    def durationChanged(self, duration):
        self.slider1.setRange(0, duration)
 
    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
    
    def positionChanged2(self, position):
        self.slider2.setValue(position)

    def durationChanged2(self, duration):
        self.slider2.setRange(0, duration)
 
    def setPosition2(self, position):
        self.mediaPlayer2.setPosition(position)

    def positionChanged3(self, position):
        self.slider3.setValue(position)

    def durationChanged3(self, duration):
        self.slider3.setRange(0, duration)
 
    def setPosition3(self, position):
        self.mediaPlayer3.setPosition(position)

    def showdialog(self, icon, title, text):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setText(text)
        # msg.setInformativeText("Select a video file to proceed")
        msg.setWindowTitle(title)
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())

