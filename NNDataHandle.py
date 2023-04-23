import sqlite3
import matplotlib
import matplotlib.pyplot
import os

FullNNList = []
con = sqlite3.connect("NNData.db")
cur = con.cursor()
for row in cur.execute("SELECT modelNo, epoch, loss FROM data ORDER BY modelNo"):
    FullNNList.append(row)
dirList = os.listdir(path=os.getcwd())
if(('GraphFolder' in dirList) == False):
    os.mkdir(os.getcwd()+'\\GraphFolder')
print(FullNNList)
for i in range(max(FullNNList[:][0])+1):
    xList = []
    yList = []
    for j in range(len(FullNNList)):
        if FullNNList[j][0] == i:
            xList.append(FullNNList[j][1])
            yList.append(FullNNList[j][2])
    if len(xList) > 50:
        workingFig = matplotlib.pyplot.plot(xList, yList)
        matplotlib.pyplot.title('Model'+str(i))
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Loss')
        matplotlib.pyplot.savefig(os.getcwd()+'\\GraphFolder\\Model'+str(i)+' epochloss.png', dpi='figure', format='png')
        matplotlib.pyplot.close('all')