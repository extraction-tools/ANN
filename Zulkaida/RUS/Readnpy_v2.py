import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import ROOT as root
data = np.load('generated_events_april13.npy')

#print(event.shape)
#print(event[1][6200])
#print(event)

detId = np.array([])
eleId = np.array([])
h   = root.TH2D("Generated Event","Generated Event",31,0,31,201,0,201)

for i in range(10000):
  print(i)
  for j in range(1,31):
    for k in range(1,201):
      if(data[i][200*j+k] > 0):
        #x = j
        #y = k
        #detId = np.append(detId, x)
        #eleId = np.append(eleId, y)
        h.Fill(j,k,data[i][200*j+k])

hist_content = np.array([])
for i in range(31):
  for j in range(201):
    hist_content = np.append(hist_content, h.GetBinContent(i+1, j+1))

c = root.TCanvas('c','c',600,600)
h.SetMarkerColor(4)
c.Draw('COLZ')
h.Draw()
c.SaveAs("Generated_Event.jpg")

## save in txt
dat = np.array([hist_content])
dat = dat.T
np.savetxt('HistoContent.txt', dat, delimiter = '\t', fmt='%i')

