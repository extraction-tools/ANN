import uproot
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

file = uproot.open("sim_tree3c.root")
#file = uproot.open("simDY1.root")
tree = file["tree"]

#Some of Event Information
EventId = tree["EventData/event_id"].array(library="np")
nTruthTracks = tree["EventData/n_truth_tracks"].array(library="np")
nHitsAll = tree["EventData/n_hits_all"].array(library="np")

#Some of Hits Information
HitId = tree["HitList.hit_id"].array(library="np")
DetectorId = tree["HitList.detector_id"].array(library="np")
ElementId = tree["HitList.element_id"].array(library="np")
DriftDistance = tree["HitList.drift_distance"].array(library="np")
HitTrackId = tree["HitList.track_id"].array(library="np")
HitProcessId = tree["HitList.process_id"].array(library="np")
TDCtime = tree["HitList.tdc_time"].array(library="np")

#Some of Truth Track Information
TrackId = tree["TruthTrackList.track_id"].array(library="np")
TrackCharge = tree["TruthTrackList.charge_id"].array(library="np")
Trackgpx = tree["TruthTrackList.gpx"].array(library="np")
Trackgpy = tree["TruthTrackList.gpy"].array(library="np")
Trackgpz = tree["TruthTrackList.gpz"].array(library="np")
Trackgvtx = tree["TruthTrackList.gvtx"].array(library="np")
Trackgvty = tree["TruthTrackList.gvty"].array(library="np")
Trackgvtz = tree["TruthTrackList.gvtz"].array(library="np")

#Some of Kalman Reco Track Information
KRecpx = tree["RecTrackList.px"].array(library="np")
KRecpy = tree["RecTrackList.py"].array(library="np")
KRecpz = tree["RecTrackList.pz"].array(library="np")
KRecvtx = tree["RecTrackList.vtx"].array(library="np")
KRecvty = tree["RecTrackList.vty"].array(library="np")
KRecvtz = tree["RecTrackList.vtz"].array(library="np")

#Some of Truth Dimuon 
gDimuM = tree["TruthDimuon.mass"].array(library="np")
gDimuPhi = tree["TruthDimuon.phi"].array(library="np")
gDimuXf = tree["TruthDimuon.xF"].array(library="np")
gDimuE = tree["TruthDimuon.dimu_E"].array(library="np")

#Some of Reco Dimuon
RecDimuM = tree["RecDimuon.mass"].array(library="np")
RecDimuE = tree["RecDimuon.dimu_E"].array(library="np")

# Print general info
nEvents = EventId.size
print("Number of Events = {}".format(str(nEvents)))

print("Printing General Information for each event")
print("==================================")
TDC_D0X = np.array([])
TDC_D2X = np.array([])
for id in range(0,nEvents):
  print("================================")
  print("Event Id = {}".format(str(EventId[id])))
  print("Number of Truth tracks = {}".format(str(nTruthTracks[id])))
  print("Number of All hits = {}".format(str(nHitsAll[id])))
  print("================================")
  
  for id_tracks in range(0,TrackId[id].size):
    print("Track Id | Charge | gpx | gpy | gpz | vtx | vty | vtz")
    print('%i   %i   %2f  %2f  %2f  %2f  %2f  %2f ' %(TrackId[id][id_tracks], TrackCharge[id][id_tracks], Trackgpx[id][id_tracks], Trackgpy[id][id_tracks], Trackgpz[id][id_tracks], Trackgvtx[id][id_tracks], Trackgvty[id][id_tracks], Trackgvtz[id][id_tracks]) )
    print("Hit Information for this track: ")
    print("Detector Id |  Element Id | Drift distance | Process Id") 
    for id_hit in range(nHitsAll[id]):
      if HitTrackId[id][id_hit] == TrackId[id][id_tracks]:
        print('%i %i %3f %i ' %(DetectorId[id][id_hit], ElementId[id][id_hit], DriftDistance[id][id_hit], HitProcessId[id][id_hit]))
        if DetectorId[id][id_hit] ==3:
           TDC_D0X = np.append(TDC_D0X, TDCtime[id][id_hit])
        if DetectorId[id][id_hit] ==16:
           TDC_D2X = np.append(TDC_D2X, TDCtime[id][id_hit])
    print("____________________________________________") 

# TrackQA analysis (examples)
DeltaPx = np.array([])
DeltaPz = np.array([])
DeltaVx = np.array([])
DeltaVy = np.array([])
for id_event in range(0, nEvents):
  for id_tracks in range(0,TrackId[id_event].size):
     if KRecpx[id_event][id_tracks] < 999:
       DeltaPx = np.append(DeltaPx, Trackgpx[id_event][id_tracks] - KRecpx[id_event][id_tracks])
       DeltaPz = np.append(DeltaPz, Trackgpz[id_event][id_tracks] - KRecpz[id_event][id_tracks])
       DeltaVx = np.append(DeltaVx, Trackgvtx[id_event][id_tracks] - KRecvtx[id_event][id_tracks])
       DeltaVy = np.append(DeltaVy, Trackgvty[id_event][id_tracks] - KRecvty[id_event][id_tracks])

DeltaDimuM = np.array([])
DeltaDimuE = np.array([])
for id_event in range(0, nEvents):
  for id_tracks in range(0,gDimuM[id_event].size):
     if RecDimuE[id_event][id_tracks] < 999:
       DeltaDimuM = np.append(DeltaDimuM, gDimuM[id_event][id_tracks] - RecDimuM[id_event][id_tracks])
       DeltaDimuE = np.append(DeltaDimuE, gDimuE[id_event][id_tracks] - RecDimuE[id_event][id_tracks])

Plot_DimuM = np.array([])
Plot_DimuXf = np.array([])
Plot_DimuPhi = np.array([])
for id_event in range(0, nEvents):
  for id_dimu in range(0,gDimuM[id_event].size):
    Plot_DimuM = np.append(Plot_DimuM, gDimuM[id_event][id_dimu])
    Plot_DimuXf = np.append(Plot_DimuXf, gDimuXf[id_event][id_dimu])
    Plot_DimuPhi = np.append(Plot_DimuPhi, gDimuPhi[id_event][id_dimu])


#plotting
plt.hist(DeltaPx, bins = 'auto')
plt.title("Delta Px (Truth - Reco)")
plt.xlabel("GeV")
plt.savefig("DpX.png")
plt.clf()

plt.hist(DeltaPz, bins = 'auto')
plt.title("Delta Pz (Truth - Reco)")
plt.xlabel("GeV")
plt.savefig("DpZ.png")
plt.clf()

plt.hist(DeltaVx, bins = 'auto')
plt.title("Delta vertex-x (Truth - Reco)")
plt.xlabel("cm")
plt.savefig("DVertX.png")
plt.clf()

plt.hist(DeltaVy, bins = 'auto')
plt.title("Delta vertex-y (Truth - Reco)")
plt.xlabel("cm")
plt.savefig("DVertY.png")
plt.clf()

plt.hist(Plot_DimuM, bins = 'auto')
plt.title("Dimuon Mass")
plt.xlabel("GeV")
plt.savefig("gDimuM.png")
plt.clf()

plt.hist(Plot_DimuPhi, bins = 'auto')
plt.title("Dimuon Phi")
plt.xlabel("Phi")
plt.savefig("gDimuPhi.png")
plt.clf()

plt.hist(DeltaDimuM, bins = 'auto')
plt.title("Delta M (Truth - Reco)")
plt.savefig("DeltaDimuM.png")
plt.clf()

plt.hist(Plot_DimuXf, bins = 'auto')
plt.title("Dimuon xF")
plt.xlabel("xF")
plt.savefig("gDimuXf.png")
plt.clf()

plt.hist(DeltaDimuE, bins = 'auto')
plt.title("Delta E (Truth - Reco)")
plt.savefig("DeltaDimuE.png")
plt.clf()

plt.hist(TDC_D0X, bins = 'auto')
plt.title("D0X TDC time")
plt.xlabel("TDC time (ns)")
plt.savefig("TDCD0X.png")
plt.clf()


plt.hist(TDC_D2X, bins = 'auto')
plt.title("D2X TDC time")
plt.xlabel("TDC time (ns)")
plt.savefig("TDCD2X.png")
plt.clf()

print(gDimuXf.shape)
print(DeltaPx.shape)
print(gDimuXf.ndim)
print(DeltaPx.ndim)
print(DeltaPx)
