import uproot
import numpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

#Import the Drell-Yan event from Target
file = uproot.open("DYTarget_April13.root")
tree = file["tree"]

#Some of Event Information
EventId = tree["EventData/event_id"].array(library="np")
nTruthTracks = tree["EventData/n_truth_tracks"].array(library="np")
nHitsAll = tree["EventData/n_hits_all"].array(library="np")

targetevents = EventId.size

#Some of Hits Information
HitId = tree["HitList.hit_id"].array(library="np")
DetectorId = tree["HitList.detector_id"].array(library="np")
ElementId = tree["HitList.element_id"].array(library="np")
DriftDistance = tree["HitList.drift_distance"].array(library="np")
HitTrackId = tree["HitList.track_id"].array(library="np")
#HitProcessId = tree["HitList.process_id"].array(library="np")
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

#This reads the dimuon tracks from the target into an array
pos_events=np.zeros((targetevents,100))
neg_events=np.zeros((targetevents,100))
pos_kinematics=np.zeros((targetevents,3))
neg_kinematics=np.zeros((targetevents,3))
vertex=np.zeros((targetevents,3))

#Saving time information
tdc_pos_events=np.zeros((targetevents,100))
tdc_neg_events=np.zeros((targetevents,100))
drift_pos_events=np.zeros((targetevents,100))
drift_neg_events=np.zeros((targetevents,100))

#Maximum element
MaxEle = np.array([0,201,201,160,140,201,201,0,0,0,0,0,0,128,128,112,112,128,128, 134, 134, 116, 116, 134, 134, 134, 134, 116, 116, 134, 134])

for id_event in range(targetevents):
    if(id_event%100==0):print(id_event,end="\r") #This is to keep track of how quickly the events are being generated
    for id_tracks in range(0,TrackId[id_event].size):
        if TrackCharge[id_event][id_tracks] == 1:
           pos_kinematics[id_event][0] = Trackgpx[id_event][id_tracks]
           pos_kinematics[id_event][1] = Trackgpy[id_event][id_tracks]
           pos_kinematics[id_event][2] = Trackgpz[id_event][id_tracks]

           #vertex information. It does not matted we take it from positif or negativa track
           vertex[id_event][0] = Trackgvtx[id_event][id_tracks]
           vertex[id_event][1] = Trackgvty[id_event][id_tracks]
           vertex[id_event][2] = Trackgvtz[id_event][id_tracks]
          
        if TrackCharge[id_event][id_tracks] == -1:
           neg_kinematics[id_event][0] = Trackgpx[id_event][id_tracks]
           neg_kinematics[id_event][1] = Trackgpy[id_event][id_tracks]
           neg_kinematics[id_event][2] = Trackgpz[id_event][id_tracks]
           
        for id_hit in range(nHitsAll[id_event]):
            if HitTrackId[id_event][id_hit] == TrackId[id_event][id_tracks] and TrackCharge[id_event][id_tracks] == 1:
               pos_events[id_event][DetectorId[id_event][id_hit]] = ElementId[id_event][id_hit]
               tdc_pos_events[id_event][DetectorId[id_event][id_hit]] = TDCtime[id_event][id_hit]
               drift_pos_events[id_event][DetectorId[id_event][id_hit]] = DriftDistance[id_event][id_hit]
            if HitTrackId[id_event][id_hit] == TrackId[id_event][id_tracks] and TrackCharge[id_event][id_tracks] == -1:
               neg_events[id_event][DetectorId[id_event][id_hit]] = ElementId[id_event][id_hit]
               tdc_neg_events[id_event][DetectorId[id_event][id_hit]] = TDCtime[id_event][id_hit]
               drift_neg_events[id_event][DetectorId[id_event][id_hit]] = DriftDistance[id_event][id_hit]
              

## This part read the second file (Dump events for noise). We will overwrite most variables/array above
#Import the JPsi event from Dump
file = uproot.open("JPsiDump_April13.root")
tree = file["tree"]

#Some of Event Information
EventId = tree["EventData/event_id"].array(library="np")
nTruthTracks = tree["EventData/n_truth_tracks"].array(library="np")
nHitsAll = tree["EventData/n_hits_all"].array(library="np")
dumpevents = EventId.size

#Some of Hits Information
HitId = tree["HitList.hit_id"].array(library="np")
DetectorId = tree["HitList.detector_id"].array(library="np")
ElementId = tree["HitList.element_id"].array(library="np")
DriftDistance = tree["HitList.drift_distance"].array(library="np")
HitTrackId = tree["HitList.track_id"].array(library="np")
#HitProcessId = tree["HitList.process_id"].array(library="np")
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

#This reads the dimuon tracks from dump into an array
pos_noise_events=np.zeros((dumpevents,100))
neg_noise_events=np.zeros((dumpevents,100))
pos_noise_kinematics=np.zeros((dumpevents,3))
neg_noise_kinematics=np.zeros((dumpevents,3))

#time and drift
tdc_pos_noise_events=np.zeros((dumpevents,100))
tdc_neg_noise_events=np.zeros((dumpevents,100))
drift_pos_noise_events=np.zeros((dumpevents,100))
drift_neg_noise_events=np.zeros((dumpevents,100))

for id_event in range(dumpevents):
    if(id_event%100==0):print(id_event,end="\r") #This is to keep track of how quickly the events are being generated
    for id_tracks in range(0,TrackId[id_event].size):
        if TrackCharge[id_event][id_tracks] == 1:
           pos_noise_kinematics[id_event][0] = Trackgpx[id_event][id_tracks]
           pos_noise_kinematics[id_event][1] = Trackgpy[id_event][id_tracks]
           pos_noise_kinematics[id_event][2] = Trackgpz[id_event][id_tracks]
           
        if TrackCharge[id_event][id_tracks] == -1:
           neg_noise_kinematics[id_event][0] = Trackgpx[id_event][id_tracks]
           neg_noise_kinematics[id_event][1] = Trackgpy[id_event][id_tracks]
           neg_noise_kinematics[id_event][2] = Trackgpz[id_event][id_tracks]
        for id_hit in range(nHitsAll[id_event]):
            if HitTrackId[id_event][id_hit] == TrackId[id_event][id_tracks] and TrackCharge[id_event][id_tracks] == 1:
               pos_noise_events[id_event][DetectorId[id_event][id_hit]] = ElementId[id_event][id_hit]
               tdc_pos_noise_events[id_event][DetectorId[id_event][id_hit]] = TDCtime[id_event][id_hit]
               drift_pos_noise_events[id_event][DetectorId[id_event][id_hit]] = DriftDistance[id_event][id_hit]
            if HitTrackId[id_event][id_hit] == TrackId[id_event][id_tracks] and TrackCharge[id_event][id_tracks] == -1:
               neg_noise_events[id_event][DetectorId[id_event][id_hit]] = ElementId[id_event][id_hit]
               tdc_neg_noise_events[id_event][DetectorId[id_event][id_hit]] = TDCtime[id_event][id_hit]
               drift_neg_noise_events[id_event][DetectorId[id_event][id_hit]] = DriftDistance[id_event][id_hit]


#Clean input data
for j in range(targetevents):
    for i in range(100):
        if(pos_events[j][i]>1000):
           pos_events[j][i]=0
           tdc_pos_events[j][i]=0
           drift_pos_events[j][i]=0
        if(neg_events[j][i]>1000):
           neg_events[j][i]=0
           tdc_neg_events[j][i]=0
           drift_neg_events[j][i]=0
for j in range(dumpevents):
    for i in range(100):
        if(pos_noise_events[j][i]>1000):
           pos_noise_events[j][i]=0
           tdc_pos_noise_events[j][i]=0
           drift_pos_noise_events[j][i]=0
        if(neg_noise_events[j][i]>1000):
           neg_noise_events[j][i]=0
           tdc_neg_noise_events[j][i]=0
           drift_neg_noise_events[j][i]=0

#Background tracklets
#This generates 1 million partial tracks that can be combined with full tracks to make events.
noise_tracklets=np.zeros((1000000,100))
tdc_noise_tracklets=np.zeros((1000000,100))
drift_noise_tracklets=np.zeros((1000000,100))
for j in range(1000000):
    if(j%100==0):print(j,end="\r") #This is to keep track of how quickly the events are being generated
    for count in range(1):
        st=random.randrange(1,4)
        l=random.randrange(0,4)
        if(l==0):
            m=random.randrange(dumpevents)
            for det_id in range(100):
                if(st==1 and det_id < 7):
                   noise_tracklets[j][det_id]=pos_noise_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_pos_noise_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_pos_noise_events[m][det_id]
                if(st==2 and det_id > 12 and det_id < 19):
                   noise_tracklets[j][det_id]=pos_noise_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_pos_noise_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_pos_noise_events[m][det_id]
                if(st==3 and det_id > 18 and det_id < 31):
                   noise_tracklets[j][det_id]=pos_noise_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_pos_noise_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_pos_noise_events[m][det_id]

        if(l==1):
            m=random.randrange(dumpevents)
            for det_id in range(100):
                if(st==1 and det_id < 7):
                   noise_tracklets[j][det_id]=neg_noise_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_neg_noise_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_neg_noise_events[m][det_id]
                if(st==2 and det_id > 12 and det_id < 19):
                   noise_tracklets[j][det_id]=neg_noise_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_neg_noise_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_neg_noise_events[m][det_id]
                if(st==3 and det_id > 18 and det_id < 31):
                   noise_tracklets[j][det_id]=neg_noise_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_neg_noise_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_neg_noise_events[m][det_id]        

        if(l==2):
            m=random.randrange(targetevents)
            for det_id in range(100):
                if(st==1 and det_id < 7):
                   noise_tracklets[j][det_id]=pos_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_pos_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_pos_events[m][det_id]
                if(st==2 and det_id > 12 and det_id < 19):
                   noise_tracklets[j][det_id]=pos_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_pos_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_pos_events[m][det_id]
                if(st==3 and det_id > 18 and det_id < 31):
                   noise_tracklets[j][det_id]=pos_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_pos_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_pos_events[m][det_id]

        if(l==3):
            m=random.randrange(targetevents)
            for det_id in range(100):
                if(st==1 and det_id < 7):
                   noise_tracklets[j][det_id]=neg_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_neg_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_neg_events[m][det_id]
                if(st==2 and det_id > 12 and det_id < 19):
                   noise_tracklets[j][det_id]=neg_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_neg_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_neg_events[m][det_id]
                if(st==3 and det_id > 18 and det_id < 31):
                   noise_tracklets[j][det_id]=neg_events[m][det_id]
                   tdc_noise_tracklets[j][det_id]=tdc_neg_events[m][det_id]
                   drift_noise_tracklets[j][det_id]=drift_neg_events[m][det_id]


#Start generating the events

signalinput=np.zeros((10000,20000))
tdc_signalinput=np.zeros((10000,20000))
drift_signalinput=np.zeros((10000,20000))
tracks=np.zeros((10000,1))
dimuons=np.zeros((10000,1))
signals=np.zeros((10000,1))

for z in range(10000):
    if(z%100==0):print(z,end="\r") #This is to keep track of how quickly the events are being generated
    m=np.random.poisson(2, 1)[0] #Random number of full tracks to input
    tracks[z][0]=m
    for i in range(m):
        l=random.randrange(1,4) #Select positive or negative target or dump event
        j=random.randrange(targetevents) #Select random event number
        n=random.randrange(0,2) #Select whether or not it's a dimuon event
        if(n==1):
            for k in range(100):
                if(pos_events[j][k]>0 and l==1):
                    signalinput[z][int(200*k+pos_events[j][k])]=1
                    tdc_signalinput[z][int(200*k+pos_events[j][k])]= tdc_pos_events[j][k]
                    drift_signalinput[z][int(200*k+pos_events[j][k])]= drift_pos_events[j][k]
                if(neg_events[j][k]>0 and l==2):
                    signalinput[z][int(200*k+neg_events[j][k])]=1
                    tdc_signalinput[z][int(200*k+neg_events[j][k])]=tdc_neg_events[j][k]
                    drift_signalinput[z][int(200*k+neg_events[j][k])]=drift_neg_events[j][k]
                if(pos_noise_events[j][k]>0 and (l==3)):
                    signalinput[z][int(200*k+pos_noise_events[j][k])]=1
                    tdc_signalinput[z][int(200*k+pos_noise_events[j][k])]=tdc_pos_noise_events[j][k]
                    drift_signalinput[z][int(200*k+pos_noise_events[j][k])]=drift_pos_noise_events[j][k]
                if(neg_noise_events[j][k]>0 and l==4):
                    signalinput[z][int(200*k+neg_noise_events[j][k])]=1
                    tdc_signalinput[z][int(200*k+neg_noise_events[j][k])]=tdc_neg_noise_events[j][k]
                    drift_signalinput[z][int(200*k+neg_noise_events[j][k])]=drift_neg_noise_events[j][k]
        if(n!=1):
            for k in range(100):
                if(pos_events[j][k]>0 and l<=2):
                    signalinput[z][int(200*k+pos_events[j][k])]=10
                    tdc_signalinput[z][int(200*k+pos_events[j][k])]=tdc_pos_events[j][k]
                    drift_signalinput[z][int(200*k+pos_events[j][k])]=drift_pos_events[j][k]
                if(neg_events[j][k]>0 and l<=2):
                    signalinput[z][int(200*k+neg_events[j][k])]=10
                    tdc_signalinput[z][int(200*k+neg_events[j][k])]=tdc_neg_events[j][k]
                    drift_signalinput[z][int(200*k+neg_events[j][k])]=drift_neg_events[j][k]
                if(pos_noise_events[j][k]>0 and l>2):
                    signalinput[z][int(200*k+pos_noise_events[j][k])]=1
                    tdc_signalinput[z][int(200*k+pos_noise_events[j][k])]=tdc_pos_noise_events[j][k]
                    drift_signalinput[z][int(200*k+pos_noise_events[j][k])]=drift_pos_noise_events[j][k]
                if(neg_noise_events[j][k]>0 and l>2):
                    signalinput[z][int(200*k+neg_noise_events[j][k])]=1
                    tdc_signalinput[z][int(200*k+neg_noise_events[j][k])]=tdc_neg_noise_events[j][k]
                    drift_signalinput[z][int(200*k+neg_noise_events[j][k])]=drift_neg_noise_events[j][k]
        if(n!=1):
            tracks[z][0]=tracks[z][0]+1
            if(l==1 or l==2):
                signals[z][0]=1
    if(tracks[z][0]==2):
        dimuons[z][0]=1
    if(tracks[z][0]!=2 and signals[z][0]==1):signals[z][0]=0


#Add partial tracks to the data
for j in range(len(signalinput)):
    if(j%100==0):print(j,end="\r")
    tracklets=50-3*tracks[j][0].astype(int)
    for k in range(tracklets):
        m=random.randrange(1000000)
        for n in range(100):
            if(noise_tracklets[m][n]>0):
                signalinput[j][int(200*n+noise_tracklets[m][n])]=1
                tdc_signalinput[j][int(200*n+noise_tracklets[m][n])]=tdc_noise_tracklets[m][n]
                drift_signalinput[j][int(200*n+noise_tracklets[m][n])]=drift_noise_tracklets[m][n]



#Add edge hits
perc_edge = 5 #percentage of edge hits
cell_size_DC1 = 0.6 #cm
cell_size_DC2 = 2.
for j in range (len(signalinput)):
    if(j%100==0):print(j,end="\r")
    for k in range(31):
        if(k<7):
            for l in range(200-1):
                if(random.randrange(100)<perc_edge and tdc_signalinput[j][200*k+l] > 0.4*cell_size_DC1 and l < MaxEle[k]):
                   signalinput[j][200*k+l+1]=1
                   tdc_signalinput[j][200*k+l+1]= 0.9*cell_size_DC1
                   drift_signalinput[j][200*k+l+1]= (tdc_signalinput[j][200*k+l+1] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]
        if(k>12):
            for l in range(134-1):
                if(random.randrange(100)<perc_edge and tdc_signalinput[j][200*k+l] > 0.4*cell_size_DC2 and l < MaxEle[k]):
                   signalinput[j][200*k+l+1]=1
                   tdc_signalinput[j][200*k+l+1]= 0.9*cell_size_DC2
                   drift_signalinput[j][200*k+l+1]= (tdc_signalinput[j][200*k+l+1] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]

#Add electronic noise
perc_noise = 3 #percentage of noise
for j in range (len(signalinput)):
    if(j%100==0):print(j,end="\r")
    for k in range(31):
        m=random.randrange(targetevents)
        if(k<7):
            for l in range(200-2):
                if(random.randrange(100)<perc_noise and l < MaxEle[k]):
                   signalinput[j][200*k+l]=1
                   signalinput[j][200*k+l+1]=1
                   signalinput[j][200*k+l+2]=1

                   tdc_signalinput[j][200*k+l]=tdc_pos_events[m][k]
                   tdc_signalinput[j][200*k+l+1]=tdc_pos_events[m][k] - 5
                   tdc_signalinput[j][200*k+l+2]=tdc_pos_events[m][k] + 5

                   drift_signalinput[j][200*k+l]=drift_pos_events[m][k]
                   drift_signalinput[j][200*k+l+1]= (tdc_signalinput[j][200*k+l+1] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]
                   drift_signalinput[j][200*k+l+2]= (tdc_signalinput[j][200*k+l+2] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]

        if(k>12):
             for l in range(134-2):
                 if(random.randrange(100)<perc_noise and l < MaxEle[k]):
                    signalinput[j][200*k+l]=1
                    signalinput[j][200*k+l+1]=1
                    signalinput[j][200*k+l+2]=1

                    tdc_signalinput[j][200*k+l]=tdc_pos_events[m][k]
                    tdc_signalinput[j][200*k+l+1]=tdc_pos_events[m][k] - 5
                    tdc_signalinput[j][200*k+l+2]=tdc_pos_events[m][k] + 5

                    drift_signalinput[j][200*k+l]=drift_pos_events[m][k]
                    drift_signalinput[j][200*k+l+1]= (tdc_signalinput[j][200*k+l+1] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]
                    drift_signalinput[j][200*k+l+2]= (tdc_signalinput[j][200*k+l+2] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]                   

# add delta rays
perc_delta = 10
for j in range (len(signalinput)):
    if(j%100==0):print(j,end="\r")
    for k in range(31):
        m=random.randrange(targetevents)
        if(k<7):
            for l in range(200-2):
                if(random.randrange(100)<perc_delta and signalinput[j][200*k+l] ==10 and l < MaxEle[k]):
                   signalinput[j][200*k+l+1]=1
                   signalinput[j][200*k+l+2]=1

                   tdc_signalinput[j][200*k+l]=tdc_pos_events[m][k]
                   tdc_signalinput[j][200*k+l+1]=tdc_pos_events[m][k] - 15
                   tdc_signalinput[j][200*k+l+2]=tdc_pos_events[m][k] + 15

                   drift_signalinput[j][200*k+l]=drift_pos_events[m][k]
                   drift_signalinput[j][200*k+l+1]= (tdc_signalinput[j][200*k+l+1] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]
                   drift_signalinput[j][200*k+l+2]= (tdc_signalinput[j][200*k+l+2] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]
        if(k>12):
            for l in range(134-2):
                if(random.randrange(100)<perc_delta and signalinput[j][200*k+l] ==10 and l < MaxEle[k]):
                   signalinput[j][200*k+l+1]=1
                   signalinput[j][200*k+l+2]=1

                   tdc_signalinput[j][200*k+l]=tdc_pos_events[m][k]
                   tdc_signalinput[j][200*k+l+1]=tdc_pos_events[m][k] - 15
                   tdc_signalinput[j][200*k+l+2]=tdc_pos_events[m][k] + 15

                   drift_signalinput[j][200*k+l]=drift_pos_events[m][k]
                   drift_signalinput[j][200*k+l+1]= (tdc_signalinput[j][200*k+l+1] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]
                   drift_signalinput[j][200*k+l+2]= (tdc_signalinput[j][200*k+l+2] / tdc_signalinput[j][200*k+l]) * drift_signalinput[j][200*k+l]

#Add out of time noise
perc_oot = 2 #percentage of noise
TDC_OOT_DC1 = 2000
Drift_OOT_DC1 = 2
TDC_OOT_DC2 = 1500
Drift_OOT_DC2 = 3
for j in range (len(signalinput)):
    if(j%100==0):print(j,end="\r")
    for k in range(31):
        m=random.randrange(targetevents)
        if(k<7):
            for l in range(200):
                if(random.randrange(100)<perc_oot and signalinput[j][200*k+l] == 0 and l < MaxEle[k]):
                   signalinput[j][200*k+l]=1
                   tdc_signalinput[j][200*k+l]= TDC_OOT_DC1
                   drift_signalinput[j][200*k+l]= Drift_OOT_DC1 

        if(k>12):
            for l in range(134):
                if(random.randrange(100)<perc_oot and signalinput[j][200*k+l] == 0 and l < MaxEle[k]):
                   signalinput[j][200*k+l]=1
                   tdc_signalinput[j][200*k+l]= TDC_OOT_DC2        
                   drift_signalinput[j][200*k+l]= Drift_OOT_DC2        


#Add random noise to detectors
for j in range (len(signalinput)):
    if(j%100==0):print(j,end="\r")
    for k in range(31):
        if(k<7):
            for l in range(200):
                if(random.randrange(100)==1 and l < MaxEle[k]):signalinput[j][200*k+l]=1
        if(k>12):
            for l in range(134):
                if(random.randrange(100)==1 and l < MaxEle[k]):signalinput[j][200*k+l]=1



#Combine the data to make a single array that can be split into training.
data=np.concatenate((signalinput, tracks,dimuons,signals),axis=-1)
np.save("generated_events_april13.npy",data)
