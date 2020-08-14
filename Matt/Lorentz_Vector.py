


class LorentzVector(object):

    def __init__(self, x_ = 0.0, y_ = 0.0, z_=0.0, t_=0.0):
        self.x = x_
        self.y = y_
        self.z = z_
        self.t = t_

    def __getitem__(self, key):
        if key==0:
            return self.x
        elif key==1:
            return self.y
        elif key==2:
            return self.z
        elif key==3:
            return self.t
        else:
            return None
    
    def __add__(self, other):
        return LorentzVector(x_=self.x+other.Px(), y_=self.y+other.Py(), z_=self.z+other.Pz(), t_=self.t+other.Pt())

    def __sub__(self, other):
        return LorentzVector(x_=self.x-other.Px(), y_=self.y-other.Py(), z_=self.z-other.Pz(), t_=self.t-other.Pt())
    
    def __mul__(self, other):
        return self.dot(other)
    
    def __str__(self):
        return '{0}, {1}, {2}, {3}'.format(self.x, self.y, self.z, self.t) 

    def Vect(self):
        return [self.x, self.y, self.z]

    def Px(self):
        return self.x
    
    def Py(self):
        return self.y

    def Pz(self):
        return self.z
    
    def Pt(self):
        return self.t

    def dot(self, q):
        return (self.t*q.Pt() - self.z*q.Pz() - self.y*q.Py() - self.x*q.Px())

    def SetPxPyPzE(self, x, y ,z, e):
        self.x = x
        self.y = y
        self.z = z
        self.t = e