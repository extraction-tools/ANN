#ifndef LORENTZ_VECTOR_H_
#define LORENTZ_VECTOR_H_

typedef struct {
    double x, y, z, t;
} LorentzVector;

LorentzVector LorentzVector_add(LorentzVector self, LorentzVector other);
LorentzVector LorentzVector_sub(LorentzVector self, LorentzVector other);
double LorentzVector_mul(LorentzVector self, LorentzVector other);
double LorentzVector_dot(LorentzVector self, LorentzVector other);
LorentzVector LorentzVector_SetPxPyPzE(double x, double y, double z, double e);

LorentzVector LorentzVector_add(LorentzVector self, LorentzVector other) {
    LorentzVector newLV;

    newLV.x = self.x + other.x;
    newLV.y = self.y + other.y;
    newLV.z = self.z + other.z;
    newLV.t = self.t + other.t;

    return newLV;
}

LorentzVector LorentzVector_sub(LorentzVector self, LorentzVector other) {
    LorentzVector newLV;

    newLV.x = self.x - other.x;
    newLV.y = self.y - other.y;
    newLV.z = self.z - other.z;
    newLV.t = self.t - other.t;

    return newLV;
}

double LorentzVector_mul(LorentzVector self, LorentzVector other) {
    return LorentzVector_dot(self, other);
}

double LorentzVector_dot(LorentzVector self, LorentzVector other) {
    double x = self.x * other.x;
    double y = self.y * other.y;
    double z = self.z * other.z;
    double t = self.t * other.t;

    return t - z - y - x;
}

LorentzVector LorentzVector_SetPxPyPzE(double x, double y, double z, double e) {
    LorentzVector lv;
    
    lv.x = x;
    lv.y = y;
    lv.z = z;
    lv.t = e;

    return lv;
}

#endif // LORENTZ_VECTOR_H_
