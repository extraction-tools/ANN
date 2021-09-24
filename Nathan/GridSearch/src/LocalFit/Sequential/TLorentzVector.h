#ifndef TLORENTZVECTOR_H_
#define TLORENTZVECTOR_H_



typedef struct {
    double x, y, z, t;
} TLorentzVector;



TLorentzVector TLorentzVector_Init(double _x, double _y, double _z, double _e);

double TLorentzVector_Px(const TLorentzVector * const __restrict__ self);
double TLorentzVector_Py(const TLorentzVector * const __restrict__ self);
double TLorentzVector_Pz(const TLorentzVector * const __restrict__ self);
double TLorentzVector_E(const TLorentzVector * const __restrict__ self);

double TLorentzVector_paren_op(const TLorentzVector self, const int i);

TLorentzVector TLorentzVector_plus(const TLorentzVector self, const TLorentzVector other);
TLorentzVector TLorentzVector_minus(const TLorentzVector self, const TLorentzVector other);
double TLorentzVector_mul(const TLorentzVector self, const TLorentzVector other);



TLorentzVector TLorentzVector_Init(double _x, double _y, double _z, double _e) {
    TLorentzVector newLV;

    newLV.x = _x;
    newLV.y = _y;
    newLV.z = _z;
    newLV.t = _e;

    return newLV;
}



double TLorentzVector_Px(const TLorentzVector * const __restrict__ self) {
    return self->x;
}

double TLorentzVector_Py(const TLorentzVector * const __restrict__ self) {
    return self->y;
}

double TLorentzVector_Pz(const TLorentzVector * const __restrict__ self) {
    return self->z;
}

double TLorentzVector_E(const TLorentzVector * const __restrict__ self) {
    return self->t;
}



double TLorentzVector_paren_op(const TLorentzVector self, const int i) {
    switch (i) {
    case 0:
        return self.x;
    case 1:
        return self.y;
    case 2:
        return self.z;
    case 3:
        return self.t;
    default:
        exit(1);
    }
}



TLorentzVector TLorentzVector_plus(const TLorentzVector self, const TLorentzVector other) {
    TLorentzVector newLV;

    newLV.x = self.x + other.x;
    newLV.y = self.y + other.y;
    newLV.z = self.z + other.z;
    newLV.t = self.t + other.t;

    return newLV;
}

TLorentzVector TLorentzVector_minus(const TLorentzVector self, const TLorentzVector other) {
    TLorentzVector newLV;

    newLV.x = self.x - other.x;
    newLV.y = self.y - other.y;
    newLV.z = self.z - other.z;
    newLV.t = self.t - other.t;

    return newLV;
}

double TLorentzVector_mul(const TLorentzVector self, const TLorentzVector other) {
    const double newX = self.x * other.x;
    const double newY = self.y * other.y;
    const double newZ = self.z * other.z;
    const double newT = self.t * other.t;

    return newT - newZ - newY - newX;
}



#endif // TLORENTZVECTOR_H_
