#ifndef TLORENTZVECTOR_H_
#define TLORENTZVECTOR_H_

#include <stdlib.h>

typedef struct {
    double x, y, z, t;
} TLorentzVector;

double TLorentzVector_Parentheses_Operator(const TLorentzVector self, const int i);
double TLorentzVector_Px(const TLorentzVector * const restrict self);
double TLorentzVector_Py(const TLorentzVector * const restrict self);
double TLorentzVector_Pz(const TLorentzVector * const restrict self);
double TLorentzVector_E(const TLorentzVector * const restrict self);
TLorentzVector TLorentzVector_Plus_Operator(const TLorentzVector self, const TLorentzVector other);
TLorentzVector TLorentzVector_Minus_Operator(const TLorentzVector self, const TLorentzVector other);
double TLorentzVector_Multiplication_Operator(const TLorentzVector self, const TLorentzVector other);
void TLorentzVector_SetPxPyPzE(TLorentzVector * const restrict self, double _x, double _y, double _z, double _e);



double TLorentzVector_Parentheses_Operator(const TLorentzVector self, const int i) {
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

double TLorentzVector_Px(const TLorentzVector * const restrict self) {
    return self->x;
}

double TLorentzVector_Py(const TLorentzVector * const restrict self) {
    return self->y;
}

double TLorentzVector_Pz(const TLorentzVector * const restrict self) {
    return self->z;
}

double TLorentzVector_E(const TLorentzVector * const restrict self) {
    return self->t;
}

TLorentzVector TLorentzVector_Plus_Operator(const TLorentzVector self, const TLorentzVector other) {
    TLorentzVector newLV;

    newLV.x = self.x + other.x;
    newLV.y = self.y + other.y;
    newLV.z = self.z + other.z;
    newLV.t = self.t + other.t;

    return newLV;
}

TLorentzVector TLorentzVector_Minus_Operator(const TLorentzVector self, const TLorentzVector other) {
    TLorentzVector newLV;

    newLV.x = self.x - other.x;
    newLV.y = self.y - other.y;
    newLV.z = self.z - other.z;
    newLV.t = self.t - other.t;

    return newLV;
}

double TLorentzVector_Multiplication_Operator(const TLorentzVector self, const TLorentzVector other) {
    const double newX = self.x * other.x;
    const double newY = self.y * other.y;
    const double newZ = self.z * other.z;
    const double newT = self.t * other.t;

    return newT - newZ - newY - newX;
}

void TLorentzVector_SetPxPyPzE(TLorentzVector * const restrict self, double _x, double _y, double _z, double _e) {
    self->x = _x;
    self->y = _y;
    self->z = _z;
    self->t = _e;
}

#endif // TLORENTZVECTOR_H_
