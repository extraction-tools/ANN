import math
import numpy as np
from mosttcoefficient import coefficients as coe
import matplotlib.pyplot as plt

class coefficients(object):
    #basically some shorthand that is used a lott
    def xi(M,QQ,xB,alpha,beta):
        Q=np.sqrt(QQ)
        aux0=(2.*(alpha*((M**2)*(xB**2))))+(t*((-1.+xB)*(1.+((-2.*alpha)+((-1.\
        +beta)*xB)))));
        output=(xB/(2.-xB))+((-2.*((Q**-2.)*(((-2.+xB)**-2.)*(xB*aux0))))/(1.+\
        ((-1.+beta)*xB)))
        return output
    def Gamma(xB,M,QQ):
        return 2*xB*M/np.sqrt(QQ)
    def N(M,QQ,xB,alpha,beta):
        ee=xi(M,QQ,xB,alpha,beta)
        return np.sqrt(-4*M**2*ee**2-t*(1-ee**2))/M
    #the F fucntions which do not have the interference 
    def FUU(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        ee=xi(M,QQ,xB,alpha,beta)
        hu=coe.hampc(t,y,M,QQ,xB,phi)
        htu=coe.htampc(t,y,M,QQ,xB,phi)
        out=4*((1-ee**2)*(hu*complex(H,-Hc)*complex(H,Hc)+htu*complex(Ht,-Htc)*complex(Ht,Htc))
               -t/(4*M**2)*(hu*complex(E,-Ec)*complex(E,Ec)+ee**2*htu*complex(Et,-Etc)*complex(Et,Etc))
               -ee**2*(hu*complex(E,Ec)*complex(E,-Ec)+
                       hu*(complex(E,-Ec)*complex(H,Hc)+complex(H,-Hc)*complex(E,Ec))+
                       htu*(complex(Et,-Etc)*complex(Ht,Htc)+complex(Ht,-Htc)*complex(Et,Etc))))
        return out

    def Futout(t,y,M,QQ,xB,phi,alpha,beta,E,Et,Hc,Htc):
        n=N(M,QQ,xB,alpha,beta)
        hu=coe.hampc(t,y,M,QQ,xB,phi)
        ee=xi(M,QQ,xB,alpha,beta)
        htu=coe.htampc(t,y,M,QQ,xB,phi)
        out =n*4*(hu*complex(H,-Hc)*complex(E,Ec)-htu*ee*complex(Ht,-Htc)*complex(Et+Etc)).imag
        return out

    def FuL(): 
        return 0

    def FLL(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        ee=xi(M,QQ,xB,alpha,beta)
        hmi=coe.hminusampc(t,y,M,QQ,xB,phi)
        out=8*hmi*((1-ee**2)*complex(Ht,-Htc)*complex(H,Hc)-
                   ee**2*(complex(Ht,-Htc)*complex(E,Ec)+complex(Et,-Etc)*complex(H,Hc))-
                  (ee**2/(1+ee)+t/(4*M**2))*ee*complex(Et,-Etc)*complex(E,Ec)).real
        return out

    def FUTout():
        return 0
    def FUTin():
        return 0
    def FLTin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        n=N(M,QQ,xB,alpha,beta)
        hmi=coe.hminusampc(t,y,M,QQ,xB,phi)
        ee=xi(M,QQ,xB,alpha,beta)
        out=4*n*hmi*(complex(Ht,-Htc)*complex(E,Ec)-ee*complex(Et,-Etc)*complex(H,Hc)-
                    ee**2/(1+ee**2)*complex(Et,-Etc)*complex(E,Ec)).real
        return out
    #the Tdvcs with only Fu
    def amp1(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        fuu=FUU(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)
        finest=1/137
        gamma=2*xB*M/np.sqrt(QQ)
        return (finest**3*xB*y**2/(8*np.pi*QQ**2*np.sqrt(1+gamma**2))*fuu/QQ**2*389.39*1000).real
    #Full TDvs 
    def Tdvcs(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc,deltt,deltl,phis,h):
        #note deltt,deltl,h  all have to do with polerization,
        #phis is another angle to deal with polization should be varried separtyl
        fuu=FUU(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)
        ful=FuL()
        futin=FUTin()
        flu=0
        fll=FLL(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)
        futout=Futout(t,y,M,QQ,xB,phi,alpha,beta,E,Et,Hc,Htc)
        fltin=FLTin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)
        fltout=0
        output=1/QQ**2*(fuu+2*deltl*ful+2*deltt*(np.cos(phis-phi)*futin+np.sin(phis-phi)*futout)
                +2*h*(flu+2*deltl*fll+2*deltt*(np.cos(phis-phi)*fltin+np.sin(phis-phi)*fltout))).real
        return output
    #Cross section 
    def crosssection(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc,deltt,deltl,phis,h):
        finest=1/137
        gamma=Gamma(xB,M,QQ)
        tdvcs=Tdvcs(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc,deltt,deltl,phis,h)
        out=(finest**3*xB*y**2/(16*np.pi**2*QQ**2*np.sqrt(1+gamma**2))*tdvcs *389.39*1000).real
        return out