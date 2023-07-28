import math
import numpy as np
from mosttcoefficient import coefficients as coe
class function(object):
    #general functions
    #defined in mathematica
    def xi(M,QQ,xB,alpha,beta):
        Q=np.sqrt(QQ)
        aux0=(2.*(alpha*((M**2)*(xB**2))))+(t*((-1.+xB)*(1.+((-2.*alpha)+((-1.\
        +beta)*xB)))));
        output=(xB/(2.-xB))+((-2.*((Q**-2.)*(((-2.+xB)**-2.)*(xB*aux0))))/(1.+\
        ((-1.+beta)*xB)))
        return output
    #equation 2.4
    def Gamma(xB,M,QQ):
        return 2*xB*M/np.sqrt(QQ)
    #equation 3.25
    def N(M,QQ,xB,alpha,beta):
        ee=xi(M,QQ,xB,alpha,beta)
        return np.sqrt(-4*M**2*ee**2-t*(1-ee**2))/M
    def F1(t,M):
        GE=1/(1-t/.71)**2
        GM=2.79*GE
        out =(GE-t/(4*M**2)*GM)/(1-t/(4*M**2))
        return out
    def F2(t,M):
        GE=1/(1-t/.71)**2
        GM=2.79*GE
        out =(-GE+GM)/(1-t/(4*M**2))
        return out
    #pure DVCS terms, twist 2 
    #B.1
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

    #B.2
    def Futout(t,y,M,QQ,xB,phi,alpha,beta,E,Et,Hc,Htc):
        n=N(M,QQ,xB,alpha,beta)
        hu=coe.hampc(t,y,M,QQ,xB,phi)
        ee=xi(M,QQ,xB,alpha,beta)
        htu=coe.htampc(t,y,M,QQ,xB,phi)
        out =n*4*(hu*complex(H,-Hc)*complex(E,Ec)-htu*ee*complex(Ht,-Htc)*complex(Et+Etc)).imag
        return out

    #B.3
    def FUL(): 
        return 0

    #B.4
    def FLL(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        ee=xi(M,QQ,xB,alpha,beta)
        hmi=coe.hminusampc(t,y,M,QQ,xB,phi)
        out=8*hmi*((1-ee**2)*complex(Ht,-Htc)*complex(H,Hc)-
                   ee**2*(complex(Ht,-Htc)*complex(E,Ec)+complex(Et,-Etc)*complex(H,Hc))-
                  (ee**2/(1+ee)+t/(4*M**2))*ee*complex(Et,-Etc)*complex(E,Ec)).real
        return out

    #B.5
    def FUTin():
        return 0
    #B.6 
    def FLTin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        n=N(M,QQ,xB,alpha,beta)
        hmi=coe.hminusampc(t,y,M,QQ,xB,phi)
        ee=xi(M,QQ,xB,alpha,beta)
        out=4*n*hmi*(complex(Ht,-Htc)*complex(E,Ec)-ee*complex(Et,-Etc)*complex(H,Hc)-
                    ee**2/(1+ee**2)*complex(Et,-Etc)*complex(E,Ec)).real
        return out
    #interference terms twist 2 
    #c.1
    def FUUI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        Q=np.sqrt(QQ)
        AIU=coe.AINTunpc(t,y,M,QQ,xB,phi)
        BIU=coe.BINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        CIU= coe.CINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        out=(AIU*(F1(t,M)*complex(H,-Hc)-t/(4*M**2)*F2(t,M)*complex(E,-Ec))
             +BIU*(F1(t,M)+F2(t,M))*(complex(H,-Hc)+complex(E,-Ec))+
             CIU*(F1(t,M)+F2(t,M))*complex(Ht,-Htc)).real
        return out
    #c.2
    def FLUI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        Q=np.sqrt(QQ)
        AIL=coe.AINTpolc(t,y,M,QQ,xB,phi)
        BIL=coe.BINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        CIL= coe.CINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        out=-(AIL*(F1(t,M)*complex(H,-Hc)-t/(4*M**2)*F2(t,M)*complex(E,-Ec))
             +BIL*(F1(t,M)+F2(t,M))*(complex(H,-Hc)+complex(E,-Ec))+
             CIL*(F1(t,M)+F2(t,M))*complex(Ht,-Htc)).imag
        return out
    #c.3 
    def FULI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        Q=np.sqrt(QQ)
        ee=xi(M,QQ,xB,alpha,beta)
        AIUt=coe.AtINTunpc(t,y,M,QQ,xB,phi)
        BIUt=coe.BtINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        CIUt= coe.CtINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        out= -(AIUt*(F1(t,M)*(complex(Ht,-Htc)-ee**2/(1+ee)*complex(Et,-Etc))-F2(t,M)*t*ee/(4*M**2)*complex(Et,-Etc))+
              BIUt*(F1(t,M)+F2(t,M))*(complex(Ht,-Htc)+ee/(1+ee)*complex(Et,-Etc))-
              CIUt*(F1(t,M)+F2(t,M))*(complex(H,-Hc)+ee/(1+ee)*complex(E,-Ec)) ).imag
        return out

    #c.4
    def FLLI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        ee=xi(M,QQ,xB,alpha,beta)
        AILt=coe.AtINTpolc(t,y,M,QQ,xB,phi)
        BILt=coe.BtINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        CILt= coe.CtINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        out= -(AILt*(F1(t,M)*(complex(Ht,-Htc)-ee**2/(1+ee)*complex(Et,-Etc))-F2(t,M)*t*ee/(4*M**2)*complex(Et,-Etc))+
                  BILt*(F1(t,M)+F2(t,M))*(complex(Ht,-Htc)+ee/(1+ee)*complex(Et,-Etc))-
                   CILt*(F1(t,M)+F2(t,M))*(complex(H,-Hc)+ee/(1+ee)*complex(E,-Ec)) ).real
        return out

    #C.5
    def FUTIin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc): 
        n=N(M,QQ,xB,alpha,beta)
        AIUt=coe.AtINTunpc(t,y,M,QQ,xB,phi)
        BIUt=coe.BtINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        CIUt= coe.CtINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        ee=xi(M,QQ,xB,alpha,beta)

        out=-2/n*(AIUt*(ee*F1(t,M)*(ee*complex(Ht,-Htc)+(ee**2/(1+ee)+t/(4*M**2))*complex(Et,-Etc))
                       +F2(t,M)*t/(4*M**2)*((ee**2-1)*complex(Ht,-Htc)+ee**2*complex(Et,-Etc)))+
                BIUt*(F1(t,M)+F2(t,M))*(complex(Ht,-Htc)+(t/(4*M**2)-ee/(1+ee))*ee*complex(Et,-Etc))+
                CIUt*(F1(t,M)+F2(t,M))*(ee*complex(H,-Hc)+(ee**2/(1+ee)+t/(4*M**2))*complex(E,-Ec))).imag
        return out

    #c.6
    def FLTIin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        n=N(M,QQ,xB,alpha,beta)
        AILt=coe.AtINTpolc(t,y,M,QQ,xB,phi)
        BILt=coe.BtINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        CILt= coe.CtINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        ee=xi(M,QQ,xB,alpha,beta)

        out=-2/n*(AILt*(ee*F1(t,M)*(ee*complex(Ht,-Htc)+(ee**2/(1+ee)+t/(4*M**2))*complex(Et,-Etc))
                       +F2(t,M)*t/(4*M**2)*((ee**2-1)*complex(Ht,-Htc)+ee**2*complex(Et,-Etc)))+
                BILt*(F1(t,M)+F2(t,M))*(complex(Ht,-Htc)+(t/(4*M**2)-ee/(1+ee))*ee*complex(Et,-Etc))+
                CILt*(F1(t,M)+F2(t,M))*(ee*complex(H,-Hc)+(ee**2/(1+ee)+t/(4*M**2))*complex(E,-Ec))).real
        return out

    #c.7
    def FUTIout(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        Q=np.sqrt(QQ)
        AIU=coe.AINTunpc(t,y,M,QQ,xB,phi)
        BIU=coe.BINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        CIU= coe.CINTunpc(t,y,M,QQ,xB,phi,alpha,beta)
        n=N(M,QQ,xB,alpha,beta)
        ee=xi(M,QQ,xB,alpha,beta)
        out =2/n*(AIU*(F1(t,M)*(ee**2*complex(H,-Hc)+(ee**2+t/(4*M**2))*complex(E,-Ec))+
                      t/(4*M**2)*F2(t,M)*((ee**2-1)*complex(H,-Hc)+ee**2*complex(E,-Ec)))+
                 BIU*(F1(t,M)+F2(t,M))*(complex(H,-Hc)+t/(4*M**2)*complex(E,-Ec))-
                 CIU*ee*(F1(t,M)+F2(t,M))*(complex(Ht,-Htc)+t/(4*M**2)*complex(Et,-Etc))).imag
        return out

    #c.8
    def FLTIout(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        Q=np.sqrt(QQ)
        AIL=coe.AINTpolc(t,y,M,QQ,xB,phi)
        BIL=coe.BINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        CIL= coe.CINTpolc(t,y,M,QQ,xB,phi,alpha,beta)
        n=N(M,QQ,xB,alpha,beta)
        ee=xi(M,QQ,xB,alpha,beta)
        out =2/n*(AIL*(F1(t,M)*(ee**2*complex(H,-Hc)+(ee**2+t/(4*M**2))*complex(E,-Ec))+
                      t/(4*M**2)*F2(t,M)*((ee**2-1)*complex(H,-Hc)+ee**2*complex(E,-Ec)))+
                 BIL*(F1(t,M)+F2(t,M))*(complex(H,-Hc)+t/(4*M**2)*complex(E,-Ec))-
                 CIL*ee*(F1(t,M)+F2(t,M))*(complex(Ht,-Htc)+t/(4*M**2)*complex(Et,-Etc))).real
        return out
    #Cross section terms
    #term out front that is multiplied by |T| to get cross section equation 2.3
    def consts(y,M,QQ,xB):
        fnst=1/137
        return fnst**3*xB*y**2/(8*np.pi*QQ**2*np.sqrt(1+Gamma(xB,M,QQ)**2))
    #Cross sections from equation 3.40 for pure DVCS, broken up into its varius F components, if more than one just add together 
    def sigmaFUU(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return 1/QQ**2*consts(y,M,QQ,xB)*FUU(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000
    #note that FUL=0 as it has h+ in it. 
    def sigmaFUL():
        return 0
    #note that FUTin=0 as it has h+ in it. 
    def sigmaFUTin():
        return 0
    #have extra 2pi in denominator becasue phis is being varried here to must put back inn
    def sigmaFUTout(t,y,M,QQ,xB,phi,alpha,beta,DeltaT,phis,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return (1/(QQ**2*2*np.pi)*consts(y,M,QQ,xB)*np.sin(phis-phi)*
                2*DeltaT*Futout(t,y,M,QQ,xB,phi,alpha,beta,E,Et,Hc,Htc))*389.39*1000
    #this is zero as FLU vanish and does not even appear with other f functions
    def sigmaFLU():
        return 0
    def sigmaFLL(t,y,M,QQ,xB,phi,alpha,beta,DeltaL,h,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return 1/QQ**2*consts(y,M,QQ,xB)*2*h*2*DeltaL*FLL(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000
    #d
    def sigmaFLTin(t,y,M,QQ,xB,phi,alpha,beta,DeltaT,phis,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return (1/(QQ**2*2*np.pi)*consts(y,M,QQ,xB)*2*h*2*DeltaT*np.cos(phis-phi)*
                FLTin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc))*389.39*1000
    #this is zero as FLU vanish and does not even appear with other f functionss
    def sigmaFLTout():
        return 0
    #Cross secton from equation 3.57 for interference, broken up into its varius F components, if more than one jsut add together. 
    def sigmaFUUI(t,y,M,QQ,xB,phi,alpha,beta,el,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        Q=np.sqrt(QQ)
        return -el/(QQ*t)*consts(y,M,QQ,xB)*FUUI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000
    def sigmaFULI(t,y,M,QQ,xB,phi,alpha,beta,el,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return -el/(QQ*t)*consts(y,M,QQ,xB)*2*DeltaL*FULI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000
    def sigmaFUTIin(t,y,M,QQ,xB,phi,alpha,beta,el,DeltaT,phis,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return (-el/(QQ*t*2*np.pi)*consts(y,M,QQ,xB)*2*DeltaT*np.cos(phis-phi)*
                FUTIin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc))*389.39*1000
    def sigmaFUTIout(t,y,M,QQ,xB,phi,alpha,beta,el,DeltaT,phis,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return (-el/(QQ*t*2*np.pi)*consts(y,M,QQ,xB)*2*DeltaT*np.sin(phis-phi)*
                FUTIout(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc))*389.39*1000
    def sigmaFLUI(t,y,M,QQ,xB,phi,alpha,beta,el,h,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return -el/(QQ*t)*consts(y,M,QQ,xB)*2*h*FLUI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000
    def sigmaFLLI(t,y,M,QQ,xB,phi,alpha,beta,el,h,DeltaL,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return -el/(QQ*t)*consts(y,M,QQ,xB)*2*h*2*DeltaL*FLUI(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000
    def sigmaFLTIin(t,y,M,QQ,xB,phi,alpha,beta,el,h,DeltaT,phis,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return (-el/(QQ*t)*consts(y,M,QQ,xB)*2*h*2*DeltaT*np.cos(phis-phi)*
                FLTIin(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000)
    def sigmaFLTIout(t,y,M,QQ,xB,phi,alpha,beta,el,h,DeltaT,phis,H,E,Ht,Et,Hc,Ec,Htc,Etc):
        return (-el/(QQ*t)*consts(y,M,QQ,xB)*2*h*2*DeltaT*np.sin(phis-phi)*
                FLTIout(t,y,M,QQ,xB,phi,alpha,beta,H,E,Ht,Et,Hc,Ec,Htc,Etc)*389.39*1000)

