#!/usr/bin/env python3
"""MAIN: independent ideal-gas Riemann and isentropic reference relations."""
import math
import numpy as np

RR = 6.0221417930e23 * 1.38065e-23 * 1000  # Installed SPUMA physicalConstants
AIR_R = RR / 28.965
GAMMA = 1.4

def gas_state(p, T, gas='air'):
    """Independent scalar PR cubic root plus analytic single-phase properties."""
    if gas=='air':
        rho=p/(AIR_R*T)
        return dict(rho=rho,R=AIR_R,e=2.5*AIR_R*T,a=math.sqrt(GAMMA*AIR_R*T),
                    pr=AIR_R*T,pt=rho*AIR_R,cv=2.5*AIR_R)
    R=RR/44.0128;tc=309.52067823146;pc=7244816.7016072;omega=.1613;cp0=881.80380464488
    b=.07780*R*tc/pc;a0=.45724*(R*tc)**2/pc;k=.37464+1.54226*omega-.26992*omega**2
    q=1+k*(1-math.sqrt(T/tc));a=a0*q*q;at=-a0*k*q/math.sqrt(T*tc);att=a0*k*(1+k)/(2*math.sqrt(T**3*tc))
    A=a*p/(R*T)**2;B=b*p/(R*T)
    roots=np.roots([1,B-1,A-2*B-3*B*B,-A*B+B*B+B**3])
    z=max(float(v.real) for v in roots if abs(v.imag)<1e-9)
    rho=p/(z*R*T);br=b*rho;d=1+2*br-br*br
    pr=R*T/(1-br)**2-2*a*rho*(1+br)/d**2;pt=rho*R/(1-br)-at*rho*rho/d
    L=math.log((1+(1+math.sqrt(2))*br)/(1+(1-math.sqrt(2))*br))
    cv=cp0-R+T*att*L/(2*math.sqrt(2)*b)
    return dict(rho=rho,R=R,e=(cp0-R)*T+(T*at-a)*L/(2*math.sqrt(2)*b),
                a=math.sqrt(pr+T*pt*pt/(rho*rho*cv)),pr=pr,pt=pt,cv=cv)

def acoustic_state(x,time=0,mean_mach=0,gas='air'):
    base=gas_state(2e6,300,gas);a=base['a'];r0=base['rho'];u0=mean_mach*a
    dp=200*np.sin(2*np.pi*(np.asarray(x)-(a+u0)*time))
    rho=r0+dp/a**2
    temp=300+(dp-base['pr']*dp/a**2)/base['pt']
    if gas=='air':temp=(2e6+dp)/(rho*AIR_R)
    return dict(p=2e6+dp,rho=rho,T=temp,U=u0+dp/(r0*a)),base


def normal_shock(mach, gamma=GAMMA):
    density = (gamma+1)*mach*mach/((gamma-1)*mach*mach+2)
    pressure = 1+2*gamma/(gamma+1)*(mach*mach-1)
    return {'density_ratio': density, 'pressure_ratio': pressure,
            'temperature_ratio': pressure/density,
            'downstream_mach': math.sqrt((1+(gamma-1)*mach*mach/2)/(gamma*mach*mach-(gamma-1)/2))}


def riemann(x, time, left=(1., 0., 1e5), right=(.125, 0., 1e4), discontinuity=.5, gamma=GAMMA):
    """Exact Euler solution, primitive states (rho,u,p); ideal gas, no vacuum."""
    def wave(p, state):
        rho, u, pk = state; a = math.sqrt(gamma*pk/rho)
        if p > pk:
            return (p-pk)*math.sqrt(2/((gamma+1)*rho)/(p+(gamma-1)/(gamma+1)*pk))
        return 2*a/(gamma-1)*((p/pk)**((gamma-1)/(2*gamma))-1)
    lo, hi = 1e-12, max(left[2], right[2])
    residual = lambda p: wave(p,left)+wave(p,right)+right[1]-left[1]
    while residual(hi)<0: hi*=2
    for _ in range(100):
        mid=(lo+hi)/2
        if residual(mid)>0: hi=mid
        else: lo=mid
    ps=(lo+hi)/2; us=(left[1]+right[1]+wave(ps,right)-wave(ps,left))/2
    rho=np.empty_like(np.asarray(x,dtype=float));u=rho.copy();p=rho.copy()
    for i, xi in enumerate(x):
        if time==0:
            rho[i],u[i],p[i]=left if xi<discontinuity else right;continue
        s=(xi-discontinuity)/time
        is_left=s<=us; state=left if is_left else right
        rk,uk,pk=state; ak=math.sqrt(gamma*pk/rk); direction=-1 if is_left else 1
        if ps>pk:
            speed=uk+direction*ak*math.sqrt((gamma+1)/(2*gamma)*ps/pk+(gamma-1)/(2*gamma))
            outside=s<=speed if is_left else s>=speed
            star=rk*(ps/pk+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*ps/pk+1)
            rho[i],u[i],p[i]=state if outside else (star,us,ps)
        else:
            astar=ak*(ps/pk)**((gamma-1)/(2*gamma))
            head=uk+direction*ak;tail=us+direction*astar
            outside=s<=head if is_left else s>=head
            inside=s>=tail if is_left else s<=tail
            if outside: rho[i],u[i],p[i]=state
            elif inside: rho[i],u[i],p[i]=rk*(ps/pk)**(1/gamma),us,ps
            else:
                velocity=2/(gamma+1)*(-direction*ak+(gamma-1)*uk/2+s)
                sound=2/(gamma+1)*(ak+direction*(gamma-1)*(s-uk)/2)
                rho[i],u[i],p[i]=rk*(sound/ak)**(2/(gamma-1)),velocity,pk*(sound/ak)**(2*gamma/(gamma-1))
    return {'rho':rho,'U':u,'p':p,'T':p/(rho*AIR_R),'p_star':ps,'u_star':us}


def area_mach(area_ratio, supersonic, gamma=GAMMA):
    if abs(area_ratio-1)<1e-14:return 1.
    def ratio(m):return ((2/(gamma+1))*(1+(gamma-1)*m*m/2))**((gamma+1)/(2*(gamma-1)))/m
    lo,hi=(1.,10.) if supersonic else (1e-6,1.)
    for _ in range(100):
        m=(lo+hi)/2
        if (ratio(m)<area_ratio)==supersonic:lo=m
        else:hi=m
    return (lo+hi)/2
