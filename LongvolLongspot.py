# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:28:18 2021

@author: Toan
"""

# %% Importing Libraries and Data
from cmath import nan, sqrt
from tracemalloc import stop
from turtle import title
import pandas as pd
import numpy as np
import datetime
# import statsmodels.api as sm
# from statsmodels.regression.rolling import RollingOLS
from scipy.stats import norm
import hvplot.pandas


# %% Read model data
df = pd.read_csv("C:/Users/hannah.tudo/Desktop/Toan/Projects/LongVolLongSpot/longvollongspot_model_data.csv")

xdates = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['date'],'D')
df['datetime'] = xdates
df = df.set_index('datetime')
df = df.drop(['date'], axis=1)



# %% Calculate daily pnl for long equity and long gamma performance

dow = ['mon', 'tue', 'wed', 'thu', 'fri']

for i in range(0,len(dow)):
    varstr = 'putdailypnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'optiondailypnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'deltadailypnl_' + dow[i]
    df[varstr] = np.nan     
    varstr = 'optionstrike_' + dow[i]
    df[varstr] = np.nan     
    varstr = 'voldailypnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'volpnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'equitypnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'equitydailypnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'rvivgammaadj_' + dow[i]
    df[varstr] = np.nan
    varstr = 'rviv_' + dow[i]
    df[varstr] = np.nan
    varstr = 'gamma_' + dow[i]
    df[varstr] = np.nan
    varstr = 'gammatotal_' + dow[i]
    df[varstr] = np.nan
    varstr = 'weeklyeqret_' + dow[i]
    df[varstr] = np.nan
    
    varstr = 'intexp_' + dow[i]
    df[varstr] = np.nan
    varstr = 'cashfundneed_' + dow[i]
    df[varstr] = np.nan
    varstr = 'cashequityneed_' + dow[i]
    df[varstr] = np.nan

# day of week: Monday = 0, Tuesday = 1, Wed = 2, Thur = 3, Frid = 4

dtm = list(reversed(range(6)))
ttm = np.divide(dtm,365)
ttm[-1] = ttm[-1] + 0.00001    

for DofW in range(0,5):

    df_dow = df[df.index.dayofweek==DofW]   
    list_dof = df_dow.index.tolist()   
       
    for i in range(1,len(list_dof)):
    
        bdate = list_dof[i-1]
        edate = list_dof[i]
        pxclose = df.spx_close[bdate:edate]
        
        if len(pxclose) == 6:
            
            # Long weekly 25 delta put and delta hedge daily.
            datarng = df.spx_divy[bdate:edate]
            divy = np.multiply(datarng,0.01)
            datarng = df.ivol25dput[bdate:edate]
            ivol = np.multiply(datarng,0.01)
            datarng = df.rfr_1m[bdate:edate]
            rfr = np.multiply(datarng,0.01)        
            
            equitymv = 10000000
            delta = -0.25
            # 0.25 equity shares is applied because we need enough capital in case of puts because delta one.
            equityshares = 0.25*equitymv/pxclose[0]
            optionshares = equityshares * 1/abs(delta)
            
            # optionshares = 30000
            # equityshares = 0.25*optionshares
    
            # Calculate a option strike for a given delta
            strikestr = 'optionstrike_' + dow[DofW]
            df[strikestr][bdate] = strkgivendelta(pxclose[0],delta,ivol[0],rfr[0],ttm[0])
            optstr = 'optiondailypnl_' + dow[DofW]        
            optprc = bsprice(pxclose, df[strikestr][bdate], ivol, divy, rfr, ttm, "p")
            
            # Estimate Tcost
            if ivol[0] >= 0 and ivol[0] < 0.15:
                Tcost = 1.01
            elif ivol[0] >= 0.15 and ivol[0] < 0.25:
                Tcost = 1.0125
            elif ivol[0] >= 0.25 and ivol[0] < 0.35:                
                Tcost = 1.0175
            else:
                Tcost = 1.03                
                            
            optprc[0] = optprc[0] * Tcost
            df[optstr][bdate:edate] = optprc.diff()    
             
            optdelta = pd.Series
            dailydeltareb = pd.Series
            dailydeltarebrollsum = pd.Series
            dailymv = pd.Series
            avgcost = pd.Series
            deltapnl = pd.Series
            
            optdelta = bsdelta(pxclose, df[strikestr][bdate], ivol, divy, rfr, ttm, "p")
            
            dailydeltareb = -optdelta.diff()
            dailydeltareb[0] = -optdelta[0]
            dailydeltareb[-1] = dailydeltareb[-1] + 0.00000001
            
            dailydeltarebrollsum = np.cumsum(dailydeltareb)
           
            dailymv = np.multiply(pxclose, dailydeltareb)
            avgcost = np.divide(np.cumsum(dailymv), dailydeltarebrollsum)
            deltapnl = np.multiply(dailydeltarebrollsum, np.subtract(pxclose,avgcost))
            deltastr = 'deltadailypnl_' + dow[DofW]        
            df[deltastr][bdate:edate] = deltapnl.diff()
                
            putdailypnlstr = 'putdailypnl_' + dow[DofW] 
            voldailypnlstr = 'voldailypnl_' + dow[DofW] 
            volpnlstr = 'volpnl_' + dow[DofW] 
            equitypnlstr = 'equitypnl_' + dow[DofW] 
            equitydailypnlstr = 'equitydailypnl_' + dow[DofW] 
            weeklyeqretstr = 'weeklyeqret_' + dow[DofW] 
            
            df[putdailypnlstr][bdate:edate] = df[putdailypnlstr][bdate:edate].fillna(0) + \
                optionshares * (df[optstr][bdate:edate].fillna(0))
            
            df[voldailypnlstr][bdate:edate] = df[voldailypnlstr][bdate:edate].fillna(0) + \
                optionshares * (df[optstr][bdate:edate].fillna(0) + df[deltastr][bdate:edate].fillna(0))
                
            temploc = df.index.get_loc(bdate)
            tempdate = df.index[temploc+1]
            df[volpnlstr][edate] = optionshares * np.sum(df[voldailypnlstr][tempdate:edate].fillna(0))
        
            # long equity
            df[equitydailypnlstr][bdate:edate] = df[equitydailypnlstr][bdate:edate].fillna(0) + equityshares * pxclose.diff().fillna(0)
            
            df[equitypnlstr][edate] = equityshares * np.sum(pxclose.diff().fillna(0))
            
            df[weeklyeqretstr][edate] = np.sum(pxclose.diff().fillna(0))/pxclose[0]
            
            
            # Calculate gamma adjusted rv-iv
            gammstr = 'gamma_' + dow[DofW]
            gammtotalstr = 'gammatotal_' + dow[DofW]
            rvivstr = 'rvivgammaadj_' + dow[DofW]
            # Calculate rv-iv
            rvivnotadjstr = 'rviv_' + dow[DofW]
            
            df[gammstr][bdate:edate] = bsgamma(pxclose, df[strikestr][bdate], ivol, divy, rfr, ttm)
            df[gammtotalstr][edate] = np.sum(df[gammstr][bdate:edate])
            rvnetiv = np.subtract(np.square(np.diff(np.log(pxclose))), np.divide(np.square(ivol[0]),365))
            df[rvivstr][edate] = np.sum(np.multiply(df[gammstr][tempdate:edate],rvnetiv))
            df[rvivnotadjstr][edate] = np.sum(rvnetiv)
            
            intexpstr = 'intexp_' + dow[DofW]
            dailymv[dailymv<0]=0
            df[intexpstr][edate] = (optprc[0] * optionshares * 0.01 + max(np.sum(dailymv),0) * optionshares + equityshares * pxclose[0]) * 0.05/52
            cashstr = 'cashfundneed_' + dow[DofW]
            df[cashstr][bdate:edate] = df[cashstr][bdate:edate].fillna(0) + optprc[0] * optionshares + np.sum(dailymv) * optionshares + equityshares * pxclose[0]
            df[cashstr][edate] = 0
            cashstr = 'cashequityneed_' + dow[DofW]
            df[cashstr][bdate:edate] = df[cashstr][bdate:edate].fillna(0) + 0.25*equitymv
            df[cashstr][edate] = 0
            
            
        print(i)
                                                                 

# %% Calculate daily pnl for long equity and long gamma performance for 3 month tennor

varstr = 'putdailypnl'
df[varstr] = np.nan
varstr = 'optiondailypnl'
df[varstr] = np.nan
varstr = 'deltadailypnl'
df[varstr] = np.nan     
varstr = 'optionstrike'
df[varstr] = np.nan     
varstr = 'voldailypnl'
df[varstr] = np.nan
varstr = 'volpnl'
df[varstr] = np.nan
varstr = 'equitypnl'
df[varstr] = np.nan
varstr = 'equitydailypnl'
df[varstr] = np.nan
varstr = 'rviv'
df[varstr] = np.nan
varstr = 'gma'
df[varstr] = np.nan
varstr = 'gammatotal'
df[varstr] = np.nan
varstr = 'cashfundneed3m'
df[varstr] = np.nan
    
dtm = list(reversed(range(66)))
dtm[0] = 90
ttm = np.divide(dtm,365)
ttm[-1] = ttm[-1] + 0.00001    

list_dof = df.index.tolist()   
   
for i in range(66,len(list_dof)):

    bdate = list_dof[i-65]
    edate = list_dof[i]
    pxclose = df.spx_close[bdate:edate]
    
    if len(pxclose) == 66:
        
        # Long weekly 25 delta put and delta hedge daily.
        datarng = df.spx_divy[bdate:edate]
        divy = np.multiply(datarng,0.01)
        datarng = df.iv_90[bdate:edate]
        ivol90 = np.multiply(datarng,0.01)
        datarng = df.iv_95[bdate:edate]
        ivol95 = np.multiply(datarng,0.01)
        datarng = df.iv_97_5[bdate:edate]
        ivol97_5 = np.multiply(datarng,0.01)
        
        datarng = df.rfr_3m[bdate:edate]
        rfr = np.multiply(datarng,0.01)        
        
        optdelta90 = bsdelta(pxclose[0], 0.9*pxclose[0], ivol90[0], divy[0], rfr[0], ttm[0], "p")
        optdelta95 = bsdelta(pxclose[0], 0.95*pxclose[0], ivol95[0], divy[0], rfr[0], ttm[0], "p")
        optdelta97_5 = bsdelta(pxclose[0], 0.975*pxclose[0], ivol97_5[0], divy[0], rfr[0], ttm[0], "p")
        
        index_min = np.argmin(abs(np.subtract([optdelta90,optdelta95,optdelta97_5],0.25)))
        if index_min == 0:
            ivol = ivol90
            strike = 0.9 * pxclose[0]
        elif index_min == 1:
            ivol = ivol95
            strike = 0.95 * pxclose[0]
        else:
            ivol = ivol97_5
            strike = 0.975 * pxclose[0]
            
        optdelta = pd.Series
        dailydeltareb = pd.Series
        dailydeltarebrollsum = pd.Series
        dailymv = pd.Series
        avgcost = pd.Series
        deltapnl = pd.Series
        

        equitymv = np.sqrt(52/4) * 10000000
        
        optdelta = bsdelta(pxclose, strike, ivol, divy, rfr, ttm, "p")
        # 0.25 equity shares is applied because we need enough capital in case of puts because delta one.
        equityshares = abs(optdelta[0]) * equitymv/pxclose[0]
        optionshares = equityshares * 1/ 0.25
    
        # Calculate a option strike for a given delta
        strikestr = 'optionstrike'
        df[strikestr][bdate] = strike
        optstr = 'optiondailypnl'
        optprc = bsprice(pxclose, df[strikestr][bdate], ivol, divy, rfr, ttm, "p")
        
        # Estimate Tcost
        if ivol[0] >= 0 and ivol[0] < 0.15:
            Tcost = 1.01
        elif ivol[0] >= 0.15 and ivol[0] < 0.25:
            Tcost = 1.0125
        elif ivol[0] >= 0.25 and ivol[0] < 0.35:                
            Tcost = 1.0175
        else:
            Tcost = 1.03                
                        
        optprc[0] = optprc[0] * Tcost
        df[optstr][bdate:edate] = optprc.diff()         
        
        
        dailydeltareb = -optdelta.diff()
        dailydeltareb[0] = -optdelta[0]
        dailydeltareb[-1] = dailydeltareb[-1] + 0.00000001
        
        dailydeltarebrollsum = np.cumsum(dailydeltareb)
       
        dailymv = np.multiply(pxclose, dailydeltareb)
        avgcost = np.divide(np.cumsum(dailymv), dailydeltarebrollsum)
        deltapnl = np.multiply(dailydeltarebrollsum, np.subtract(pxclose,avgcost))
        deltastr = 'deltadailypnl'
        df[deltastr][bdate:edate] = deltapnl.diff()
            
        putdailypnlstr = 'putdailypnl'
        voldailypnlstr = 'voldailypnl'
        volpnlstr = 'volpnl'
        equitypnlstr = 'equitypnl'
        equitydailypnlstr = 'equitydailypnl'
       
        
        df[putdailypnlstr][bdate:edate] = df[putdailypnlstr][bdate:edate].fillna(0) + \
                                            optionshares * (df[optstr][bdate:edate].fillna(0))
        
        df[voldailypnlstr][bdate:edate] = df[voldailypnlstr][bdate:edate].fillna(0) + \
                                            optionshares * (df[optstr][bdate:edate].fillna(0) + 
                                            df[deltastr][bdate:edate].fillna(0))
            
        temploc = df.index.get_loc(bdate)
        tempdate = df.index[temploc+1]
        df[volpnlstr][edate] = optionshares * np.sum(df[voldailypnlstr][tempdate:edate].fillna(0))
    
        # long equity
        df[equitydailypnlstr][bdate:edate] = df[equitydailypnlstr][bdate:edate].fillna(0) + \
                                                equityshares * pxclose.diff().fillna(0)
        
        df[equitypnlstr][edate] = equityshares * np.sum(pxclose.diff().fillna(0))
        
       
        cashstr = 'cashfundneed3m'
        df[cashstr][bdate:edate] = df[cashstr][bdate:edate].fillna(0) + optprc[0] * optionshares + np.sum(dailymv) * optionshares + equityshares * pxclose[0]
        df[cashstr][edate] = 0
         
    print(i)

# %% long weekly calls

dow = ['mon', 'tue', 'wed', 'thu', 'fri']

for i in range(0,len(dow)):
    varstr = 'calldailypnl_' + dow[i]
    df[varstr] = np.nan
    varstr = 'callstrike_' + dow[i]
    df[varstr] = np.nan     
    varstr = 'cashreceived_' + dow[i]
    df[varstr] = np.nan
    
# day of week: Monday = 0, Tuesday = 1, Wed = 2, Thur = 3, Frid = 4

dtm = list(reversed(range(6)))
ttm = np.divide(dtm,365)
ttm[-1] = ttm[-1] + 0.00001    

for DofW in range(0,5):

    df_dow = df[df.index.dayofweek==DofW]   
    list_dof = df_dow.index.tolist()   
       
    for i in range(1,len(list_dof)):
    
        bdate = list_dof[i-1]
        edate = list_dof[i]
        pxclose = df.spx_close[bdate:edate]
        
        if len(pxclose) == 6:
            
            # Long weekly 25 delta call without delta hedge daily.
            datarng = df.spx_divy[bdate:edate]
            divy = np.multiply(datarng,0.01)
            datarng = df.ivol25dcall[bdate:edate]
            ivol = np.multiply(datarng,0.01)
            datarng = df.rfr_1m[bdate:edate]
            rfr = np.multiply(datarng,0.01)        
            
            equitymv = 10000000
            delta = 0.25
            # 0.25 equity shares is applied assumed same with the puts.
            equityshares = 0.25*equitymv/pxclose[0]
            optionshares = equityshares * 1/abs(delta)
            
            # optionshares = 30000
            # equityshares = 0.25*optionshares
    
            # Calculate a option strike for a given delta
            strikestr = 'callstrike_' + dow[DofW]
            df[strikestr][bdate] = strkgivendelta(pxclose[0],
                                                  delta,ivol[0],
                                                  rfr[0],
                                                  ttm[0]
                                                  )
    
            optprc = bsprice(pxclose, 
                             df[strikestr][bdate], 
                             ivol, 
                             divy, 
                             rfr, 
                             ttm, 
                             "c"
                             )
            
            # Estimate Tcost
            if ivol[0] >= 0 and ivol[0] < 0.15:
                Tcost = 1.01
            elif ivol[0] >= 0.15 and ivol[0] < 0.25:
                Tcost = 1.0125
            elif ivol[0] >= 0.25 and ivol[0] < 0.35:                
                Tcost = 1.0175
            else:
                Tcost = 1.03                
                            
            optprc[0] = optprc[0] * Tcost
        
            calldailypnlstr = 'calldailypnl_' + dow[DofW] 
            
            df[calldailypnlstr][bdate:edate] = df[calldailypnlstr][bdate:edate].fillna(0) + \
                optionshares * (optprc.diff().fillna(0))
            
 
            cashstr = 'cashreceived_' + dow[DofW]
            df[cashstr][bdate:edate] = df[cashstr][bdate:edate].fillna(0) + optprc[0] * optionshares
            df[cashstr][edate] = 0
            
        print(i)        
 
# %% function calc strike for a given option delta
      
def strkgivendelta(s,d,iv,rfrate,t2m):
    # d = delta neg for put option + for call option
    
    if d < 0:
        # put option
        f1 = norm.ppf(d+1) * iv * np.sqrt(t2m)
        f2 = rfrate * t2m
        f3 = t2m * np.square(iv)/2
        denom = np.exp(f1-f2-f3)
        strike = s/denom        
    else:
        # call option
        f1 = norm.ppf(d) * iv * np.sqrt(t2m)
        f2 = rfrate * t2m
        f3 = t2m * np.square(iv)/2
        denom = np.exp(f1-f2-f3)
        strike = s/denom        
        
    return strike

# %% function to calculate Black Schole call and put option price
# s=pxclose
# x=pxclose[0]
# iv=ivol
# divy=divy
# rfrate=rfr
# t2m=ttm
# t2m[-1]=0+0.00001
# pc="c"

def bsprice(s, x, iv, divy, rfrate, t2m, pc):
    f1 = np.log(np.divide(s,x))
    fwdrate = np.subtract(rfrate,divy)
    rfrateT = np.multiply(rfrate,t2m)
    divyT = np.multiply(divy,t2m)
    
    f2 = np.multiply(np.add(fwdrate,0.5*np.square(iv)),t2m)
    f3 = np.multiply(iv,np.sqrt(t2m))
    d1 = np.divide(np.add(f1,f2),f3)
    d2 = np.subtract(d1,f3)
    if pc=="c":
        f1 = np.multiply(np.multiply(s, np.exp(-divyT)), norm.cdf(d1))
        f2 = np.multiply(np.multiply(x, np.exp(-rfrateT)), norm.cdf(d2))
        optionprice = np.subtract(f1,f2)
    else:
        f1 = np.multiply(np.multiply(x, np.exp(-rfrateT)), norm.cdf(-d2))
        f2 = np.multiply(np.multiply(s, np.exp(-divyT)), norm.cdf(-d1))
        optionprice = np.subtract(f1,f2)
        
    return optionprice        
                                               

# %% function to calc Black Schole delta

def bsdelta(s, x, iv, divy, rfrate, t2m, pc):
    f1 = np.log(np.divide(s,x))
    fwdrate = np.subtract(rfrate,divy)
    divyT = np.multiply(divy,t2m)
    
    f2 = np.multiply(np.add(fwdrate,0.5*np.square(iv)),t2m)
    f3 = np.multiply(iv,np.sqrt(t2m))
    d1 = np.divide(np.add(f1,f2),f3)
    # d2 = np.subtract(d1,f3)
    
    if pc=="c":
        dlta = np.multiply(np.exp(-divyT),norm.cdf(d1))
    else:
        dlta = np.multiply(np.exp(-divyT), np.subtract(norm.cdf(d1),1))        
    
    return dlta               
    

# %% function to calc Black Schole gamma

def bsgamma(s, x, iv, divy, rfrate, t2m):
    f1 = np.log(np.divide(s,x))
    fwdrate = np.subtract(rfrate,divy)
    divyT = np.multiply(divy,t2m)
    
    f2 = np.multiply(np.add(fwdrate,0.5*np.square(iv)),t2m)
    f3 = np.multiply(iv,np.sqrt(t2m))
    d1 = np.divide(np.add(f1,f2),f3)
 
    v1 = np.exp(-divyT)
    v2 = np.multiply(s,np.multiply(iv,np.sqrt(t2m)))
    v3 = np.divide(1,np.sqrt(2*np.pi))
    v4 = np.exp(-0.5*np.square(d1))
    
    # G = np.multiply(np.divide(v1,v2),v3,v4)
    G = np.multiply(np.divide(v1,v2),np.multiply(v3,v4))

    return G         

# %% Analyze results

# Adding up all three days of the week for Long Vol Long Spot return
allwk_pnl = pd.concat(
    [df.equitydailypnl_mon.fillna(0), df.voldailypnl_mon.fillna(0), df.calldailypnl_mon.fillna(0),
    df.equitydailypnl_wed.fillna(0), df.voldailypnl_wed.fillna(0), df.calldailypnl_wed.fillna(0),
    df.equitydailypnl_fri.fillna(0), df.voldailypnl_fri.fillna(0), df.calldailypnl_fri.fillna(0)], 
    axis=1
    )
allwk_pnl = allwk_pnl.sum(axis=1)
allwk_pnl = pd.DataFrame(allwk_pnl)
allwk_pnl.columns = ['lvlp']
allwk_pnl = allwk_pnl.reset_index()
yearly_pnl = allwk_pnl.groupby(allwk_pnl.datetime.dt.to_period("y"))['lvlp'].sum()
allwk_pnl = allwk_pnl.set_index('datetime')

allwk_cash = pd.concat(
    [df.cashequityneed_mon.fillna(0), df.cashfundneed_mon.fillna(0), df.cashreceived_mon.fillna(0),
    df.cashequityneed_tue.fillna(0), df.cashfundneed_tue.fillna(0), df.cashreceived_tue.fillna(0),
    df.cashequityneed_fri.fillna(0), df.cashfundneed_fri.fillna(0), df.cashreceived_fri.fillna(0)],
    axis=1
)
allwk_cash = allwk_cash.sum(axis=1)
allwk_cash = pd.DataFrame(allwk_cash)
allwk_cash.columns = ['cash']
allwk_cash = allwk_cash.reset_index()
yearly_cash = allwk_cash.groupby(allwk_cash.datetime.dt.to_period("y"))['cash'].mean()

# S&P return
alleq_pnl = pd.concat(
    [df.equitydailypnl_mon.fillna(0), df.equitydailypnl_wed.fillna(0), df.equitydailypnl_fri.fillna(0)],
    axis=1
    )

alleq_pnl = alleq_pnl.sum(axis=1)
alleq_pnl = pd.DataFrame(alleq_pnl)
alleq_pnl.columns = ['sp500']

# Adding return for 3 month long vol long spot
all3m_pnl = pd.concat(
    [df.equitydailypnl.fillna(0), df.voldailypnl.fillna(0)], 
    axis=1
    )
all3m_pnl = all3m_pnL.sum(axis=1)
all3m_pnl = pd.DataFrame(all3m_pnl)
all3m_pnl.columns = ['lvlp_3m']

# Adding weekly long vol long spot without the long call option.  We do so we can compare to the 3 month long vol long spot 
# because we have no call for 3m strategy
allwk_nocall_pnl = pd.concat(
    [df.equitydailypnl_mon.fillna(0), df.voldailypnl_mon.fillna(0),
    df.equitydailypnl_wed.fillna(0), df.voldailypnl_wed.fillna(0),
    df.equitydailypnl_fri.fillna(0), df.voldailypnl_fri.fillna(0)], 
    axis=1
    )
allwk_nocall_pnl = allwk_nocall_pnl.sum(axis=1)
allwk_nocall_pnl = pd.DataFrame(allwk_nocall_pnl)
allwk_nocall_pnL.columns = ['lvlp']


yearly_roc = yearly_pnl / yearly_cash

df_roc = yearly_roc.to_frame()

df_roc.hvplot.bar(
    xlabel = 'Year',
    ylabel = '%',
    title = 'Strategy % Return on Cash Deployed',
    )

# %% Assess return/risk profile
columns = ["Backtest Results"]
metrics = [
    "Avg Cash Deployed",
    "Annualized Return",
    "Annual Volatility",
    "Information Ratio",
    "Sortino Ratio",
    "S&P 500 IR",
    "LVLS 3M Tenor IR"]

portfolio_eval_df = pd.DataFrame(index=metrics, columns=columns)
portfolio_eval_df.loc["Avg Cash Deployed"] =(
    format(int(allwk_cash['cash'].mean()), ","))
portfolio_eval_df.loc["Annualized Return"] =(
    format(int(allwk_pnl['lvlp'].mean() * 252), ","))
portfolio_eval_df.loc["Annual Volatility"] =(
    format(int(allwk_pnl['lvlp'].std() * np.sqrt(252)), ","))
portfolio_eval_df.loc["Information Ratio"] =(
    round(allwk_pnl['lvlp'].mean() * 252 / (allwk_pnl['lvlp'].std() * np.sqrt(252)), 2)) 
portfolio_eval_df.loc["Sortino Ratio"] =(
    round(allwk_pnl['lvlp'].mean() * 252 / (allwk_pnl['lvlp'][allwk_pnl['lvlp']<0].std() * np.sqrt(252)), 2))
portfolio_eval_df.loc["S&P 500 IR"] =(
    round(alleq_pnl['sp500'].mean() * 252 / (alleq_pnl['sp500'].std() * np.sqrt(252)), 2)) 
portfolio_eval_df.loc["LVLS 3M Tenor IR"] =(
    round(all3m_pnl['lvlp_3m'].mean() * 252 / (all3m_pnl['lvlp_3m'].std() * np.sqrt(252)), 2)) 


portfolio_eval_df

# %% Charting cumulative returns

allwk_pnl.cumsum().hvplot(title = 'Long Vol Long Spot Weekly Tenor', yformatter='%.0f')

# %% Charting cumulative returns
alleq_pnl.cumsum().hvplot(title = 'S&P 500 Total Return', yformatter='%.0f')


# %%
all3m_pnl.cumsum().hvplot(title = 'Long Vol Long Spot 3 Month Tenor', yformatter='%.0f')

