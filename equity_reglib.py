# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:05:34 2023

@author: gcubb
"""
import numpy as np
from numpy.linalg import lstsq
from numba import njit
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

@njit
def nb_ols(y,x):
    pars = lstsq(x,y)[0]
    pred = np.sum(x*pars,axis=1)
    resid = y-pred
    rsq = 1 - (np.sum(np.power(resid,2))/np.sum(np.power(y-np.mean(y),2)))
    return pars, rsq, resid, pred

@njit
def nb_roll_2s_oos_ols(y, x1, x2, window):
    pars1 = np.zeros_like(y[window:])
    pars2 = np.zeros_like(x2[window:])
    rsqs = np.zeros_like(y[window:])
    step1impactoos = np.zeros_like(y[window:])
    step2impactoos = np.zeros_like(x2[window:])

    for i in range(x1.shape[0]-window):
        xx = x1[i:i+window]
        yy = y[i:i+window]
        pars1[i] = lstsq(xx,yy)[0][1]
        step1impactoos[i] = x1[i+window,1]*pars1[i]
        yy_adj = xx[:,1]*pars1[i]
        yy2 = yy - yy_adj
        xx2 = x2[i:i+window]
        pars2[i] = lstsq(xx2,yy2)[0]
        step2impactoos[i] = x2[i+window]*pars2[i]
        resid_2step = yy - np.sum(xx2*pars2[i],axis=1) - xx[:,1]*pars1[i]
        rsqs[i] = 1 - (np.sum(np.power(resid_2step,2))/np.sum(np.power(yy-np.mean(yy),2)))
    return pars1, pars2, rsqs, step1impactoos, step2impactoos


#=============================================================================
# Fast classes, wrapping the raw regression functions, many using numba
#=============================================================================
class FastOLS:
    def fit(self, y, x):
        self.cleanVars(y,x) #Remember this adds in a constant!
        self.results = nb_ols(self.y_arr, self.x_arr)
    def cleanVars(self, y, x):
        if isinstance(y,pd.DataFrame):
            y_pd = y.iloc[:,0]
        y_pd = y.dropna()
        if isinstance(x,pd.Series):
            x_pd = pd.DataFrame(x)
        else:
            x_pd = x.copy()
        common_index = y_pd.index.intersection(x_pd.index)
        y_pd = y_pd.loc[common_index]
        y_arr = y_pd.to_numpy()
        x_arr = x_pd.loc[common_index]
        x_arr = x_arr.to_numpy()
        x_arr = np.insert(x_arr,0,1,axis=1)
        self.y     = y_pd
        self.y_arr = y_arr
        self.x     = x_pd.loc[common_index]
        self.x_arr = x_arr
    @property
    def pars(self):
        return self.results[0]
    @property
    def rsq(self):
        return self.results[1]
    @property
    def resid(self):
        return self.results[2]
    @property
    def pred(self):
        return self.results[3]

class FastRolling2sOOSOLS:
    # Despite some of the 1/29/25 additions, which shouldn't hurt anything, this fails if send empty x1
    def fit(self, y, x1, x2, window):
        self.window = window
        self.cleanVars(y,x1,x2)
        self.results = nb_roll_2s_oos_ols(self.y_arr, self.x1_arr, self.x2_arr, self.window)
    def cleanVars(self, y, x1, x2):
        if isinstance(y,pd.DataFrame):
            y_pd = y.iloc[:,0] #what is this??
        y_pd = y.dropna()
        if isinstance(x1,pd.Series):
            x1_pd = pd.DataFrame(x1)
        else:
            #x1_pd = x1.copy()
            x1_pd = pd.DataFrame(x1).copy() # added 1/29/25...don't think this hurts anything
        if isinstance(x2,pd.Series):
            x2_pd =  pd.DataFrame(x2)
        else:
            x2_pd = x2.copy()
        if x1_pd.shape[0] == 0:
            common_index = y_pd.index.intersection(x2_pd.index)
        else:
            common_index = y_pd.index.intersection(x1_pd.index).intersection(x2_pd.index)
        y_pd = y_pd.loc[common_index]
        y_arr = y_pd.to_numpy()
        if self.window >= y_pd.shape[0]:
            raise ValueError('Length of rolling window greater than number of obs!')
        if x1_pd.shape[0] == 0:
            x1_arr = np.ones((y_pd.shape[0],1)) #added 1/29/25...hopefully now can send empty x1 and works??
        else:
            x1_arr = x1_pd.loc[common_index]
            x1_arr = x1_arr.to_numpy()
            x1_arr = np.insert(x1_arr,0,1,axis=1)
        x2_arr = x2_pd.loc[common_index]
        x2_arr = x2_arr.to_numpy()
        x2_arr = np.insert(x2_arr,0,1,axis=1)
        self.y = y_pd
        self.xstep1 = x1_pd
        self.xstep2 = x2_pd
        self.y_arr = y_arr
        self.x1_arr = x1_arr
        self.x2_arr = x2_arr
    @property
    def rollingConstant(self):
        return pd.Series(self.results[1][:,0],index = self.y.index[self.window:],name='constant')
    @property
    def rollingStep1Coef(self):
        return pd.Series(self.results[0],index = self.y.index[self.window:],name=self.xstep1.columns[0])
    @property
    def rollingStep2Coef(self):
        return pd.DataFrame(self.results[1][:,1:],index = self.y.index[self.window:],columns=self.xstep2.columns)
    @property
    def rollingParams(self):
        return pd.concat([self.rollingConstant,self.rollingStep2Coef, self.rollingStep1Coef], axis=1)
    @property
    def rollingRsq(self):
        return pd.DataFrame(self.results[2],index = self.y.index[self.window:],columns=['rsqd'])
    @property
    def rollingStep1BetaOOS(self):
        return pd.DataFrame(self.results[3],index = self.y.index[self.window:],columns=['step1betaoos'])
    @property
    def rollingBMBetaOOS(self):
        return pd.DataFrame(self.results[3],index = self.y.index[self.window:],columns=self.xstep1.columns)
    @property
    def rollingStep2BetaOOS(self):
        return pd.DataFrame(np.sum(self.results[4][:,1:],axis=1),index = self.y.index[self.window:],columns=['step2betaoos'])
    @property
    def rollingStep2AllBetaOOS(self):
        return pd.DataFrame(self.results[4][:,1:],index = self.y.index[self.window:],columns=self.xstep2.columns)
    @property
    def rollingAllBetaOOS(self):
        return pd.concat([self.rollingBMBetaOOS,self.rollingStep2AllBetaOOS],axis=1)
    @property
    def rollingAlphaOOS(self):
        return pd.DataFrame(self.y.iloc[self.window:].sub(self.rollingStep1BetaOOS.iloc[:,0]).sub(self.rollingStep2BetaOOS.iloc[:,0].rename('alphaOOS')))
    @property
    def rollingBetaOOS(self):
        return pd.DataFrame(self.rollingStep1BetaOOS.iloc[:,0].add(self.rollingStep2BetaOOS.iloc[:,0].rename('betaOOS')))
    @property
    def rsqAlphaOOS(self):
        # Does the OOS alpha predict returns?
        reg = FastOLS()
        reg.fit(self.y.iloc[self.window:],self.rollingAlphaOOS)
        return reg.rsq
    @property
    def rsqBetaOOS(self):
        # Does the OOS beta predict returns?
        reg = FastOLS()
        reg.fit(self.y.iloc[self.window:],self.rollingBetaOOS)
        return reg.rsq
    @property
    def parsAlphaOOS(self):
        reg = FastOLS()
        reg.fit(self.y.iloc[self.window:],self.rollingAlphaOOS)
        return reg.pars
    @property
    def parsBetaOOS(self):
        reg = FastOLS()
        reg.fit(self.y.iloc[self.window:],self.rollingBetaOOS)
        return reg.pars
    @property
    def rollingPredOOS(self):
        return pd.DataFrame(self.rollingBetaOOS.iloc[:,0].add(self.rollingParams['constant']).rename('predOOS'))
    @property
    def rollingResidOOS(self):
        return pd.DataFrame(self.y.iloc[self.window:].sub(self.rollingPredOOS.iloc[:,0]).rename('residOOS'))
    @property
    def rollingOOSResults(self):
        return pd.concat([self.y.iloc[self.window:].rename('yOOS'),self.rollingPredOOS,self.rollingResidOOS],axis=1)

#==============================================================================
# Group classes
#==============================================================================
class GroupFastRolling2sOOSOLS:
    def __init__(self, y, x1, x2, start=None, end=None, verbose = False, **kwargs):
        # Note: start and end are not used in this class, but are included for consistency with other classes
        self.y = y.copy()
        self.x1 = x1.copy()
        self.x2 = x2.copy()
        self.calc_results(verbose, **kwargs)
    def calc_results(self, verbose, **kwargs):
        executor = ThreadPoolExecutor(max_workers=10)
        futures_dict = {}
        for col in self.y:
            futures_dict[executor.submit(self.calc_model, self.y[col], self.x1, self.x2, **kwargs)] = col
        self.results = {}
        for future in as_completed(futures_dict):
            try:
                self.results[futures_dict[future]] = future.result()
            except Exception as e:
                if verbose:
                    print(f"Error on {futures_dict[future]}: {e}")
                else:
                    pass
    def calc_model(self, y, x1, x2, **kwargs):
        model = FastRolling2sOOSOLS()
        model.fit(y, x1, x2, **kwargs)
        return model
    @property
    def constants(self):
        constants=pd.DataFrame({})
        for name, model in self.results.items():
            if constants.shape[0]<1:
                constants = pd.DataFrame(model.rollingConstant).rename(columns={'constant':name})
            else:
                constants = pd.merge(constants,pd.DataFrame(model.rollingConstant).rename(columns={'constant':name}), left_index=True, right_index=True, how='outer')
        return constants
    @property
    def alpha(self):
        alpha=pd.DataFrame({})
        for name, model in self.results.items():
            if alpha.shape[0]<1:
                alpha = pd.DataFrame(model.rollingAlphaOOS).rename(columns={0:name})
            else:
                #alpha = pd.merge(alpha,pd.DataFrame(model.rollingAlphaOOS).rename(columns={0:name}), left_index=True, right_index=True)
                alpha = pd.merge(alpha,pd.DataFrame(model.rollingAlphaOOS).rename(columns={0:name}), left_index=True, right_index=True, how='outer') #try 12/10/2024
        return alpha
    @property
    def beta(self):
        beta=pd.DataFrame({})
        for name, model in self.results.items():
            if beta.shape[0]<1:
                beta = pd.DataFrame(model.rollingBetaOOS).rename(columns={0:name})
            else:
                beta = pd.merge(beta,pd.DataFrame(model.rollingBetaOOS).rename(columns={0:name}), left_index=True, right_index=True, how='outer')
        return beta
    @property
    def rsqs(self):
        rsqs=pd.DataFrame({})
        for name, model in self.results.items():
            if rsqs.shape[0]<1:
                rsqs = pd.DataFrame(model.rollingRsq).rename(columns={'rsqd':name})
            else:
                rsqs = pd.merge(rsqs,pd.DataFrame(model.rollingRsq).rename(columns={'rsqd':name}), left_index=True, right_index=True, how='outer')
        return rsqs
    @property
    # For each model, these are the alpha and beta contributions, which sum to the return
    def attribution(self):
        allbetas=pd.DataFrame({})
        for name, model in self.results.items():
            if allbetas.shape[0]<1:
                bet1 = model.rollingAllBetaOOS
                ba1  = model.rollingAlphaOOS
                ba1.columns=['alpha']
                bet1 = pd.concat([bet1,ba1],axis=1)
                bet1.columns = pd.MultiIndex.from_product([[model.y.name],bet1.columns])
                allbetas = bet1
            else:
                bet1 = model.rollingAllBetaOOS
                ba1  = model.rollingAlphaOOS
                ba1.columns=['alpha']
                bet1 = pd.concat([bet1,ba1],axis=1)
                bet1.columns = pd.MultiIndex.from_product([[model.y.name],bet1.columns])
                allbetas = pd.concat([allbetas,bet1], axis=1)
        return allbetas
    @property
    # For each model, these are the alpha and beta contributions, which sum to the return
    def params(self):
        allbetas=pd.DataFrame({})
        for name, model in self.results.items():
            if allbetas.shape[0]<1:
                bet1 = model.rollingParams
                bet1.columns = pd.MultiIndex.from_product([[model.y.name],bet1.columns])
                allbetas = bet1
            else:
                bet1 = model.rollingParams
                bet1.columns = pd.MultiIndex.from_product([[model.y.name],bet1.columns])
                allbetas = pd.concat([allbetas,bet1], axis=1)
        return allbetas

