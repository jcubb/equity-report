# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:05:34 2023

@author: gcubb
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


class FactorAttribution:
    ''' Class to perform factor attribution analysis '''
    def __init__(self, attrib, pwgts, bwgts, sector, hrzn, halflife, **kwargs):
        self.attrib = attrib.copy()
        if isinstance(pwgts, pd.DataFrame):
            self.pwgts = pwgts[pwgts.columns[0]]
        else:
            self.pwgts = pwgts.copy()
        if isinstance(bwgts, pd.DataFrame):
            self.bwgts = bwgts[bwgts.columns[0]]
        else:
            self.bwgts = bwgts.copy()
        self.sector = sector.copy()
        self.hrzn = hrzn
        self.halflife = halflife
        #self.calc_results(verbose, **kwargs)
    def alphaAttribution(self, rtnattrib, pwgts, sector, hrzn):
        # Check if pwgts is a DataFrame, which would break the multiply function
        if isinstance(pwgts, pd.DataFrame):
            pwgts = pwgts[pwgts.columns[0]]
        alpsect = (
            rtnattrib.multiply(pwgts, level=0)
            .xs('alpha', level=1, axis=1).T
            .merge(sector, left_index=True, right_index=True)
            .reset_index(drop=True)
            .set_index(sector.name)
            .groupby(sector.name)
            .sum()
            .pipe(lambda x: pd.DataFrame(x.T.iloc[-hrzn:,:].sum()))
        )
        secwgts = (
            pd.merge(pwgts,sector, left_index=True, right_index=True)
            .groupby(sector.name)
            .sum()
        )
        return secwgts, alpsect
    @property
    def alphaAllocation(self):
        # 10/9/25: somewhere in here this is not robust to bmk securities missing sector info!
        # Get everything in PnL terms, make sure pwgts and bwgts align
        secw_port, alp_port = self.alphaAttribution(self.attrib, self.pwgts, self.sector, self.hrzn)
        secw_bmk, alp_bmk   = self.alphaAttribution(self.attrib, self.bwgts, self.sector, self.hrzn)
        alp_port.columns    = ['palp']
        alp_bmk.columns     = ['balp']
        secw_port, secw_bmk = reindexZero(secw_port, secw_bmk)
        alp_port, alp_bmk   = reindexZero(alp_port, alp_bmk)
        allwgts = (pd.merge(secw_port, secw_bmk, left_index=True, right_index=True)
            .rename(columns={0: 'pwgt', 1: 'bwgt'})
            .assign(
                    relwgt=lambda x: x['pwgt'] - x['bwgt'],
                    ratiowgt=lambda x: (x['pwgt']/x['bwgt']).fillna(0).replace(np.inf, 0)
            )
        )
        # Allocation
        alpwgtsbmk = (
            pd.merge(alp_bmk, secw_bmk, left_index=True, right_index=True)
            .assign(ascale_bmk = lambda x: x['balp']/allwgts['bwgt'])
            .fillna(0)
            .replace(np.inf,0)
        )
        ascale_bmk = alpwgtsbmk['ascale_bmk']-alp_bmk.sum()[0]
        alphaalloc = (
            pd.merge(allwgts['relwgt'], ascale_bmk, left_index=True, right_index=True)
            .assign(**{'Alpha Allocation': lambda x: x['relwgt']*ascale_bmk})
            ['Alpha Allocation']
        )
        # Selection
        alphaselec = (
            pd.merge(alp_port, allwgts['ratiowgt'], left_index=True, right_index=True)
            .merge(alp_bmk, left_index=True, right_index=True)
            .assign(**{'Alpha Selection': lambda x: x['palp']-x['ratiowgt']*x['balp']})
            ['Alpha Selection']
        )
        # Selection by security (1/31/25: pretty sure this is just for the holdings, not alpha on implicit shorts vs. bmk, but I have now forgotten for sure and need to check...)
        # recreate beginning of alphaAttribution for the portfolio
        pnls = self.attrib.multiply(self.pwgts, level=0)
        secbmalp = pd.merge(pd.DataFrame(self.sector),alpwgtsbmk['ascale_bmk'], left_on=self.sector.name, right_index=True)
        alpsect = (
            pnls
            .xs('alpha', level=1, axis=1)
            .iloc[-self.hrzn:,:]
            .sum()
            .to_frame('palp')
            .merge(secbmalp, left_index=True, right_index=True)
            .merge(self.pwgts, left_index=True, right_index=True)
            .assign(**{'Alpha Selection': lambda x: x['palp'] - x['pwgt']*x['ascale_bmk']})
        )
        return alphaalloc, alphaselec, alpsect[[self.sector.name,'Alpha Selection']]
    @property
    def factorRisk(self):
        # Get everything in PnL terms, make sure pwgts and bwgts align
        pw, bw              = reindexZero(self.pwgts, self.bwgts)
        pnltick = (
            pd.merge(pw, bw, left_index=True, right_index=True)
            .rename(columns={0: 'pwgt', 1: 'bwgt'})
            .assign(relwgt=lambda x: x['pwgt'] - x['bwgt'])
            .pipe(lambda x: self.attrib.multiply(x['relwgt'], level=0))
            .T
           .pipe(lambda df: df.set_index(pd.MultiIndex.from_arrays(
               [df.index.get_level_values(0),
                df.index.get_level_values(1)],
               names=['Ticker', 'Factors'])))
            .join(self.sector, how='inner')
            .reset_index()
            .set_index(['Ticker', self.sector.name, 'Factors'])
            .groupby([self.sector.name, 'Factors'])
            .sum()        
        )
        covmat   = risk_model(pnltick.T, self.halflife)
        ones     = np.ones(covmat.shape[0])
        # Make margte a pivot table, with Factors as columns and Sectors as rows
        margte = (
            pd.DataFrame(
                (covmat @ ones)/np.sqrt(ones.T @ covmat @ ones)
            )
            .reset_index()
            .pivot(index=self.sector.name, columns='Factors', values=0)
        )
        return covmat, margte 

def risk_model(data, halflife):
    ''' Calculate a covariance matrix risk model '''
    ewdat        = data.ewm(halflife=halflife).cov()
    covmat       = ewdat[-data.shape[1]:] #Changed 2/6/25
    covmat.index = covmat.index.droplevel(level=0)
    return covmat

def get_friday_returns(ydatin):
    # add in any missing days to ydat and set returns to zero
    ydatall    = ydatin.resample('D').asfreq()
    ydatall    = ydatall.fillna(0)
    yroll      = ydatall.rolling(7).sum()
    yfridatout = yroll[yroll.index.weekday==4] #4 is Friday
    return yfridatout

def reindexZero(vec1, vec2):
    indunion = vec1.index.union(vec2.index)
    vec1 = vec1.reindex(indunion, fill_value=0)
    vec2 = vec2.reindex(indunion, fill_value=0)
    return vec1, vec2

def factor_builder(facin, newfac_lin=None, newfac_reg=None):
    ''' Add new factors to facin that are linear combinations of existing factors or regression residuals '''
    facout = facin.copy()
    if newfac_lin is not None:
        newx = pd.DataFrame()
        for value in newfac_lin:
            zl = newfac_lin[value]['z']
            newx[value] = pd.DataFrame([facin[zl[ind][1]]*zl[ind][0] for (ind,k) in enumerate(zl)]).sum(axis=0, skipna=False)
        facout = pd.merge(facout, newx, left_index=True, right_index=True, how='outer')
    if newfac_reg is not None:
        newx = pd.DataFrame()
        for value in newfac_reg:
            yvar = facin[newfac_reg[value]['y']].dropna()
            xvar = facin[newfac_reg[value]['x']].dropna()
            overlap = xvar.index.intersection(yvar.index)
            xvar = xvar.loc[overlap]
            yvar = yvar.loc[overlap]
            xvarc = sm.add_constant(xvar)
            OLS_model = sm.OLS(yvar,xvarc).fit()
            newx[value] = OLS_model.resid
        facout = pd.merge(facout, newx, left_index=True, right_index=True, how='outer')
    return facout



