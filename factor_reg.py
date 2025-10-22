import numpy as np
import pandas as pd
import os
import argparse
import warnings
import pickle
import matplotlib.pyplot as plt
import time
from adjustText import adjust_text
import equity_risklib as rl
import equity_reglib as rgl
import chartlib as cl
warnings.filterwarnings("ignore")


def main(argv=None):
    parser = argparse.ArgumentParser(description='Drop portfolio csv in directory, run fin-data, then this generates equity report.')
    parser.add_argument('--db', '-d', help='Path to data', default='C:\\Users\\gcubb\\OneDrive\\Python\\data-hub')
    parser.add_argument('--pfile', '-p', help='Portfolio CSV file', default='ComstockFund_24Q3.csv')
    args = parser.parse_args(argv)
    data_db_root = args.db
    gcsv_file = args.pfile

    betabm = 'SPY'
    spdrdatfile = "spdrfactors"
    mainrtns = 'sprtns'
    mainsect = 'spsect'
    sp500 = 'sp500_history'
    start_date_data = '2012-09-01'
    window = 100
    olswindow_weeks = 520
    pdf_file = gcsv_file[:-4]+".pdf"

    ##==============================================================##
    ## Get the data ready, including factor exposures of all stocks ##
    ##==============================================================##
    with open(os.path.join(data_db_root, spdrdatfile)+".pickle","rb") as f:
        spdrdat = pickle.load(f)
    lin_fac_dict =  {'QUALd':{'z':list(zip([1.0,-1.0],['QUAL',betabm]))}
                    ,'VLUEd':{'z':list(zip([1.0,-1.0],['VLUE',betabm]))}
                    ,'MTUMd':{'z':list(zip([1.0,-1.0],['MTUM',betabm]))}
                    ,'SIZEd':{'z':list(zip([1.0,-1.0],['SIZE',betabm]))}
                    ,'USMVd':{'z':list(zip([1.0,-1.0],['USMV',betabm]))}
                    }
    reg_fac_dict = {'IQLTres': {'y':'IQLT','x':['ACWX']}
                    ,'IVLUres': {'y':'IVLU','x':['ACWX']}
                    ,'MOMres': {'y':'MTUM','x':['QUAL','VLUE','SIZE']}
                    } #not needed, just to demo
    fdat2 =  rl.factor_builder(spdrdat, newfac_lin=lin_fac_dict, newfac_reg=reg_fac_dict)
    x1 = pd.DataFrame(fdat2[betabm].copy())
    x1.columns = ['Beta']
    x2 = fdat2[['QUALd','VLUEd','MTUMd','SIZEd','USMVd']].copy()
    x2.columns = ['Qual','Value','Mom','Small','MinVol']

    # Read in the individual stock return (mainrtns), secmaster (mainsect), and sp500 weights (sp500) data
    with open(os.path.join(data_db_root, mainrtns)+".pickle","rb") as f:
        ydat = pickle.load(f)

    with open(os.path.join(data_db_root, mainsect)+".pickle","rb") as f:
        yinfo = pickle.load(f)

    with open(os.path.join(data_db_root, sp500)+".pickle","rb") as f:
        sp500_dict = pickle.load(f)

    # Get latest sp500 index weights
    sp500_index = sp500_dict[max(sp500_dict.keys())]['sp500_weight']

    # Convert to weekly and then run factor regressions on the individual stock data
    ydrec    = ydat[ydat.index>start_date_data]
    yfridat  = rl.get_friday_returns(ydat)
    x1fridat = rl.get_friday_returns(x1)
    x2fridat = rl.get_friday_returns(x2)
    yfridat2 = yfridat[yfridat.index>start_date_data]
    olsstart = yfridat2.index[-olswindow_weeks].strftime("%Y-%m-%d")
    olsend = yfridat2.index[-1].strftime("%Y-%m-%d")

    numregs = (yfridat2[yfridat2.index>olsstart].shape[0]-window)*yfridat2[yfridat2.index>olsstart].shape[1]
    start = time.time()
    olsgrp = rgl.GroupFastRolling2sOOSOLS(yfridat2, x1fridat, x2fridat, olsstart, olsend, window=window)
    end = time.time()
    print("Time for OLS using Fast running",f"{numregs:,.0f}","regressions: ", np.round(end-start,1),"seconds")

    params = olsgrp.params
    alphas_stocks = olsgrp.alpha
    attrib = olsgrp.attribution # attrib sums exactly to the weekly return for each stock
    rsqs = olsgrp.rsqs

    # Things lightly fail if don't have sector for every index security, so just drop them for now (sigh)
    # ~it is fine if there is missing sector info for portfolio securities - it just all goes to alpha
    yinfo_dropna = yinfo[(yinfo['Sector']!='N/A') & (yinfo.index.isin(sp500_index.index))]
    gfall = (
        pd.read_csv(gcsv_file, index_col=0)
        .assign(wgt=lambda x: x['Value'] / x['Value'].sum())
        .set_index('Ticker')
        .merge(sp500_index, left_index=True, right_index=True, how='outer')
        .merge(yinfo_dropna, left_index=True, right_index=True, how='inner')
        .assign(wgtsub=lambda x: x['wgt'] / x['wgt'].sum())
    )

    # Run an example of a portfolio report vs benchmark of the S&P500 ##
    #bwgts = yinfo['weight_07Oct2024']
    bwgts = (gfall['sp500_weight']
            .fillna(0)
            .pipe(lambda x: x / x.sum())  # reweight since dropped index stocks not in yinfo
            .rename('bwgt'))
    #pwgts = bwgts.nlargest(50)/(bwgts.nlargest(50).sum())
    pwgts = (gfall['wgtsub']
            .fillna(0)
            .loc[lambda x: x != 0]
            .rename('pwgt'))
    horiz = 12
    halflife = 52

    # Begin: Test 
    DOTESTING = False
    if DOTESTING:
        #aws = pd.merge(pwgts, bwgts, left_index=True, right_index=True, how='inner')
        #aws['relwgt'] = aws['pwgt'] - aws['bwgt']
        #aws[aws['pwgt']==aws['relwgt']]
        yinfo_test = yinfo.copy()
        yinfo_test.loc['XRAY', 'Sector'] = 'N/A'  # delete this after testing!!!
        gfall2 = (
            pd.read_csv(gcsv_file, index_col=0)
            .assign(wgt=lambda x: x['Value'] / x['Value'].sum())
            .set_index('Ticker')
            .merge(sp500_index, left_index=True, right_index=True, how='outer')
            .merge(yinfo_test, left_index=True, right_index=True, how='inner')
            .assign(wgtsub=lambda x: x['wgt'] / x['wgt'].sum())
        )
        bwgts2 = (gfall2['sp500_weight']
                .fillna(0)
                .rename('bwgt'))
        pwgts2 = (gfall2['wgtsub']
                .fillna(0)
                .loc[lambda x: x != 0]
                .rename('pwgt'))
        fa2 = rl.FactorAttribution(attrib, pwgts2, bwgts2, yinfo_test['Sector'], horiz, halflife)
        secw_port, alp_port = fa2.alphaAttribution(fa2.attrib, fa2.pwgts, fa2.sector, fa2.hrzn)
        secw_bmk, alp_bmk   = fa2.alphaAttribution(fa2.attrib, fa2.bwgts, fa2.sector, fa2.hrzn)
    # End: Test

    # Calc alpha attribution (agg both by sector, then individ port holdings alphas), margTE by sector*factor, then margTE summed by factor
    facatt = rl.FactorAttribution(attrib, pwgts, bwgts, yinfo_dropna['Sector'], horiz, halflife)
    alpalloc, alpselec, aseclevel = facatt.alphaAllocation
    cov1yr, margte1yr = facatt.factorRisk
    mtesum = pd.DataFrame(margte1yr.sum()).rename(columns={0:'MargTE'})*np.sqrt(52)

    # Merge alpalloc and alpselec into one dataframe
    alp = pd.concat([alpalloc,alpselec], axis=1)
    alp = alp.groupby(alp.index).sum()

    # Calculate the factor contributions to relative performance, to go along with the alpha contributions above to get total portfolio attribution
    pw, bw = rl.reindexZero(pwgts, bwgts)
    allwgts             = pd.merge(pw, bw, left_index=True, right_index=True)
    allwgts.columns     = ['pwgt','bwgt']
    allwgts['relwgt']   = allwgts['pwgt'] - allwgts['bwgt']
    relwgts             = allwgts['relwgt']
    pnls    = attrib.multiply(relwgts, level=0)
    portrtns = yfridat2[aseclevel.index].iloc[-horiz:].sum()
    alpbet_attrib = pd.DataFrame(pnls.iloc[-horiz:].sum()).groupby(level=[1]).sum()
    alpbet_attrib.columns = ['Attribution']
    alpbet_attrib['Sign'] = alpbet_attrib['Attribution'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    # Note this check should equal zero: alpalloc.sum()+alpselec.sum()-alpbet_attrib.loc['alpha','Attribution']

    # Get relative sector weights
    sectwgts = pd.merge(relwgts, yinfo, left_index=True, right_index=True, how='inner')
    sectwgts = sectwgts.groupby('Sector').sum()

    # Calculate the relative factor exposures (betas)
    relcont    = params.multiply(relwgts, level=0)
    relexp = pd.DataFrame(relcont.iloc[-1:]).T.groupby(level=[1]).sum()
    relexp.columns = ['Relative Factor Betas']

    # Create Top/Bottom quartile Alpha Selection and Returns for cleaner charts below
    aseclevel = aseclevel.sort_values('Alpha Selection', ascending=False)
    quartilenum = int(len(aseclevel)/4)
    asecleveltb = pd.concat([aseclevel.head(quartilenum), aseclevel.tail(quartilenum)], axis=0)
    portrtnstb = portrtns[asecleveltb.index]

    ##========================================##
    # Run a one-page attribution report       ##
    ##========================================##
    maintitle = gcsv_file[:-4]+" Factor Attribution Relative to Benchmark: 12 weeks ending "+attrib.index[-1].strftime("%m/%d/%Y")
    fig = plt.figure(figsize=(18,12))
    gs = fig.add_gridspec(22,20)
    fig.suptitle(maintitle, fontsize=20, fontweight='bold')
    fig.subplots_adjust(hspace = 0.4)

    ax = fig.add_subplot(gs[0:9,0:6])
    ax = cl.jbarplot(ax, alpbet_attrib.reset_index(), 'Factor Contribution to Relative Performance', None, 'Return (%)', x='index', y='Attribution', hue='Sign', palette={'Positive': 'royalblue', 'Negative': 'salmon'})

    ax = fig.add_subplot(gs[0:9,7:13])
    ax = cl.jbarplot(ax, relexp.reset_index(), 'Factor Betas Relative-to-Benchmark', None, 'Factor Beta', x='index', y='Relative Factor Betas', color='seagreen')

    ax = fig.add_subplot(gs[0:9,14:20])
    ax = cl.jbarplot(ax, mtesum.reset_index(), 'Annual Marginal Tracking Error', None, 'Marginal TE (%)', x='Factors', y='MargTE', color='orange')

    ax = fig.add_subplot(gs[11:22,0:6])
    tidy = alp.reset_index().melt(id_vars='Sector').rename(columns=str.title)
    ax = cl.jbarplot(ax, tidy, 'Alpha Allocation and Selection', None, 'Return (%)', x='Sector', y='Value', hue='Variable')
    ax2 = ax.twinx()
    #ax2.scatter(sectwgts['Sector'], sectwgts['Value'], color='limegreen', marker='^')
    ax2.axhline(y=0, color='blue', linestyle='--') # add a horizontal line at 0
    ax2.scatter(sectwgts.index, sectwgts['relwgt'], color='limegreen', marker='^')
    ax2.set_ylabel('Relative Weight (%)', fontsize=10)

    ax = fig.add_subplot(gs[11:16,8:20])
    ax.scatter(portrtnstb, asecleveltb['Alpha Selection'])
    ax.axhline(y=0, color='blue', linestyle='--') # add a horizontal line at 0
    texts = [plt.text(portrtnstb[i], asecleveltb['Alpha Selection'][i], asecleveltb.index[i], ha='center', va='center') for i in range(len(asecleveltb))]
    at = adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'))
    ax.set_title('Top/Bottom Quartile Alpha Selection vs Total Return (top) and by Sector (bottom)', fontsize=12)
    ax.set_ylabel('Alpha Selection (%)', fontsize=10)
    ax.grid(which="major", color='k', linestyle='-.', linewidth=0.5)

    ax = fig.add_subplot(gs[17:22,8:20])
    asecleveltb = asecleveltb.sort_values('Sector')
    ax = cl.jlabelstripplot(ax, asecleveltb, 'Sector', 'Alpha Selection', None, None, 'Alpha Selection (%)', jitter=False, marker='o', color='red')
    ax.axhline(y=0, color='blue', linestyle='--') # add a horizontal line at 0
    fig.savefig(pdf_file)
    #fig.show()


if __name__ == '__main__':
    main()