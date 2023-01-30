#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 23:10:06 2023

@author: scarlett
"""

#%% Import library 
import pandas as pd
import numpy as np
import datetime as dt
import wrds
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
import pickle as pkl
from functions import *

#%% Connect to WRDS
conn = wrds.Connection(wrds_username = 'sijia_yin')

#%% TTM Functions
#######################################################################################################################
#                                                    TTM functions                                                    #
#######################################################################################################################


def ttm4(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm4
    """
    lag = pd.DataFrame()
    for i in range(1, 4):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby('gvkey')['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s1' % series] + lag['%s2' % series] + lag['%s3' % series]
    return result


def ttm12(series, df):
    """

    :param series: variables' name
    :param df: dataframe
    :return: ttm12
    """
    lag = pd.DataFrame()
    for i in range(1, 12):
        lag['%(series)s%(lag)s' % {'series': series, 'lag': i}] = df.groupby('permno')['%s' % series].shift(i)
    result = df['%s' % series] + lag['%s1' % series] + lag['%s2' % series] + lag['%s3' % series] +\
             lag['%s4' % series] + lag['%s5' % series] + lag['%s6' % series] + lag['%s7' % series] +\
             lag['%s8' % series] + lag['%s9' % series] + lag['%s10' % series] + lag['%s11' % series]
    return result

print('TTM')

#%%
#######################################################################################################################
#                                                  Compustat Block                                                    #
#######################################################################################################################
comp = conn.raw_sql("""
                    /*header info*/
                    select c.gvkey, f.cusip, f.datadate, f.fyear, c.cik, substr(c.sic,1,2) as sic2, c.sic, c.naics,
                    
                    /*firm variables*/
                    /*income statement*/
                    f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda,
                    f.ebit, f.nopi, f.spi, f.pi, f.txp, f.ni, f.txfed, f.txfo, f.txt, f.xint, f.xpp, f.xacc,
                    
                    /*CF statement and others*/
                    f.capx, f.oancf, f.dvt, f.ob, f.gdwlia, f.gdwlip, f.gwo, f.mib, f.oiadp, f.ivao, f.ivst,
                    
                    /*assets*/
                    f.rect, f.act, f.che, f.ppegt, f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, f.fatl,
                    
                    /*liabilities*/
                    f.lct, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, 
                    f.dcpstk, f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi, f.dltis, f.dltr, f.dlcch,
                    
                    /*equity and other*/
                    f.ceq, f.scstkc, f.emp, f.csho, f.seq, f.txditc, f.pstkrv, f.pstkl, f.np, f.txdc, 
                    f.dpc, f.ajex, f.tstkp, f.oibdp, f.capxv, f.dvpa, f.epspx,
                    
                    /*market*/
                    abs(f.prcc_f) as prcc_f, abs(f.prcc_c) as prcc_c, f.dvc, f.prstkc, f.sstk, f.fopt, f.wcap
                    
                    from comp.funda as f
                    left join comp.company as c
                    on f.gvkey = c.gvkey
                    
                    /*get consolidated, standardized, industrial format statements*/
                    where f.indfmt = 'INDL' 
                    and f.datafmt = 'STD'
                    and f.popsrc = 'D'
                    and f.consol = 'C'
                    and f.datadate >= '01/01/1959'
                    """)

# convert datadate to date fmt
comp['datadate'] = pd.to_datetime(comp['datadate'])

# sort and clean up
comp = comp.sort_values(by=['gvkey', 'datadate']).drop_duplicates()

# clean up csho
comp['csho'] = np.where(comp['csho'] == 0, np.nan, comp['csho'])

# calculate Compustat market equity
comp['mve_f'] = comp['csho'] * comp['prcc_f']

# do some clean up. several variables have lots of missing values
condlist = [comp['drc'].notna() & comp['drlt'].notna(),
            comp['drc'].notna() & comp['drlt'].isnull(),
            comp['drlt'].notna() & comp['drc'].isnull()]
choicelist = [comp['drc']+comp['drlt'],
              comp['drc'],
              comp['drlt']]
comp['dr'] = np.select(condlist, choicelist, default=np.nan)

condlist = [comp['dcvt'].isnull() & comp['dcpstk'].notna() & comp['pstk'].notna() & comp['dcpstk'] > comp['pstk'],
            comp['dcvt'].isnull() & comp['dcpstk'].notna() & comp['pstk'].isnull()]
choicelist = [comp['dcpstk']-comp['pstk'],
              comp['dcpstk']]
comp['dc'] = np.select(condlist, choicelist, default=np.nan)
comp['dc'] = np.where(comp['dc'].isnull(), comp['dcvt'], comp['dc'])

comp['xint0'] = np.where(comp['xint'].isnull(), 0, comp['xint'])
comp['xsga0'] = np.where(comp['xsga'].isnull, 0, 0)

comp['ceq'] = np.where(comp['ceq'] == 0, np.nan, comp['ceq'])
comp['at'] = np.where(comp['at'] == 0, np.nan, comp['at'])
comp = comp.dropna(subset=['at'])
print('compustat')




#%% CRSP BLOCK

#######################################################################################################################
#                                                       CRSP Block                                                    #
#######################################################################################################################
# Create a CRSP Subsample with Monthly Stock and Event Variables
# Restrictions will be applied later
# Select variables from the CRSP monthly stock and event datasets
crsp = conn.raw_sql("""
                      select a.prc, a.ret, a.retx, a.shrout, a.vol, a.cfacpr, a.cfacshr, a.date, a.permno, a.permco,
                      b.ticker, b.ncusip, b.shrcd, b.exchcd
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date >= '01/01/1959'
                      and b.exchcd between 1 and 3
                      """)

# change variable format to int
crsp[['permco', 'permno', 'shrcd', 'exchcd']] = crsp[['permco', 'permno', 'shrcd', 'exchcd']].astype(int)

# Line up date to be end of month
crsp['date'] = pd.to_datetime(crsp['date'])
crsp['monthend'] = crsp['date'] + MonthEnd(0)  # set all the date to the standard end date of month

crsp = crsp.dropna(subset=['prc'])
crsp['me'] = crsp['prc'].abs() * crsp['shrout']  # calculate market equity 

# if Market Equity is Nan then let return equals to 0
crsp['ret'] = np.where(crsp['me'].isnull(), 0, crsp['ret'])
crsp['retx'] = np.where(crsp['me'].isnull(), 0, crsp['retx'])

# impute me
crsp = crsp.sort_values(by=['permno', 'date']).drop_duplicates()
crsp['me'] = np.where(crsp['permno'] == crsp['permno'].shift(1), crsp['me'].fillna(method='ffill'), crsp['me'])

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['monthend', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['monthend', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=['monthend', 'permco', 'me'])
# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=['monthend', 'permco'])
# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by=['permno', 'monthend']).drop_duplicates()
print('crsp')




#%%

#######################################################################################################################
#                                                        CCM Block                                                    #
#######################################################################################################################
# merge CRSP and Compustat
# reference: https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
ccm = conn.raw_sql("""
                  select gvkey, lpermno as permno, linktype, linkprim, 
                  linkdt, linkenddt
                  from crsp.ccmxpf_linktable
                  where substr(linktype,1,1)='L'
                  and (linkprim ='C' or linkprim='P')
                  """)

ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])

# if linkenddt is missing then set to today date
ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.to_datetime('today'))

# merge ccm and comp
ccm1 = pd.merge(comp, ccm, how='left', on=['gvkey'])

# we can only get the accounting data after the firm public their report
# for annual data, we use 5 or 6 months lagged data
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['datadate'] + MonthEnd(4)

# set link date bounds
ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]

# link comp and crsp
crsp2 = crsp2.rename(columns={'monthend': 'jdate'})
data_rawa = pd.merge(crsp2, ccm2, how='inner', on=['permno', 'jdate'])

# filter exchcd & shrcd
data_rawa = data_rawa[((data_rawa['exchcd'] == 1) | (data_rawa['exchcd'] == 2) | (data_rawa['exchcd'] == 3)) &
                   ((data_rawa['shrcd'] == 10) | (data_rawa['shrcd'] == 11))]

# process Market Equity
'''
Note: me is CRSP market equity, mve_f is Compustat market equity. Please choose the me below.
'''
data_rawa['me'] = data_rawa['me']/1000  # CRSP ME
# data_rawa['me'] = data_rawa['mve_f']  # Compustat ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
data_rawa['me'] = np.where(data_rawa['me'] == 0, np.nan, data_rawa['me'])
data_rawa = data_rawa.dropna(subset=['me'])

# count single stock years
# data_rawa['count'] = data_rawa.groupby(['gvkey']).cumcount()

# deal with the duplicates
data_rawa.loc[data_rawa.groupby(['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
data_rawa = data_rawa[data_rawa['temp'].notna()]
data_rawa.loc[data_rawa.groupby(['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
data_rawa = data_rawa[data_rawa['temp'].notna()]

data_rawa = data_rawa.sort_values(by=['permno', 'jdate'])
print('ccm')

##################  FINISH MERHING ####################

#%% 2.

# acc
data_rawa['act_l1'] = data_rawa.groupby(['permno'])['act'].shift(1)
data_rawa['lct_l1'] = data_rawa.groupby(['permno'])['lct'].shift(1)

condlist = [data_rawa['np'].isnull(),
          data_rawa['act'].isnull() | data_rawa['lct'].isnull()]
choicelist = [((data_rawa['act']-data_rawa['lct'])-(data_rawa['act_l1']-data_rawa['lct_l1'])/(10*data_rawa['be'])),
              (data_rawa['ib']-data_rawa['oancf'])/(10*data_rawa['be'])]
data_rawa['acc'] = np.select(condlist,
                            choicelist,
                            default=((data_rawa['act']-data_rawa['lct']+data_rawa['np'])-
                                     (data_rawa['act_l1']-data_rawa['lct_l1']+data_rawa['np'].shift(1)))/(10*data_rawa['be']))
#%% 5 
# agr
data_rawa['at_l1'] = data_rawa.groupby(['permno'])['at'].shift(1)
data_rawa['agr'] = (data_rawa['at']-data_rawa['at_l1'])/data_rawa['at_l1']
#%% 9
# bm
data_rawa['bm'] = data_rawa['be'] / data_rawa['me']

#%% 10

# bm_ia
df_temp = data_rawa.groupby(['datadate', 'ffi49'], as_index=False)['bm'].mean()
df_temp = df_temp.rename(columns={'bm': 'bm_ind'})
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=['datadate', 'ffi49'])
data_rawa['bm_ia'] = data_rawa['bm'] - data_rawa['bm_ind']

#%% 11
# cash
data_rawa['cash'] = data_rawa['che']/data_rawa['at']

#%% 17 
# chcsho
data_rawa['chcsho'] = (data_rawa['csho']/data_rawa['csho_l1'])-1

#%% 19
# chinv
data_rawa['chinv'] = (data_rawa['invt'] - data_rawa['invt_l1'])/((data_rawa['at'] + data_rawa['at_l2'])/2)

#%% 22

# chtx
data_rawa['txt_l1'] = data_rawa.groupby(['permno'])['txt'].shift(1)
data_rawa['chtx'] = (data_rawa['txt']-data_rawa['txt_l1'])/data_rawa['at_l1']

#%% 25

# currat
data_rawa['currat'] = data_rawa['act']/data_rawa['lct']

#%% 32
# egr
data_rawa['ceq_l1'] = data_rawa.groupby(['permno'])['ceq'].shift(1)
data_rawa['egr'] = ((data_rawa['ceq']-data_rawa['ceq_l1'])/data_rawa['ceq_l1'])

#%% 36
# grltnoa
data_rawa['aco_l1'] = data_rawa.groupby(['permno'])['aco'].shift(1)
data_rawa['intan_l1'] = data_rawa.groupby(['permno'])['intan'].shift(1)
data_rawa['ao_l1'] = data_rawa.groupby(['permno'])['ao'].shift(1)
data_rawa['ap_l1'] = data_rawa.groupby(['permno'])['ap'].shift(1)
data_rawa['lco_l1'] = data_rawa.groupby(['permno'])['lco'].shift(1)
data_rawa['lo_l1'] = data_rawa.groupby(['permno'])['lo'].shift(1)
data_rawa['rect_l1'] = data_rawa.groupby(['permno'])['rect'].shift(1)

data_rawa['grltnoa'] = ((data_rawa['rect']+data_rawa['invt']+data_rawa['ppent']+data_rawa['aco']+data_rawa['intan']+
                       data_rawa['ao']-data_rawa['ap']-data_rawa['lco']-data_rawa['lo'])
                        -(data_rawa['rect_l1']+data_rawa['invt_l1']+data_rawa['ppent_l1']+data_rawa['aco_l1']
                       +data_rawa['intan_l1']+data_rawa['ao_l1']-data_rawa['ap_l1']-data_rawa['lco_l1']
                       -data_rawa['lo_l1'])
                        -(data_rawa['rect']-data_rawa['rect_l1']+data_rawa['invt']-data_rawa['invt_l1']
                          +data_rawa['aco']-data_rawa['aco_l1']
                          -(data_rawa['ap']-data_rawa['ap_l1']+data_rawa['lco']-data_rawa['lco_l1'])-data_rawa['dp']))\
                       /((data_rawa['at']+data_rawa['at_l1'])/2)


#%% 37
# herf
data_rawa['sic'] = data_rawa['sic'].astype(int)
data_rawa['ffi49'] = ffi49(data_rawa)
data_rawa['ffi49'] = data_rawa['ffi49'].fillna(49)
data_rawa['ffi49'] = data_rawa['ffi49'].astype(int)
df_temp = data_rawa.groupby(['datadate', 'ffi49'], as_index=False)['sale'].sum()
df_temp = df_temp.rename(columns={'sale': 'indsale'})
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=['datadate', 'ffi49'])
data_rawa['herf'] = (data_rawa['sale']/data_rawa['indsale'])*(data_rawa['sale']/data_rawa['indsale'])
df_temp = data_rawa.groupby(['datadate', 'ffi49'], as_index=False)['herf'].sum()
data_rawa = data_rawa.drop(['herf'], axis=1)
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=['datadate', 'ffi49'])

#%% 38
# hire
data_rawa['emp_l1'] = data_rawa.groupby(['permno'])['emp'].shift(1)
data_rawa['hire'] = (data_rawa['emp'] - data_rawa['emp_l1'])/data_rawa['emp_l1']
data_rawa['hire'] = np.where((data_rawa['emp'].isnull()) | (data_rawa['emp_l1'].isnull()), 0, data_rawa['hire'])

#%% 42
# invest
data_rawa['ppent_l1'] = data_rawa.groupby(['permno'])['ppent'].shift(1)
data_rawa['invt_l1'] = data_rawa.groupby(['permno'])['invt'].shift(1)

data_rawa['invest'] = np.where(data_rawa['ppegt'].isnull(), ((data_rawa['ppent']-data_rawa['ppent_l1'])+
                                                             (data_rawa['invt']-data_rawa['invt_l1']))/data_rawa['at_l1'],
                             ((data_rawa['ppegt']-data_rawa['ppent_l1'])+(data_rawa['invt']-data_rawa['invt_l1']))/data_rawa['at_l1'])


#%% 54
# operprof
data_rawa['operprof'] = (data_rawa['revt']-data_rawa['cogs']-data_rawa['xsga0']-data_rawa['xint0'])/data_rawa['ceq_l1']


#%% 56
# pchcapx
data_rawa['capx_l1'] = data_rawa.groupby(['permno'])['capx'].shift(1)
data_rawa['pchcapx'] = (data_rawa['capx']-data_rawa['capx_l1'])/data_rawa['capx_l1']


#%% 57
# pchcurrat
data_rawa['pchcurrat'] = ((data_rawa['act']/data_rawa['lct'])-(data_rawa['act_l1']/data_rawa['lct_l1']))\
                         /(data_rawa['act_l1']/data_rawa['lct_l1'])

#%% 58
# pchdepr
data_rawa['dp_l1'] = data_rawa.groupby(['permno'])['dp'].shift(1)
data_rawa['pchdepr'] = ((data_rawa['dp']/data_rawa['ppent'])-(data_rawa['dp_l1']
                                                              /data_rawa['ppent_l1']))\
                       / (data_rawa['dp_l1']/data_rawa['ppent'])

#%% 59
# pchgm_pchsale
data_rawa['cogs_l1'] = data_rawa.groupby(['permno'])['cogs'].shift(1)
data_rawa['pchgm_pchsale'] = (((data_rawa['sale']-data_rawa['cogs'])
                               - (data_rawa['sale_l1']-data_rawa['cogs_l1']))/(data_rawa['sale_l1']-data_rawa['cogs_l1']))\
                             - ((data_rawa['sale']-data_rawa['sale_l1'])/data_rawa['sale'])


#%% 60
# pchquick
data_rawa['pchquick'] = ((data_rawa['act']-data_rawa['invt'])/data_rawa['lct']
                         -(data_rawa['act_l1']-data_rawa['invt_l1'])/data_rawa['lct_l1'])\
                        /((data_rawa['act_l1']-data_rawa['invt_l1'])/data_rawa['lct_l1'])


#%% 61
# pchsale_pchinvt
data_rawa['pchsale_pchinvt'] = ((data_rawa['sale'] - data_rawa['sale_l1'])/data_rawa['sale_l1'])\
                               - ((data_rawa['invt']-data_rawa['invt_l1'])/data_rawa['invt_l1'])


#%% 62 
# pchsale_pchrect
data_rawa['rect_l1'] = data_rawa.groupby(['permno'])['rect'].shift(1)
data_rawa['pchsale_pchrect'] = ((data_rawa['sale']-data_rawa['sale_l1'])/data_rawa['sale_l1'])\
                               - ((data_rawa['rect']-data_rawa['rect_l1'])/data_rawa['rect_l1'])


#%% 63
# pchsale_pchxsga
data_rawa['xsga_l1'] = data_rawa.groupby(['permno'])['xsga'].shift(1)
data_rawa['pchsale_pchxsga'] = ((data_rawa['sale']-data_rawa['sale_l1'])/data_rawa['sale_l1'])\
                               - ((data_rawa['xsga']-data_rawa['xsga_l1'])/data_rawa['xsga_l1'])


#%% 64
# pchsale_pchinvt
data_rawa['pchsale_pchinvt'] = ((data_rawa['sale'] - data_rawa['sale_l1'])/data_rawa['sale_l1'])\
   

#%% 68
# quick
data_rawa['quick'] = (data_rawa['act']-data_rawa['invt'])/data_rawa['lct']

#%% 72
# realestate
data_rawa['realestate'] = (data_rawa['fatb']+data_rawa['fatl'])/data_rawa['ppegt']
data_rawa['realestate'] = np.where(data_rawa['ppegt'].isnull(),
                                  (data_rawa['fatb']+data_rawa['fatl'])/data_rawa['ppent'], data_rawa['realestate'])

#%% 77
# roic
data_rawa['roic'] = (data_rawa['ebit'] - data_rawa['nopi'])/(data_rawa['ceq'] + data_rawa['lt'] - data_rawa['che'])

#%% 79
# salecash
data_rawa['salecash'] = data_rawa['sale']/data_rawa['che']

#%% 80
# saleinv
data_rawa['saleinv'] = data_rawa['sale']/data_rawa['invt']


#%% 81
# salerec
data_rawa['salerec']= data_rawa['sale']/data_rawa['rect']

#%% 84
# sgr
data_rawa['sgr'] = (data_rawa['sale']/data_rawa['sale_l1'])-1

#%% 
#########################################################################
################                   Quarterly       ######################
#########################################################################


#######################################################################################################################
#                                              Compustat Quarterly Raw Info                                           #
#######################################################################################################################
comp = conn.raw_sql("""
                    /*header info*/
                    select c.gvkey, f.cusip, f.datadate, f.fyearq,  substr(c.sic,1,2) as sic2, c.sic, f.fqtr, f.rdq,

                    /*income statement*/
                    f.ibq, f.saleq, f.txtq, f.revtq, f.cogsq, f.xsgaq, f.revty, f.cogsy, f.saley,

                    /*balance sheet items*/
                    f.atq, f.actq, f.cheq, f.lctq, f.dlcq, f.ppentq, f.ppegtq,

                    /*others*/
                    abs(f.prccq) as prccq, abs(f.prccq)*f.cshoq as mveq_f, f.ceqq, f.seqq, f.pstkq, f.ltq,
                    f.pstkrq, f.gdwlq, f.intanq, f.mibq, f.oiadpq, f.ivaoq,
                    
                    /* v3 my formula add*/
                    f.ajexq, f.cshoq, f.txditcq, f.npq, f.xrdy, f.xrdq, f.dpq, f.xintq, f.invtq, f.scstkcy, f.niq,
                    f.oancfy, f.dlttq, f.rectq, f.acoq, f.apq, f.lcoq, f.loq, f.aoq

                    from comp.fundq as f
                    left join comp.company as c
                    on f.gvkey = c.gvkey

                    /*get consolidated, standardized, industrial format statements*/
                    where f.indfmt = 'INDL' 
                    and f.datafmt = 'STD'
                    and f.popsrc = 'D'
                    and f.consol = 'C'
                    and f.datadate >= '01/01/1959'
                    """)

# comp['cusip6'] = comp['cusip'].str.strip().str[0:6]
comp = comp.dropna(subset=['ibq'])

# sort and clean up
comp = comp.sort_values(by=['gvkey', 'datadate']).drop_duplicates()
comp['cshoq'] = np.where(comp['cshoq'] == 0, np.nan, comp['cshoq'])
comp['ceqq'] = np.where(comp['ceqq'] == 0, np.nan, comp['ceqq'])
comp['atq'] = np.where(comp['atq'] == 0, np.nan, comp['atq'])
comp = comp.dropna(subset=['atq'])

# convert datadate to date fmt
comp['datadate'] = pd.to_datetime(comp['datadate'])

# merge ccm and comp
ccm1 = pd.merge(comp, ccm, how='left', on=['gvkey'])
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['datadate'] + MonthEnd(3)  # we change quarterly lag here
# ccm1['jdate'] = ccm1['datadate']+MonthEnd(4)

# set link date bounds
ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (ccm1['jdate'] <= ccm1['linkenddt'])]

# merge ccm2 and crsp2
# crsp2['jdate'] = crsp2['monthend']
data_rawq = pd.merge(crsp2, ccm2, how='inner', on=['permno', 'jdate'])

# filter exchcd & shrcd
data_rawq = data_rawq[((data_rawq['exchcd'] == 1) | (data_rawq['exchcd'] == 2) | (data_rawq['exchcd'] == 3)) &
                   ((data_rawq['shrcd'] == 10) | (data_rawq['shrcd'] == 11))]

# process Market Equity
'''
Note: me is CRSP market equity, mveq_f is Compustat market equity. Please choose the me below.
'''
data_rawq['me'] = data_rawq['me']/1000  # CRSP ME
# data_rawq['me'] = data_rawq['mveq_f']  # Compustat ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
data_rawq['me'] = np.where(data_rawq['me'] == 0, np.nan, data_rawq['me'])
data_rawq = data_rawq.dropna(subset=['me'])

# count single stock years
# data_rawq['count'] = data_rawq.groupby(['gvkey']).cumcount()

# deal with the duplicates
data_rawq.loc[data_rawq.groupby(['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
data_rawq = data_rawq[data_rawq['temp'].notna()]
data_rawq.loc[data_rawq.groupby(['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
data_rawq = data_rawq[data_rawq['temp'].notna()]

data_rawq = data_rawq.sort_values(by=['permno', 'jdate'])
print('quarterly raw')





























#%% 23

# cinvest
data_rawq['ppentq_l1'] = data_rawq.groupby(['permno'])['ppentq'].shift(1)
data_rawq['ppentq_l2'] = data_rawq.groupby(['permno'])['ppentq'].shift(2)
data_rawq['ppentq_l3'] = data_rawq.groupby(['permno'])['ppentq'].shift(3)
data_rawq['ppentq_l4'] = data_rawq.groupby(['permno'])['ppentq'].shift(4)
data_rawq['saleq_l1'] = data_rawq.groupby(['permno'])['saleq'].shift(1)
data_rawq['saleq_l2'] = data_rawq.groupby(['permno'])['saleq'].shift(2)
data_rawq['saleq_l3'] = data_rawq.groupby(['permno'])['saleq'].shift(3)

data_rawq['c_temp1'] = (data_rawq['ppentq_l1'] - data_rawq['ppentq_l2']) / data_rawq['saleq_l1']
data_rawq['c_temp2'] = (data_rawq['ppentq_l2'] - data_rawq['ppentq_l3']) / data_rawq['saleq_l2']
data_rawq['c_temp3'] = (data_rawq['ppentq_l3'] - data_rawq['ppentq_l4']) / data_rawq['saleq_l3']

data_rawq['cinvest'] = ((data_rawq['ppentq'] - data_rawq['ppentq_l1']) / data_rawq['saleq'])\
                       -(data_rawq[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))

data_rawq['c_temp1'] = (data_rawq['ppentq_l1'] - data_rawq['ppentq_l2']) / 0.01
data_rawq['c_temp2'] = (data_rawq['ppentq_l2'] - data_rawq['ppentq_l3']) / 0.01
data_rawq['c_temp3'] = (data_rawq['ppentq_l3'] - data_rawq['ppentq_l4']) / 0.01

data_rawq['cinvest'] = np.where(data_rawq['saleq']<=0, ((data_rawq['ppentq'] - data_rawq['ppentq_l1']) / 0.01)
                                -(data_rawq[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1)), data_rawq['cinvest'])

data_rawq = data_rawq.drop(['c_temp1', 'c_temp2', 'c_temp3'], axis=1)

#%% 26
# depr
data_rawq['depr'] = ttm4('dpq', data_rawq)/data_rawq['ppentq']


#%% 33
# ep
data_rawq['ep'] = data_rawq['ibq4']/data_rawq['me']

#%% 34
# gma
data_rawq['revtq4'] = ttm4('revtq', data_rawq)
data_rawq['cogsq4'] = ttm4('cogsq', data_rawq)
data_rawq['gma'] = (data_rawq['revtq4']-data_rawq['cogsq4'])/data_rawq['atq_l4']
# gma
data_rawq['revtq4'] = ttm4('revtq', data_rawq)
data_rawq['cogsq4'] = ttm4('cogsq', data_rawq)
data_rawq['gma'] = (data_rawq['revtq4']-data_rawq['cogsq4'])/data_rawq['atq_l4']


#%% 43
# lev
data_rawq['lev'] = data_rawq['ltq']/data_rawq['me']

#%% 44
# lgr
data_rawa['lt_l1'] = data_rawa.groupby(['permno'])['lt'].shift(1)
data_rawa['lgr'] = (data_rawa['lt']/data_rawa['lt_l1'])-1

# lgr
data_rawq['lgr'] = (data_rawq['ltq']/data_rawq['ltq_l4'])-1



#%% 53
# nincr
data_rawq['ibq_l1'] = data_rawq.groupby(['permno'])['ibq'].shift(1)
data_rawq['ibq_l2'] = data_rawq.groupby(['permno'])['ibq'].shift(2)
data_rawq['ibq_l3'] = data_rawq.groupby(['permno'])['ibq'].shift(3)
data_rawq['ibq_l4'] = data_rawq.groupby(['permno'])['ibq'].shift(4)
data_rawq['ibq_l5'] = data_rawq.groupby(['permno'])['ibq'].shift(5)
data_rawq['ibq_l6'] = data_rawq.groupby(['permno'])['ibq'].shift(6)
data_rawq['ibq_l7'] = data_rawq.groupby(['permno'])['ibq'].shift(7)
data_rawq['ibq_l8'] = data_rawq.groupby(['permno'])['ibq'].shift(8)

data_rawq['nincr_temp1'] = np.where(data_rawq['ibq'] > data_rawq['ibq_l1'], 1, 0)
data_rawq['nincr_temp2'] = np.where(data_rawq['ibq_l1'] > data_rawq['ibq_l2'], 1, 0)
data_rawq['nincr_temp3'] = np.where(data_rawq['ibq_l2'] > data_rawq['ibq_l3'], 1, 0)
data_rawq['nincr_temp4'] = np.where(data_rawq['ibq_l3'] > data_rawq['ibq_l4'], 1, 0)
data_rawq['nincr_temp5'] = np.where(data_rawq['ibq_l4'] > data_rawq['ibq_l5'], 1, 0)
data_rawq['nincr_temp6'] = np.where(data_rawq['ibq_l5'] > data_rawq['ibq_l6'], 1, 0)
data_rawq['nincr_temp7'] = np.where(data_rawq['ibq_l6'] > data_rawq['ibq_l7'], 1, 0)
data_rawq['nincr_temp8'] = np.where(data_rawq['ibq_l7'] > data_rawq['ibq_l8'], 1, 0)

data_rawq['nincr'] = (data_rawq['nincr_temp1']
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']*data_rawq['nincr_temp4'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']*data_rawq['nincr_temp4']*data_rawq['nincr_temp5'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']*data_rawq['nincr_temp4']*data_rawq['nincr_temp5']*data_rawq['nincr_temp6'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']*data_rawq['nincr_temp4']*data_rawq['nincr_temp5']*data_rawq['nincr_temp6']*data_rawq['nincr_temp7'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']*data_rawq['nincr_temp4']*data_rawq['nincr_temp5']*data_rawq['nincr_temp6']*data_rawq['nincr_temp7']*data_rawq['nincr_temp8']))

data_rawq = data_rawq.drop(['ibq_l1', 'ibq_l2', 'ibq_l3', 'ibq_l4', 'ibq_l5', 'ibq_l6', 'ibq_l7', 'ibq_l8', 'nincr_temp1',
                            'nincr_temp2', 'nincr_temp3', 'nincr_temp4', 'nincr_temp5', 'nincr_temp6', 'nincr_temp7',
                            'nincr_temp8'], axis=1)




#%% 65
# pctacc
condlist = [data_rawq['npq'].isnull(),
            data_rawq['actq'].isnull() | data_rawq['lctq'].isnull()]
choicelist = [((data_rawq['actq']-data_rawq['lctq'])-(data_rawq['actq_l4']-data_rawq['lctq_l4']))/abs(ttm4('ibq', data_rawq)), np.nan]
data_rawq['pctacc'] = np.select(condlist, choicelist,
                              default=((data_rawq['actq']-data_rawq['lctq']+data_rawq['npq'])-(data_rawq['actq_l4']-data_rawq['lctq_l4']+data_rawq['npq_l4']))/
                                      abs(ttm4('ibq', data_rawq)))

#%% 69
# rd
data_rawq['xrdq4'] = ttm4('xrdq', data_rawq)
data_rawq['xrdq4'] = np.where(data_rawq['xrdq4'].isnull(), data_rawq['xrdy'], data_rawq['xrdq4'])

data_rawq['xrdq4/atq_l4'] = data_rawq['xrdq4']/data_rawq['atq_l4']
data_rawq['xrdq4/atq_l4_l4'] = data_rawq.groupby(['permno'])['xrdq4/atq_l4'].shift(4)
data_rawq['rd'] = np.where(((data_rawq['xrdq4']/data_rawq['atq'])-data_rawq['xrdq4/atq_l4_l4'])/data_rawq['xrdq4/atq_l4_l4']>0.05, 1, 0)


#%% 71
# rd_sale
data_rawq['rd_sale'] = data_rawq['xrdq4']/data_rawq['saleq4']

#%% 78
# rsup
data_rawq['saleq_l4'] = data_rawq.groupby(['permno'])['saleq'].shift(4)
data_rawq['rsup'] = (data_rawq['saleq'] - data_rawq['saleq_l4'])/data_rawq['me']




#%% 86
# sp
data_rawq['sp'] = data_rawq['saleq4']/data_rawq['me']




#%% 89
def chars_std(start, end, df, chars):
    """

    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :param chars: lag chars
    :return: std of factor
    """
    lag = pd.DataFrame()
    lag_list = []
    for i in range(start, end):
        lag['chars_l%s' % i] = df.groupby(['permno'])['%s' % chars].shift(i)
        lag_list.append('chars_l%s' % i)
    result = lag[lag_list].std(axis=1)
    return result

data_rawq['stdacc'] = chars_std(0, 16, data_rawq, 'sacc')




#%% 90
# stdcf
data_rawq['scf'] = (data_rawq['ibq']/data_rawq['saleq']) - data_rawq['sacc']
data_rawq['scf'] = np.where(data_rawq['saleq']<=0, (data_rawq['ibq']/0.01) - data_rawq['sacc'], data_rawq['sacc'])

data_rawq['stdcf'] = chars_std(0, 16, data_rawq, 'scf')
















#%% Momentum 

#######################################################################################################################
#                                                       Momentum                                                      #
#######################################################################################################################
crsp_mom = conn.raw_sql("""
                        select permno, date, ret, retx, prc, shrout, vol
                        from crsp.msf
                        where date >= '01/01/1959'
                        """)


crsp_mom['permno'] = crsp_mom['permno'].astype(int)
crsp_mom['date'] = pd.to_datetime(crsp_mom['date'])
crsp_mom['jdate'] = pd.to_datetime(crsp_mom['date']) + MonthEnd(0)
crsp_mom = crsp_mom.dropna(subset=['ret', 'retx', 'prc'])

# add delisting return
dlret = conn.raw_sql("""
                     select permno, dlret, dlstdt 
                     from crsp.msedelist
                     """)

dlret.permno = dlret.permno.astype(int)
dlret['dlstdt'] = pd.to_datetime(dlret['dlstdt'])
dlret['jdate'] = dlret['dlstdt'] + MonthEnd(0)

# merge delisting return to crsp return
crsp_mom = pd.merge(crsp_mom, dlret, how='left', on=['permno', 'jdate'])
crsp_mom['dlret'] = crsp_mom['dlret'].fillna(0)
crsp_mom['ret'] = crsp_mom['ret'].fillna(0)
crsp_mom['retadj'] = (1 + crsp_mom['ret']) * (1 + crsp_mom['dlret']) - 1
crsp_mom['me'] = crsp_mom['prc'].abs() * crsp_mom['shrout']  # calculate market equity
crsp_mom['retx'] = np.where(crsp_mom['me'].isnull(), 0, crsp_mom['retx'])
crsp_mom = crsp_mom.drop(['dlret', 'dlstdt'], axis=1)#delete prc,shrout


#Seasonality

#Rla
crsp_mom['rla'] = crsp_mom.groupby(['permno'])['ret'].shift(12)

#Rln
lag = pd.DataFrame()
result = 0
for i in range(1, 12):
    lag['mom%s' % i] = crsp_mom.groupby(['permno'])['ret'].shift(i)
    result = result + lag['mom%s' % i]
crsp_mom['rln'] = result/11

#R[2,5]a
#R[2,5]n
lag = pd.DataFrame()
result = 0
for i in range(13,61):
    lag['mom%s' % i] = crsp_mom.groupby(['permno'])['ret'].shift(i)
    if i not in [24,36,48,60]:
        result = result + lag['mom%s' % i]

crsp_mom['r25a'] = (lag['mom24']+lag['mom36']+lag['mom48']+lag['mom60'])/4
crsp_mom['r25n'] = result/44



def mom(start, end, df):
    """
    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result


# 46
crsp_mom['mom12m'] = mom(1, 12, crsp_mom)

# 47
crsp_mom['mom1m'] = crsp_mom['ret']
# 49
crsp_mom['mom6m'] = mom(1, 6, crsp_mom)
# 48
crsp_mom['mom36m'] = mom(1, 36, crsp_mom)

#%% 93 turn
crsp_mom['vol_l1'] = crsp_mom.groupby(['permno'])['vol'].shift(1)
crsp_mom['vol_l2'] = crsp_mom.groupby(['permno'])['vol'].shift(2)
crsp_mom['vol_l3'] = crsp_mom.groupby(['permno'])['vol'].shift(3)
crsp_mom['prc_l2'] = crsp_mom.groupby(['permno'])['prc'].shift(2)
crsp_mom['dolvol'] = np.log(crsp_mom['vol_l2']*crsp_mom['prc_l2']).replace([np.inf, -np.inf], np.nan)
crsp_mom['turn'] = ((crsp_mom['vol_l1']+crsp_mom['vol_l2']+crsp_mom['vol_l3'])/3)/crsp_mom['shrout']

#%% 29
# dolvol
crsp_mom['dolvol'] = np.log(crsp_mom['vol_l2']*crsp_mom['prc_l2']).replace([np.inf, -np.inf], np.nan)



























#%% 
wrds_db = wrds.Connection(wrds_username='sijia_yin') 
data = wrds_db.raw_sql("""
    data_a AS(
    SELECT 
    data.jdate, 
    CASE WHEN LEAD(data.jdate) OVER w1 is null
        THEN 
            data.jdate + interval '1 year'
        ELSE 
            LEAD(data.jdate) OVER w1
        END 
            as jdate_end,
    data.fyear, data.sic2,
    data.gvkey, data.permno, data.roa, data.cfroa, data.oancf, 
    data.ni, data.xrdint, data.capxint, data.xadint,
    data.absacc, data.acc, data.age, data.agr, data.bm, data.cashdebt,
    data.cashpr, data.cfp, data.chato, data.chcsho, data.chinv, 
    data.chpm, data.convind, 
    data.currat, data.depr, data.divi, data.divo, data.dy, data.egr, 
    data.ep, data.gma, data.grcapx, data.grltnoa, data.hire, 
    data.invest, data.tb_1, data.mve_f, data.lev, data.lgr, 
    data.operprof, data.cpi, data.xsga, data.pchcapx, data.avgat, 
    data.pchcurrat, data.pchdepr, data.pchgm_pchsale, data.pchquick, 
    data.pchsale_pchinvt, data.pchsale_pchrect, data.pchsale_pchxsga, 
    data.pchsaleinv, data.pctacc, data.ps, data.quick, data.rd, 
    data.rd_mve, data.rd_sale, data.realestate, data.roic, data.sale,
    data.salecash, data.saleinv, data.salerec, data.secured, 
    data.securedind, data.sgr, data.sin, data.sp, data.tang
    

    /*get consolidated, standardized, industrial format statements*/
    where data.indfmt = 'INDL' 
    and data.datafmt = 'STD'
    and data.popsrc = 'D'
    and data.consol = 'C'
    and data.datadate >= '01/01/1959'
    
    """)
    
