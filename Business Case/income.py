import numpy as np
totalmw = 100
crore_rupee = (10**9) / 100
rup_usd = 0.014
usd_eur = 0.82
totalkwh = (100*10**3) * (24*365.25)
rup_billion_euro = rup_usd * usd_eur / (10 ** 9)
#revenue
tamil_nadu_acos = 6.69 #average cost of supply/kwh
keral_acos = 5.43
karnataka_acos = 6.11
andhra_pradesh_acos = 7.61
acos_list = np.array((tamil_nadu_acos,keral_acos,karnataka_acos,andhra_pradesh_acos))
revenue_yearly_billion = acos_list*totalkwh*rup_billion_euro*1000
#subsidy
pv_subsidy_mw = 1.25 #crore/mw
pv_subsidy_ik = pv_subsidy_mw * totalmw *crore_rupee * rup_usd * usd_eur 
#investment