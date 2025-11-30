import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astricaltools as at
from astropy.io import fits
#prova a leggere file da due telescopi

file_numbers = range(1, 10)  # da 001 a 009
base_path = "astri_{:03d}_43_009_00002_R_201023_004_0201_SEB.lv0"  # pattern del nome file

dfs = []  # lista per accumulare i DataFrame
print("\n")
for i in file_numbers:
    filename = base_path.format(i)
    print(f"{filename}")

    with fits.open(filename) as hdul:
        data = hdul[1].data

        # Conversione dei campi
        time_ns = data["TIME_NS"].byteswap().newbyteorder().astype(np.int64)
        time_s  = data["TIME_S"].byteswap().newbyteorder().astype(np.int64)
        event   = data["EVTNUM"].byteswap().newbyteorder().astype(np.int64)
        mcrun   = data["MCRUNNUM"].byteswap().newbyteorder().astype(np.int64)
        time_abs = time_s * 10**9 + time_ns  # tempo assoluto in ns
        #creo il dataframe
        df = pd.DataFrame({
            "TIME_NS": time_ns,
            "TIME_S": time_s,
            "TIME_ABS": time_abs,
            "EVENT": event,
            "MCRUN": mcrun,
            "TEL_ID": i  
        })

        dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)
df_all_sorted= df_all.sort_values(by="TIME_ABS")
print(df_all.head())
print(df_all_sorted.head())



#calcolo differenze di tempi
delta_t=np.diff(df_all_sorted["TIME_ABS"])
delta_ts = delta_t[delta_t > 0] #uso maschere e cambio
#print(len(delta_t), len(delta_ts))
#print(delta_t[delta_t <= 0])

# Crea l'istogramma tutti gli eventi
#usando le maschere, non suddivido tra eventi diversi e stesso evento
#escludo gli zeri
plt.figure(figsize=(6, 5))
plt.hist(np.log10(delta_ts), bins=100, color='cornflowerblue', alpha=0.8)
plt.yscale('log')
plt.xlabel(r'$\log_{10}(\Delta t\ \mathrm{[ns]})$', fontsize=12)
plt.ylabel('Events', fontsize=12)
plt.tight_layout()
plt.grid(alpha=0.3)
#plt.show()


#ordino eventi con il for e suddivisione same e different
df_all_sorted=df_all.sort_values(["MCRUN","EVENT","TIME_ABS"]).reset_index(drop=True)
delta_t_same = []
delta_t_diff = []

for i in range(1, len(df_all_sorted)):
    if ((df_all_sorted.loc[i, "MCRUN"]==df_all_sorted.loc[i-1, "MCRUN"])&
        (df_all_sorted.loc[i, "EVENT"] ==df_all_sorted.loc[i-1, "EVENT"])):
        delta_t_same.append(df_all_sorted.loc[i, "TIME_ABS"]-
                            df_all_sorted.loc[i-1, "TIME_ABS"])
    else:
        delta_t_diff.append(df_all_sorted.loc[i, "TIME_ABS"]-
                            df_all_sorted.loc[i-1, "TIME_ABS"])
delta_t_same = np.array(delta_t_same)
delta_t_diff = np.array(delta_t_diff)
delta_t_same_safe = np.where(delta_t_same > 0, delta_t_same, 1)
delta_t_diff_safe = np.where(delta_t_diff > 0, delta_t_diff, 1)
#suddivido tra stesso evento e evento separato sostituendo i deltat con 1
plt.figure(figsize=(6, 5))
plt.hist(np.log10(delta_t_same_safe), bins=80, color='cornflowerblue',
         alpha=0.8,label='same')
plt.hist(np.log10(delta_t_diff_safe), bins=80, color='tomato',
         alpha=0.8,label='different')
plt.yscale('log')
plt.xlabel(r'$\log_{10}(\Delta t\ \mathrm{[ns]})$', fontsize=12)
plt.ylabel('same', fontsize=12)
#plt.xlim(left=0)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.legend()
#plt.show()


#tolgo gli zeri
delta_t_same_saf0 =delta_t_same[delta_t_same > 0]
delta_t_diff_saf0 =delta_t_diff[delta_t_diff > 0]
plt.figure(figsize=(6, 5))
plt.hist(np.log10(delta_t_same_saf0), bins=50, color='cornflowerblue',
         alpha=0.8,label='same')
plt.hist(np.log10(delta_t_diff_saf0), bins=50, color='tomato',
         alpha=0.8,label='different')
plt.yscale('log')
plt.xlabel(r'$\log_{10}(\Delta t\ \mathrm{[ns]})$', fontsize=12)
plt.ylabel('events', fontsize=12)
#plt.xlim(left=0)
plt.tight_layout()
plt.grid(alpha=0.3)
plt.legend()
plt.show()

hdul.close()
