"""
Simple plot script for wave LCOE
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

fps = ['./atlantic_rm5/atlantic_rm5_agg.csv',
       './pacific_rm5/pacific_rm5_agg.csv',
       ]

for fp in fps:
    df = pd.read_csv(fp)
    a = plt.scatter(df.longitude, df.latitude, c=df["mean_lcoe"],
                    s=0.5, vmin=0, vmax=1500)
    plt.colorbar(a, label='lcoe_fcr ($/MWh)')
    tag = os.path.basename(fp).replace('_agg.csv', '')
    fp_out = './lcoe_fcr_{}.png'.format(tag)
    plt.savefig(fp_out, dpi=300)
    plt.close()
