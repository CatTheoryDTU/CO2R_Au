#!/usr/bin/env python

import sys, os
import numpy as np
from scipy.optimize import curve_fit, leastsq
from copy import deepcopy
import pickle

import matplotlib.pyplot as plt
#from rtools.helpers.matplotlibhelpers import tumcolors as tumcs
#import rtools.helpers.matplotlibhelpers as matplotlibhelpers
from scipy.constants import golden_ratio, inch
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

CRED = '\033[91m'
CEND = '\033[0m'

def _set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5/2,\
                    lrbt=[0.135,0.80,0.25,0.95],fsize=9.0,font='helvetica'):
    # set plot geometry	
    plt.style.use(['science','ieee'])
    rcParams['figure.figsize'] = (width, height) # x,y
    rcParams['font.size'] = fsize
    rcParams['figure.subplot.left'] = lrbt[0]  # the left side of the subplots of the figure
    rcParams['figure.subplot.right'] = lrbt[1] #0.965 # the right side of the subplots of the figure
    rcParams['figure.subplot.bottom'] = lrbt[2] # the bottom of the subplots of the figure
    rcParams['figure.subplot.top'] = lrbt[3] # the bottom of the subplots of the figure

    rcParams['xtick.top'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.right'] = True
    rcParams['ytick.direction'] = 'in'

    rcParams['legend.fancybox'] = False
    #rcParams['legend.framealpha'] = 1.0
    rcParams['legend.edgecolor'] = 'k'
    if font != 'None':
        matplotlibhelpers.set_latex(rcParams,font=font) #poster

def _compute_partial_jecsa(data, adsorbate, maxpH=100.):
    data = deepcopy(data)
    out = {}
    for k in data:
        if data[k]['roughness'] != 'None' and adsorbate in data[k]:
            # transform total current into j-ecsa
            j_ecsa = data[k]['I(mA/cm2)'] / float(data[k]['roughness'])
            # multiply with FE for chosen product
            pj = np.array([data[k][adsorbate][i,:]*j_ecsa[i] for i in range(len(j_ecsa))])
            ind = np.where(pj != 0.0)[0]
            pj = np.absolute(pj[ind,:]) # correct for current definition
            urhe = data[k]['V_RHE'][ind,:]
            ushe = deepcopy(urhe)
            ushe[:,0] -= 0.059 * float(data[k]['pH'])
            if float(data[k]['pH']) <= maxpH:
                out.update({k:{'U_RHE':urhe, 'U_SHE':ushe, 'j_partial':pj, 'cell':data[k]['cell'], 'mode':data[k]['mode'], 'pH':data[k]['pH']}})
        else:
            print("%s does not contain roughness or %s"%(k, adsorbate))
    return(out)


def _load_pickle_file(filename,py23=False):
    pickle_file = open(filename,'rb')
    if py23:
        data = pickle.load(pickle_file, encoding='latin1')
    else:
        data = pickle.load(pickle_file)
    pickle_file.close()
    return(data)

def _write_pickle_file(filename,data,py2=False):
    ptcl = 0
    if py2:
        filename = filename.split('.')[0]+'_py2.'+filename.split('.')[1]
        ptcl = 2
    output = open(filename, 'wb')
    pickle.dump(data, output, protocol=ptcl)
    output.close()

def plot_partial_current_densities(filename, data, pot, clr='data'):
    _set_plotting_env(width=3.37*1.3,height=3.37,\
                   lrbt=[0.15,0.75,0.13,0.98],fsize=7.0,font='None')

   #vcolors = [tumcs['tumorange'],tumcs['tumgreen'],tumcs['tumblue'],\
   #    tumcs['tumlightblue'],tumcs['acc_yellow'],tumcs['diag_violet']]
    if clr == 'data':
        ks = list(data.keys())
    elif clr == 'pH':
        ks = (np.unique([float(data[k]['pH']) for k in data])).tolist()
    colors = plt.cm.jet(np.linspace(0,1,len(ks)))
    kclr = {ks[i]:colors[i] for i in range(len(ks))}
    lss = {'COR':'-', 'CO2R':'--'}
    mks = {'H-cell':'o', 'GDE':'x'}


    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for k in data:
        x1 = data[k][pot][:,0]; xerr1 = data[k][pot][:,1]
        #x2 = data[k]['U_RHE'][:,0]; xerr2 = data[k]['U_RHE'][:,1]
        if np.all(xerr1 == 0.0):
            xerr1 = None
       #if np.all(xerr2 == 0.0):
       #    xerr2 = None
        kc = k
        if clr == 'pH':
            kc = float(data[k]['pH'])
        
        y = data[k]['j_partial'][:,0]; yerr = data[k]['j_partial'][:,1]
        if np.all(yerr == 0.0):
            yerr = None
        #if data[k]['cell'] == 'GDE' and data[k]['mode'] == 'COR':
        if True:
            ax1.errorbar(x1, y, xerr=xerr1, yerr=yerr, color=kclr[kc], ls=lss[data[k]['mode']], marker=mks[data[k]['cell']], markersize=3)
            ax2.errorbar(x1, y, xerr=xerr1, yerr=yerr, color=kclr[kc], ls=lss[data[k]['mode']], marker=mks[data[k]['cell']], markersize=3)

    # legend
    for l in lss:
        ax1.plot(np.nan, np.nan, color='k', ls=lss[l], label=r'%s'%l)
    for m in mks:
        ax1.plot(np.nan, np.nan, color='k', ls='None', marker=mks[m], label=r'%s'%m)
    for k in ks:
        ax1.plot(np.nan, np.nan, color=kclr[k], label=r'%s'%k)
    ax1.legend(loc=1,prop={'size':4},bbox_to_anchor=(1.45, 1.))
    ax2.set_zorder(-1)
    #ax2.set_ylim(1e-5,10)
    ax2.set_ylim(1e-5,100)

    # axis labels
    ax1.set_xticklabels([])
    ax2.set_yscale('log')
    ax2.set_xlabel('U vs. %s (V)'%pot.split('_')[1])
    ax2.set_ylabel('$j_{\mathrm{ECSA}}$ (mA/cm$^2$)')
    ax2.yaxis.set_label_coords(-0.15, 1.05)
    plt.subplots_adjust(hspace=0.05)
    #plt.show()
    #matplotlibhelpers.write(filename+'_'+pot,folder='output',\
#        write_info=False,write_png=False,write_pdf=True,write_eps=False)
    writefig(filename+'_'+pot,folder='output')


def writefig(filename, folder='output',  write_eps=False):
    """
      wrapper for creating figures
      Parameters
      ----------
      filename : string
        name of the produced figure (without extention)
      folder : string
        subfolder in which to write the figure (default = output)
      write_eps : bool
        whether to create an eps figure (+ the usual pdf)
    """
    # folder for output
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fileloc = os.path.join(folder, filename)
    print("writing {}".format(fileloc+'.pdf'))
    plt.savefig(fileloc+'.pdf')
    if write_eps:
        print("writing {}".format(fileloc+'.eps'))
        plt.savefig(fileloc+'.eps')
    plt.close(plt.gcf())

def create_input_pkl(folder):
    scripts = [ script for script in os.listdir(folder)
            if script[-3:] == '.py']
    basedir=os.getcwd()
    if len(scripts) > 1:
        print(CRED+"Careful! There's more than one python script in %s"%folder+CEND)
    os.chdir(basedir+'/'+folder)
    os.system('python3 %s'%scripts[0])
    os.chdir(basedir)

def load_pkl_data(folders,data=None):
    if data is None:
        data={}
    # collect all data
    for f in folders:
        pklfile = "%s/%s.pkl"%(f,f)
        if pklfile.split('/')[-1] not in os.listdir(f):
            print('\033[92mCouldnt find input pkl in %s, creating it\033[0m'%f)
            create_input_pkl(f)

        dat = _load_pickle_file(pklfile)

        tag = f.split('_')[0][:5]
        for k in dat:
            if k != 'Cu_oh': # faulty potential
                data.update({tag+'-'+k:dat[k]})
    return data

if __name__ == "__main__":
    folders = ["Bertheussen_COR", "Bertheussen_COR-pcCu", "Huang_CO2R", 
        "Kanan_CO2R-ODCu", "Kanan_COR-ODCu", "Kuhl_CO2", "Wang_COR", "Wang_COR-Cuflower", "Raciti_COR",
        "Jouny_COR", "Luc_COR", "Ma_CO2R", "Gregorio_CO2R", "Sargent_CO2R-CuN-C", "Zuettel_CO2R", "Kanan_COR-GDE"]

    # collect all data
    data = load_pkl_data(folders)
    maxpH = 100.
    for ads in ['Acetate', 'Acetaldehyde', 'Ethanol', 'Ethylene', 'Hydrogen', 'Methane']:
        a_dat = _compute_partial_jecsa(data, adsorbate=ads, maxpH=maxpH)
        plot_partial_current_densities('partial_j_%s_maxpH%.0f'%(ads,maxpH), a_dat, pot='U_SHE')
        #plot_partial_current_densities('partial_j_pHclr_%s'%(ads), a_dat, pot='U_SHE', clr='pH')
        #plot_partial_current_densities('partial_j_%s_maxpH%.0f'%(ads,maxpH), a_dat, pot='U_RHE')

