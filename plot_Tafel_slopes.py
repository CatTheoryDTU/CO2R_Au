from plot_paper_data import load_pkl_data, _compute_partial_jecsa
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

CRED = '\033[91m'
CEND = '\033[0m'


def main():
    folders = ["Bertheussen_COR", "Bertheussen_COR-pcCu", "Huang_CO2R",
               "Kanan_CO2R-ODCu", "Kanan_COR-ODCu", "Kuhl_CO2", "Wang_COR",
               "Wang_COR-Cuflower", "Raciti_COR",
               "Jouny_COR", "Luc_COR", "Ma_CO2R", "Gregorio_CO2R", "Sargent_CO2R-CuN-C", "Zuettel_CO2R", "Kanan_COR-GDE"]
#    folders = ["Bertheussen_COR", "Bertheussen_COR-pcCu", "Huang_CO2R",
#             "Kuhl_CO2", "Wang_COR", "Wang_COR-Cuflower", "Raciti_COR",
#            "Jouny_COR", "Luc_COR", "Ma_CO2R", "Gregorio_CO2R"]
    products = ['Methane']

#    perform_Tafel_analysis(folders, products)
    perform_Tafel_analysis(folders, products, mode='COR')
#    perform_Tafel_analysis(folders, products, mode='CO2R')


def perform_Tafel_analysis(folders, products, maxpH=100, outdir='output',
                           min_datapoints=3, max_standard_deviation=0.95,
                           mode=None, potscale='SHE'):
    """
    Main function for making the plots
    """

    data = load_pkl_data(folders)

    for totads in products:
      print('Tafel slopes for %s in studies:' % totads)
      tot_all_data, Tafdata, Tafred = {}, [], []
      for ads in totads.split('+'):
        a_dat = _compute_partial_jecsa(data, adsorbate=ads, maxpH=maxpH)
        colors = plt.cm.jet(np.linspace(0, 1, len(a_dat.keys())))

        for istudy, study in enumerate(a_dat.keys()):
            if mode and a_dat[study]['mode'] != mode:
                continue

            if study not in tot_all_data.keys():
                tot_all_data[study] = []

            if not all(par in a_dat[study].keys()
                       for par in ['U_'+potscale, 'j_partial']):
                print(CRED + 'Couldnt find U_%s and j_partial in %s'
                      % (potscale, study) + CEND)
                continue

            this_reddata, this_all_data, coeff = \
                preselect_datapoints(a_dat[study],
                                     max_standard_deviation,
                                     potscale)

            if ads == 'Methane':
                if a_dat[study]['mode'] == 'CO2R':
                    pH = 6.8
                elif a_dat[study]['mode'] == 'COR':
                    pH = 13
                this_reddata[:, 1] += pH
                this_all_data[:, 1] += pH

            if len(totads.split('+')) == 1:
                print('%s: %1.2f mV/dec' % (study, 1000/coeff))

                if len(this_reddata) > min_datapoints:
                    Tafred.extend([pot_j for pot_j in this_reddata])
                Tafdata.extend([pot_j for pot_j in this_all_data])

                plt.plot(this_all_data[:, 0], this_all_data[:, 1], 'o',
                         markerfacecolor='w', markeredgecolor=colors[istudy])
                plt.plot(this_reddata[:, 0], this_reddata[:, 1], 'o',
                         color=colors[istudy], label='%s: %1.1f mV/dec'
                         % (study, 1000/coeff))
            else:
                tot_all_data[study].extend([pot_j for pot_j in this_all_data])

      if len(totads.split('+')) > 1:
        final_all_data, final_red_data = {}, {}
        colors = plt.cm.jet(np.linspace(0, 1, len(tot_all_data.keys())))
        for istudy, study in enumerate(tot_all_data.keys()):
            tot_all_data[study] = np.array(tot_all_data[study])
            final_all_data[study] = []

            for pot in np.unique(tot_all_data[study][:, 0]):
              js_at_pot = (np.where(tot_all_data[study][:, 0] == pot)[0])
              final_all_data[study].append([pot, np.log10(np.sum(
                  10**tot_all_data[study][js_at_pot, 1]))])

            final_all_data[study] = np.array(final_all_data[study])

            final_red_data[study], this_all_data, coeff = \
                preselect_datapoints(final_all_data[study],
                                     max_standard_deviation,
                                     potscale)

            print('%s: %1.2f mV/dec' % (study, 1000/coeff))

            plt.plot(final_all_data[study][:, 0],
                     final_all_data[study][:, 1],
                     'o', markerfacecolor='w',
                     markeredgecolor=colors[istudy])

            if len(final_red_data[study]) >= min_datapoints:
                plt.plot(final_red_data[study][:, 0],
                         final_red_data[study][:, 1], 'o',
                         color=colors[istudy],
                         label='%s: %1.1f mV/dec' % (study, 1000/coeff))
                Tafred.extend([pot_j for pot_j in final_red_data[study]])

            Tafdata.extend([pot_j for pot_j in final_all_data[study]])

      finalize_Tafplot(Tafred, Tafdata, totads, potscale, mode)


def finalize_Tafplot(Tafred, Tafdata, ads, potscale, mode=None,
                     outdir='output', maxpH=100):
    Tafred, Tafdata = np.array(Tafred), np.array(Tafdata)
    coeff, dum = curve_fit(lin_fun, Tafred[:, 0], Tafred[:, 1])
    print(CRED+'Overall %s Tafel slope: ' % ads, 1000/coeff[0], CEND)
    plt.plot(Tafred[:, 0], lin_fun(Tafred[:, 0], *coeff), 'k-')
    plt.annotate('%1.1f mV/dec' % (1000/coeff[0]),
                 ((max(Tafred[:, 0])+min(Tafred[:, 0]))/2.,
                 max(Tafred[:, 1])))
    plt.xlabel('U$_{%s}$ [V]' % potscale)
    plt.ylabel('Current density [10$^x$ mA/cm$^2$]')
    plt.legend(bbox_to_anchor=(1, 1),prop={'size':4})
    plt.title(ads)
    plt.tight_layout()
    if mode:
        plt.savefig(outdir+'/Tafel_analysis_%s_maxpH%s_U_%s_%s.pdf'
                    % (ads, maxpH, potscale, mode))
    else:
        plt.savefig(outdir+'/Tafel_analysis_%s_maxpH%s_U_%s.pdf'
                    % (ads, maxpH, potscale))
    plt.close()


def preselect_datapoints(dat, max_standard_deviation, potscale):
    Tafdata = []
    if isinstance(dat, dict):
        for ipot, pot in enumerate(dat['U_'+potscale]):
            Tafdata.append([pot[0], np.log10(dat['j_partial'][ipot][0])])
        Tafdata = np.array(Tafdata)
    else:
        Tafdata = dat

    Tafdata = Tafdata[np.argsort(Tafdata[:, 0])]
    Tafred = detect_mass_trans(Tafdata, max_standard_deviation)
    coeff, dum = curve_fit(lin_fun, Tafred[:, 0], Tafred[:, 1])

    return Tafred, Tafdata, coeff[0]


def detect_mass_trans(Tafdata, max_standard_deviation=0.95):
    counter, r_squared = 0, 0
    while r_squared - max_standard_deviation < 0:
        Tafred = Tafdata.copy()
        if counter:
            Tafred = Tafred[counter:]

        coeff, dum = curve_fit(lin_fun, Tafred[:, 0], Tafred[:, 1])
        residuals = Tafred[:, 1] - lin_fun(Tafred[:, 0], *coeff)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Tafred[:, 1]-np.mean(Tafred[:, 1]))**2)
        r_squared = 1 - (ss_res / ss_tot)
        counter += 1
    return Tafred


def lin_fun(x, a, b):
    return a*x+b


if __name__ == "__main__":
    main()
