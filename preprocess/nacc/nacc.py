import pandas as pd
import numpy as np
import os
import re
from icecream import ic
import toml
import warnings
warnings.filterwarnings("ignore")

def features_conversion(df):

    # Form Header and Milestones Form
    df['ID'] = df['NACCID']

    # FORM A1: Subject Demographics
    df['his_NACCREAS'] = df['NACCREAS'].replace({1: 0, 2: 1, 7: 2, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCREFR'] = df['NACCREFR'].replace({1: 0, 2: 1, 8: 2, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_BIRTHMO'] = df['BIRTHMO'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_BIRTHYR'] = df['BIRTHYR'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_SEX'] = df['SEX'].replace({1: 'male', 2: 'female', -4: np.NaN, '-4':np.NaN})
    df['his_HISPANIC'] = df['HISPANIC'].replace({0: 'no', 1: 'yes', 9:np.NaN, -4: np.NaN, '-4':np.NaN})
    df['his_HISPOR'] = df['HISPOR'].replace({50: 7, 88: 0, 99: np.NaN, -4: np.NaN, '-4':np.NaN})
    df['his_HISPORX'] = df['HISPORX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_RACE'] = df['RACE'].replace({1:'whi', 2:'blk', 3:'ind', 4:'haw', 5:'asi', 50:'oth', 99:np.NaN, -4: np.NaN, '-4':np.NaN})
    df['his_RACEX'] = df['RACEX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_RACESEC'] = df['RACESEC'].replace({1:'whi', 2:'blk', 3:'ind', 4:'haw', 5:'asi', 50:'oth', 88:np.NaN, 99:np.NaN, -4: np.NaN, '-4':np.NaN})
    df['his_RACESECX'] = df['RACESECX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_RACETER'] = df['RACETER'].replace({1:'whi', 2:'blk', 3:'ind', 4:'haw', 5:'asi', 50:'oth', 88:np.NaN, 99:np.NaN, -4: np.NaN, '-4':np.NaN})
    df['his_RACETERX'] = df['RACETERX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_PRIMLANG'] = df['PRIMLANG'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PRIMLANX'] = df['PRIMLANX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_EDUC'] = df['EDUC'].replace({99: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_MARISTAT'] = df['MARISTAT'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_LIVSIT'] = df['NACCLIVS'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_INDEPEND'] = df['INDEPEND'].replace({1: 0, 2: 1, 3: 2, 4: 3, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_RESIDENC'] = df['RESIDENC'].replace({1: 0, 2: 1, 3: 2, 4: 3, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_HANDED'] = df['HANDED'].replace({1: 0, 2: 1, 3: 2, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCAGE'] = df['NACCAGE'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCAGEB'] = df['NACCAGEB'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCNIHR'] = df['NACCNIHR'].replace({1:'whi', 2:'blk', 3:'ind', 4:'haw', 5:'asi', 6:'mul', 99:np.NaN, -4: np.NaN, '-4':np.NaN})

    # APOE
    df['apoe_NACCNE4S'] = df['NACCNE4S'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM A2: Co-participant Demographics
    # df['his_INRELTO'] = df['INRELTO'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM A3: Subject Family History
    df['his_NACCFAM'] = df['NACCFAM'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCMOM'] = df['NACCMOM'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCDAD'] = df['NACCDAD'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCFADM'] = df['NACCFADM'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCAM'] = df['NACCAM'].replace({8: 4, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCAMX'] = df['NACCAMX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_NACCAMS'] = df['NACCAMS'].replace({1: 0, 2: 1, 3: 2, 8: 3, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCAMSX'] = df['NACCAMSX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_NACCFFTD'] = df['NACCFFTD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCFM'] = df['NACCFM'].replace({8: 5, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCFMX'] = df['NACCFMX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_NACCFMS'] = df['NACCFMS'].replace({1: 0, 2: 1, 3: 2, 8: 3, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCFMSX'] = df['NACCFMSX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_NACCOM'] = df['NACCOM'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCOMX'] = df['NACCOMX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_NACCOMS'] = df['NACCOMS'].replace({1: 0, 2: 1, 3: 2, 8: 3, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCOMSX'] = df['NACCOMSX'].replace({-4: np.NaN, '-4':np.NaN})

    # FORM A4: Subject Medications
    df['med_ANYMEDS'] = df['ANYMEDS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCAMD'] = df['NACCAMD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCAHTN'] = df['NACCAHTN'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCHTNC'] = df['NACCHTNC'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCACEI'] = df['NACCACEI'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCAAAS'] = df['NACCAAAS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCBETA'] = df['NACCBETA'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCCCBS'] = df['NACCCCBS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCDIUR'] = df['NACCDIUR'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCVASD'] = df['NACCVASD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCANGI'] = df['NACCANGI'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCLIPL'] = df['NACCLIPL'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCNSD'] = df['NACCNSD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCAC'] = df['NACCAC'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCADEP'] = df['NACCADEP'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCAPSY'] = df['NACCAPSY'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCAANX'] = df['NACCAANX'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCADMD'] = df['NACCADMD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCPDMD'] = df['NACCPDMD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCEMD'] = df['NACCEMD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCEPMD'] = df['NACCEPMD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['med_NACCDBMD'] = df['NACCDBMD'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM A5: Subject Health History
    df['his_TOBAC30'] = df['TOBAC30'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TOBAC100'] = df['TOBAC100'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_SMOKYRS'] = df['SMOKYRS'].replace({88: np.NaN, 99: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PACKSPER'] = df['PACKSPER'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_QUITSMOK'] = df['QUITSMOK'].replace({888: np.NaN, 999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ALCOCCAS'] = df['ALCOCCAS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ALCFREQ'] = df['ALCFREQ'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVHATT'] = df['CVHATT'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_HATTMULT'] = df['HATTMULT'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_HATTYEAR'] = df['HATTYEAR'].replace({8888: np.NaN, 9999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVAFIB'] = df['CVAFIB'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVANGIO'] = df['CVANGIO'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVBYPASS'] = df['CVBYPASS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVPACDEF'] = df['CVPACDEF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVPACE'] = df['CVPACE'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVCHF'] = df['CVCHF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVANGINA'] = df['CVANGINA'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVHVALVE'] = df['CVHVALVE'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVOTHR'] = df['CVOTHR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CVOTHRX'] = df['CVOTHRX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_CBSTROKE'] = df['CBSTROKE'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_STROKMUL'] = df['STROKMUL'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCSTYR'] = df['NACCSTYR'].replace({8888: np.NaN, 9999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_CBTIA'] = df['CBTIA'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TIAMULT'] = df['TIAMULT'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NACCTIYR'] = df['NACCTIYR'].replace({8888: np.NaN, 9999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PD'] = df['PD'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PDYR'] = df['PDYR'].replace({8888: np.NaN, 9999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PDOTHR'] = df['PDOTHR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PDOTHRYR'] = df['PDOTHRYR'].replace({8888: np.NaN, 9999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_SEIZURES'] = df['SEIZURES'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TBI'] = df['TBI'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TBIBRIEF'] = df['TBIBRIEF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TRAUMBRF'] = df['TRAUMBRF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TBIEXTEN'] = df['TBIEXTEN'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TRAUMEXT'] = df['TRAUMEXT'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TBIWOLOS'] = df['TBIWOLOS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TRAUMCHR'] = df['TRAUMCHR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_TBIYEAR'] = df['TBIYEAR'].replace({8888: np.NaN, 9999: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NCOTHR'] = df['NCOTHR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NCOTHRX'] = df['NCOTHRX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_DIABETES'] = df['DIABETES'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_DIABTYPE'] = df['DIABTYPE'].replace({1: 0, 2: 1, 3: 2, 8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_HYPERTEN'] = df['HYPERTEN'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_HYPERCHO'] = df['HYPERCHO'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_B12DEF'] = df['B12DEF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_THYROID'] = df['THYROID'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ARTHRIT'] = df['ARTHRIT'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ARTHTYPE'] = df['ARTHTYPE'].replace({1: 0, 2: 1, 3: 2, 8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ARTHTYPX'] = df['ARTHTYPX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_ARTHUPEX'] = df['ARTHUPEX'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ARTHLOEX'] = df['ARTHLOEX'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ARTHSPIN'] = df['ARTHSPIN'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ARTHUNK'] = df['ARTHUNK'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_INCONTU'] = df['INCONTU'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_INCONTF'] = df['INCONTF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_APNEA'] = df['APNEA'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_RBD'] = df['RBD'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_INSOMN'] = df['INSOMN'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_OTHSLEEP'] = df['OTHSLEEP'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_OTHSLEEX'] = df['OTHSLEEX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_ALCOHOL'] = df['ALCOHOL'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ABUSOTHR'] = df['ABUSOTHR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ABUSX'] = df['ABUSX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_PTSD'] = df['PTSD'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_BIPOLAR'] = df['BIPOLAR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_SCHIZ'] = df['SCHIZ'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_DEP2YRS'] = df['DEP2YRS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_DEPOTHR'] = df['DEPOTHR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_ANXIETY'] = df['ANXIETY'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_OCD'] = df['OCD'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_NPSYDEV'] = df['NPSYDEV'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PSYCDIS'] = df['PSYCDIS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['his_PSYCDISX'] = df['PSYCDISX'].replace({-4: np.NaN, '-4':np.NaN})
    df['his_NACCTBI'] = df['NACCTBI'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM B1: Physical
    df['ph_HEIGHT'] = df['HEIGHT'].replace({88.8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_WEIGHT'] = df['WEIGHT'].replace({888: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_NACCBMI'] = df['NACCBMI'].replace({888.8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_BPSYS'] = df['BPSYS'].replace({777: np.NaN, 888: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    # df['ph_BPDEVICE'] = df['BPDEVICE'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_BPDIAS'] = df['BPDIAS'].replace({777: np.NaN, 888: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_HRATE'] = df['HRATE'].replace({888: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_VISION'] = df['VISION'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_VISCORR'] = df['VISCORR'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_VISWCORR'] = df['VISWCORR'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_HEARING'] = df['HEARING'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_HEARAID'] = df['HEARAID'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['ph_HEARWAID'] = df['HEARWAID'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM B2: HIS and CVD
    df['cvd_ABRUPT'] = df['ABRUPT'].replace({2: 1, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_STEPWISE'] = df['STEPWISE'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_SOMATIC'] = df['SOMATIC'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_EMOT'] = df['EMOT'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_HXHYPER'] = df['HXHYPER'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_HXSTROKE'] = df['HXSTROKE'].replace({2: 1, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_FOCLSYM'] = df['FOCLSYM'].replace({2: 1, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_FOCLSIGN'] = df['FOCLSIGN'].replace({2: 1, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_HACHIN'] = df['HACHIN'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDCOG'] = df['CVDCOG'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_STROKCOG'] = df['STROKCOG'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDIMAG'] = df['CVDIMAG'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDIMAG1'] = df['CVDIMAG1'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDIMAG2'] = df['CVDIMAG2'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDIMAG3'] = df['CVDIMAG3'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDIMAG4'] = df['CVDIMAG4'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['cvd_CVDIMAGX'] = df['CVDIMAGX'].replace({-4: np.NaN, '-4':np.NaN})

    # FORM B3: Unified Parkinson’s Disease Rating Scale (UPDRS)
    df['updrs_PDNORMAL'] = df['PDNORMAL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_SPEECH'] = df['SPEECH'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_SPEECHX'] = df['SPEECHX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_FACEXP'] = df['FACEXP'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_FACEXPX'] = df['FACEXPX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRESTFAC'] = df['TRESTFAC'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRESTFAX'] = df['TRESTFAX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRESTRHD'] = df['TRESTRHD'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRESTRHX'] = df['TRESTRHX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRESTLHD'] = df['TRESTLHD'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRESTLHX'] = df['TRESTLHX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRESTRFT'] = df['TRESTRFT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRESTRFX'] = df['TRESTRFX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRESTLFT'] = df['TRESTLFT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRESTLFX'] = df['TRESTLFX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRACTRHD'] = df['TRACTRHD'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRACTRHX'] = df['TRACTRHX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TRACTLHD'] = df['TRACTLHD'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TRACTLHX'] = df['TRACTLHX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_RIGDNECK'] = df['RIGDNECK'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_RIGDNEX'] = df['RIGDNEX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_RIGDUPRT'] = df['RIGDUPRT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_RIGDUPRX'] = df['RIGDUPRX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_RIGDUPLF'] = df['RIGDUPLF'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_RIGDUPLX'] = df['RIGDUPLX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_RIGDLORT'] = df['RIGDLORT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_RIGDLORX'] = df['RIGDLORX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_RIGDLOLF'] = df['RIGDLOLF'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_RIGDLOLX'] = df['RIGDLOLX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TAPSRT'] = df['TAPSRT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TAPSRTX'] = df['TAPSRTX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_TAPSLF'] = df['TAPSLF'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_TAPSLFX'] = df['TAPSLFX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_HANDMOVR'] = df['HANDMOVR'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_HANDMVRX'] = df['HANDMVRX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_HANDMOVL'] = df['HANDMOVL'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_HANDMVLX'] = df['HANDMVLX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_HANDALTR'] = df['HANDALTR'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_HANDATRX'] = df['HANDATRX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_HANDALTL'] = df['HANDALTL'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_HANDATLX'] = df['HANDATLX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_LEGRT'] = df['LEGRT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_LEGRTX'] = df['LEGRTX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_LEGLF'] = df['LEGLF'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_LEGLFX'] = df['LEGLFX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_ARISING'] = df['ARISING'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_ARISINGX'] = df['ARISINGX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_POSTURE'] = df['POSTURE'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_POSTUREX'] = df['POSTUREX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_GAIT'] = df['GAIT'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_GAITX'] = df['GAITX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_POSSTAB'] = df['POSSTAB'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_POSSTABX'] = df['POSSTABX'].replace({-4: np.NaN, '-4':np.NaN})
    df['updrs_BRADYKIN'] = df['BRADYKIN'].replace({8: 5, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['updrs_BRADYKIX'] = df['BRADYKIX'].replace({-4: np.NaN, '-4':np.NaN})

    # FORM B5: Neuropsychiatric Inventory Questionnaire (NPI-Q)
    df['npiq_NPIQINF'] = df['NPIQINF'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_NPIQINFX'] = df['NPIQINFX'].replace({-4: np.NaN, '-4':np.NaN})
    df['npiq_DEL'] = df['DELSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_HALL'] = df['HALLSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_AGIT'] = df['AGITSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_DEPD'] = df['DEPDSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_ANX'] = df['ANXSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_ELAT'] = df['ELATSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_APA'] = df['APASEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_DISN'] = df['DISNSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_IRR'] = df['IRRSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_MOT'] = df['MOTSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_NITE'] = df['NITESEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['npiq_APP'] = df['APPSEV'].replace({8: 0.0, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM B6: Geriatric Depression Scale (GDS)
    df['gds_NOGDS'] = df['NOGDS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_SATIS'] = df['SATIS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_DROPACT'] = df['DROPACT'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_EMPTY'] = df['EMPTY'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_BORED'] = df['BORED'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_SPIRITS'] = df['SPIRITS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_AFRAID'] = df['AFRAID'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_HAPPY'] = df['HAPPY'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_HELPLESS'] = df['HELPLESS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_STAYHOME'] = df['STAYHOME'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_MEMPROB'] = df['MEMPROB'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_WONDRFUL'] = df['WONDRFUL'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_WRTHLESS'] = df['WRTHLESS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_ENERGY'] = df['ENERGY'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_HOPELESS'] = df['HOPELESS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_BETTER'] = df['BETTER'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['gds_NACCGDS'] = df['NACCGDS'].replace({88: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM B7: NACC Functional Assessment Scale (FAS)
    df['faq_BILLS'] = df['BILLS'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_TAXES'] = df['TAXES'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_SHOPPING'] = df['SHOPPING'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_GAMES'] = df['GAMES'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_STOVE'] = df['STOVE'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_MEALPREP'] = df['MEALPREP'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_EVENTS'] = df['EVENTS'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_PAYATTN'] = df['PAYATTN'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_REMDATES'] = df['REMDATES'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['faq_TRAVEL'] = df['TRAVEL'].replace({8: np.NaN, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)

    # FORM B8: Physical/Neurological Exam Findings
    df['exam_NACCNREX'] = df['NACCNREX'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_NORMEXAM'] = df['NORMEXAM'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_FOCLDEF'] = df['FOCLDEF'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_GAITDIS'] = df['GAITDIS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_EYEMOVE'] = df['EYEMOVE'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_PARKSIGN'] = df['PARKSIGN'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_RESTTRL'] = df['RESTTRL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_RESTTRR'] = df['RESTTRR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_SLOWINGL'] = df['SLOWINGL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_SLOWINGR'] = df['SLOWINGR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_RIGIDL'] = df['RIGIDL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_RIGIDR'] = df['RIGIDR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_BRADY'] = df['BRADY'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_PARKGAIT'] = df['PARKGAIT'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_POSTINST'] = df['POSTINST'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CVDSIGNS'] = df['CVDSIGNS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CORTDEF'] = df['CORTDEF'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_SIVDFIND'] = df['SIVDFIND'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CVDMOTL'] = df['CVDMOTL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CVDMOTR'] = df['CVDMOTR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CORTVISL'] = df['CORTVISL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CORTVISR'] = df['CORTVISR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_SOMATL'] = df['SOMATL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_SOMATR'] = df['SOMATR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_POSTCORT'] = df['POSTCORT'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_PSPCBS'] = df['PSPCBS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_EYEPSP'] = df['EYEPSP'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_DYSPSP'] = df['DYSPSP'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_AXIALPSP'] = df['AXIALPSP'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_GAITPSP'] = df['GAITPSP'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_APRAXSP'] = df['APRAXSP'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_APRAXL'] = df['APRAXL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_APRAXR'] = df['APRAXR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CORTSENL'] = df['CORTSENL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_CORTSENR'] = df['CORTSENR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_ATAXL'] = df['ATAXL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_ATAXR'] = df['ATAXR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_ALIENLML'] = df['ALIENLML'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_ALIENLMR'] = df['ALIENLMR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_DYSTONL'] = df['DYSTONL'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_DYSTONR'] = df['DYSTONR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_MYOCLLT'] = df['MYOCLLT'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_MYOCLRT'] = df['MYOCLRT'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_ALSFIND'] = df['ALSFIND'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_GAITNPH'] = df['GAITNPH'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_OTHNEUR'] = df['OTHNEUR'].replace({8: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['exam_OTHNEURX'] = df['OTHNEURX'].replace({-4: np.NaN, '-4':np.NaN})

    # FORMS C1, C2, C2T: Neuropsychological Battery Summary Scores
    df['bat_MMSECOMP'] = df['MMSECOMP'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MMSELOC'] = df['MMSELOC'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MMSELAN'] = df['MMSELAN'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MMSELANX'] = df['MMSELANX'].replace({-4: np.NaN, '-4':np.NaN})
    df['bat_MMSEVIS'] = df['MMSEVIS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MMSEHEAR'] = df['MMSEHEAR'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MMSEORDA'] = df['MMSEORDA'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MMSEORLO'] = df['MMSEORLO'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_PENTAGON'] = df['PENTAGON'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_NACCMMSE'] = df['NACCMMSE'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_NPSYCLOC'] = df['NPSYCLOC'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_NPSYLAN'] = df['NPSYLAN'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_NPSYLANX'] = df['NPSYLANX'].replace({-4: np.NaN, '-4':np.NaN})
    df['bat_LOGIMO'] = df['LOGIMO'].replace({88: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_LOGIDAY'] = df['LOGIDAY'].replace({88: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_LOGIYR'] = df['LOGIYR'].replace({8888: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_LOGIPREV'] = df['LOGIPREV'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_LOGIMEM'] = df['LOGIMEM'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MEMUNITS'] = df['MEMUNITS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MEMTIME'] = df['MEMTIME'].replace({99: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSBENTC'] = df['UDSBENTC'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSBENTD'] = df['UDSBENTD'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSBENRS'] = df['UDSBENRS'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGIF'] = df['DIGIF'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGIFLEN'] = df['DIGIFLEN'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGIB'] = df['DIGIB'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGIBLEN'] = df['DIGIBLEN'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_ANIMALS'] = df['ANIMALS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_VEG'] = df['VEG'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_TRAILA'] = df['TRAILA'].replace({995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_TRAILARR'] = df['TRAILARR'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_TRAILALI'] = df['TRAILALI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_TRAILB'] = df['TRAILB'].replace({995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_TRAILBRR'] = df['TRAILBRR'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_TRAILBLI'] = df['TRAILBLI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_WAIS'] = df['WAIS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_BOSTON'] = df['BOSTON'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERFC'] = df['UDSVERFC'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERFN'] = df['UDSVERFN'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERNF'] = df['UDSVERNF'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERLC'] = df['UDSVERLC'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERLR'] = df['UDSVERLR'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERLN'] = df['UDSVERLN'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERTN'] = df['UDSVERTN'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERTE'] = df['UDSVERTE'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_UDSVERTI'] = df['UDSVERTI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_COGSTAT'] = df['COGSTAT'].replace({9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MODCOMM'] = df['MODCOMM'].replace({1: 0, 2: 1, 3: 2, 9: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCACOMP'] = df['MOCACOMP'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAREAS'] = df['MOCAREAS'].replace({95.0: 0.0, 96.0: 1.0, 97.0: 2.0, 98.0: 3.0, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCALOC'] = df['MOCALOC'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCALAN'] = df['MOCALAN'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCALANX'] = df['MOCALANX'].replace({-4: np.NaN, '-4':np.NaN})
    df['bat_MOCAVIS'] = df['MOCAVIS'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAHEAR'] = df['MOCAHEAR'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCATOTS'] = df['MOCATOTS'].replace({88: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_NACCMOCA'] = df['NACCMOCA'].replace({88: np.NaN, 99: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCATRAI'] = df['MOCATRAI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCACUBE'] = df['MOCACUBE'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCACLOC'] = df['MOCACLOC'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCACLON'] = df['MOCACLON'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCACLOH'] = df['MOCACLOH'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCANAMI'] = df['MOCANAMI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAREGI'] = df['MOCAREGI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCADIGI'] = df['MOCADIGI'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCALETT'] = df['MOCALETT'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCASER7'] = df['MOCASER7'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAREPE'] = df['MOCAREPE'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAFLUE'] = df['MOCAFLUE'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAABST'] = df['MOCAABST'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCARECN'] = df['MOCARECN'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCARECC'] = df['MOCARECC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCARECR'] = df['MOCARECR'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAORDT'] = df['MOCAORDT'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAORMO'] = df['MOCAORMO'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAORYR'] = df['MOCAORYR'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAORDY'] = df['MOCAORDY'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAORPL'] = df['MOCAORPL'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCAORCT'] = df['MOCAORCT'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_CRAFTVRS'] = df['CRAFTVRS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_CRAFTURS'] = df['CRAFTURS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGFORCT'] = df['DIGFORCT'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGFORSL'] = df['DIGFORSL'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGBACCT'] = df['DIGBACCT'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_DIGBACLS'] = df['DIGBACLS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_CRAFTDVR'] = df['CRAFTDVR'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_CRAFTDRE'] = df['CRAFTDRE'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_CRAFTDTI'] = df['CRAFTDTI'].replace({99: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_CRAFTCUE'] = df['CRAFTCUE'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MINTTOTS'] = df['MINTTOTS'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MINTTOTW'] = df['MINTTOTW'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MINTSCNG'] = df['MINTSCNG'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MINTSCNC'] = df['MINTSCNC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MINTPCNG'] = df['MINTPCNG'].replace({95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MINTPCNC'] = df['MINTPCNC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_MOCBTOTS'] = df['MOCBTOTS'].replace({88: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_NACCMOCB'] = df['NACCMOCB'].replace({88: np.NaN, 99: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY1REC'] = df['REY1REC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY1INT'] = df['REY1INT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY2REC'] = df['REY2REC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY2INT'] = df['REY2INT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY3REC'] = df['REY3REC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY3INT'] = df['REY3INT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY4REC'] = df['REY4REC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY4INT'] = df['REY4INT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY5REC'] = df['REY5REC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY5INT'] = df['REY5INT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY6REC'] = df['REY6REC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REY6INT'] = df['REY6INT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_OTRAILA'] = df['OTRAILA'].replace({888: np.NaN, 995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_OTRLARR'] = df['OTRLARR'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, 888: np.NaN, 995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_OTRLALI'] = df['OTRLALI'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, 888: np.NaN, 995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_OTRAILB'] = df['OTRAILB'].replace({888: np.NaN, 995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_OTRLBRR'] = df['OTRLBRR'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, 888: np.NaN, 995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_OTRLBLI'] = df['OTRLBLI'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, 888: np.NaN, 995: np.NaN, 996: np.NaN, 997: np.NaN, 998: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REYDREC'] = df['REYDREC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REYDINT'] = df['REYDINT'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REYTCOR'] = df['REYTCOR'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_REYFPOS'] = df['REYFPOS'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_VNTTOTW'] = df['VNTTOTW'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_VNTPCNC'] = df['VNTPCNC'].replace({88: np.NaN, 95: np.NaN, 96: np.NaN, 97: np.NaN, 98: np.NaN, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPVAL'] = df['RESPVAL'].replace({1: 0, 2: 1, 3: 2, -4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPHEAR'] = df['RESPHEAR'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPDIST'] = df['RESPDIST'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPINTR'] = df['RESPINTR'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPDISN'] = df['RESPDISN'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPFATG'] = df['RESPFATG'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPEMOT'] = df['RESPEMOT'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPASST'] = df['RESPASST'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPOTH'] = df['RESPOTH'].replace({-4: np.NaN, '-4':np.NaN}).astype(float)
    df['bat_RESPOTHX'] = df['RESPOTHX'].replace({-4: np.NaN, '-4':np.NaN})
    
    # CDR
    df['cdr_CDRGLOB'] = df['CDRGLOB'].replace({-4: np.NaN, '-4':np.NaN})
    df['cdr_CDRSUM'] = df['CDRSUM'].replace({-4: np.NaN, '-4':np.NaN})

    return df

def labels_conversion(row):
    flag = 0
    # NC
    if row['NACCUDSD'] == 1:
        row['NC'] = 1
    else:
        row['NC'] = 0

    # IMCI
    if row['NACCUDSD'] == 2:
        row['IMCI'] = 1
    else:
        row['IMCI'] = 0

    # MCI
    if row['NACCUDSD'] == 3:
        row['MCI'] = 1
    else:
        row['MCI'] = 0
    
    # DE
    if row['NACCUDSD'] == 4:
        row['DE'] = 1
    else:
        row['DE'] = 0 

    # AD
    if ((row['NACCALZP'] == 1) | (row['NACCALZP'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['AD'] = 1
    else:
        row['AD'] = 0

    # LBD
    if ((row['NACCLBDP'] == 1) | (row['NACCLBDP'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['LBD'] = 1
    else:
        row['LBD'] = 0

    # VD
    if (((row['CVDIF'] == 1) | (row['CVDIF'] == 2)) | ((row['VASCIF'] == 1) | (row['VASCIF'] == 2)) | ((row['VASCPSIF'] == 1) | (row['VASCPSIF'] == 2)) | ((row['STROKIF'] == 1) | (row['STROKIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['VD'] = 1
    else:
        row['VD'] = 0

    # Prion disease (CJD, other)
    if ((row['PRIONIF'] == 1) | (row['PRIONIF'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['PRD'] = 1
    else:
        row['PRD'] = 0

    # FTLD and its variants, including CBD and PSP, and with or without ALS
    if (((row['FTLDNOIF'] == 1) | (row['FTLDNOIF'] == 2)) | ((row['FTDIF'] == 1) | (row['FTDIF'] == 2)) | ((row['FTLDMOIF'] == 1) | (row['FTLDMOIF'] == 2)) | ((row['PPAPHIF'] == 1) | (row['PPAPHIF'] == 2)) | ((row['PSPIF'] == 1) | (row['PSPIF'] == 2)) | ((row['CORTIF'] == 1) | (row['CORTIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['FTD'] = 1
    else:
        row['FTD'] = 0

    # NPH
    if ((row['HYCEPHIF'] == 1) | (row['HYCEPHIF'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['NPH'] = 1
    else:
        row['NPH'] = 0

    # Infectious (HIV included), metabolic, substance abuse / alcohol, medications, systemic disease, delirium (Systemic and External Factors)
    if (((row['OTHCOGIF'] == 1) | (row['OTHCOGIF'] == 2)) | ((row['HIVIF'] == 1) | (row['HIVIF'] == 2)) | ((row['ALCDEMIF'] == 1) | (row['ALCDEMIF'] == 2)) | ((row['IMPSUBIF'] == 1) | (row['IMPSUBIF'] == 2)) | ((row['DYSILLIF'] == 1) | (row['DYSILLIF'] == 2)) | ((row['MEDSIF'] == 1) | (row['MEDSIF'] == 2)) | ((row['DELIRIF'] == 1) | (row['DELIRIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['SEF'] = 1
    else:
        row['SEF'] = 0

    
    # Psychiatric including schizophrenia, depression, bipolar, anxiety, PTSD
    if (((row['DEPIF'] == 1) | (row['DEPIF'] == 2)) | ((row['BIPOLDIF'] == 1) | (row['BIPOLDIF'] == 2)) | ((row['SCHIZOIF'] == 1) | (row['SCHIZOIF'] == 2)) | ((row['ANXIETIF'] == 1) | (row['ANXIETIF'] == 2)) | ((row['PTSDDXIF'] == 1) | (row['PTSDDXIF'] == 2)) | ((row['OTHPSYIF'] == 1) | (row['OTHPSYIF'] == 2))) & (row['NACCUDSD'] == 4):
        flag = 1
        row['PSY'] = 1
    else:
        row['PSY'] = 0

    # TBI
    if ((row['BRNINJIF'] == 1) | (row['BRNINJIF'] == 2)) & (row['NACCUDSD'] == 4):
        flag = 1
        row['TBI'] = 1
    else:
        row['TBI'] = 0

    # OTHER
    if  (row['NACCUDSD'] == 4) & ((flag == 0) | ((row['MSAIF'] == 1) | (row['MSAIF'] == 2)) | ((row['ESSTREIF'] == 1) | (row['ESSTREIF'] == 2)) | ((row['DOWNSIF'] == 1) | (row['DOWNSIF'] == 2)) | ((row['HUNTIF'] == 1) | (row['HUNTIF'] == 2)) | ((row['EPILEPIF'] == 1) | (row['EPILEPIF'] == 2)) | ((row['NEOPIF'] == 1) | (row['NEOPIF'] == 2)) | ((row['DEMUNIF'] == 1) | (row['DEMUNIF'] == 2)) | ((row['COGOTHIF'] == 1) | (row['COGOTHIF'] == 2)) | ((row['COGOTH2F'] == 1) | (row['COGOTH2F'] == 2)) | ((row['COGOTH3F'] == 1) | (row['COGOTH3F'] == 2))):
        row['ODE'] = 1
    else:
        row['ODE'] = 0
        
    return row

mri_data_path = '/your_path/NACC/UDS_with_NP_and_Genetics_plus_FTLD_and_LBD_modules/investigator_scan_mriqc_nacc67.csv'
ftlblbd_data_path = '/your_path/NACC/UDS_with_NP_and_Genetics_plus_FTLD_and_LBD_modules/investigator_ftldlbd_nacc67.csv'
# Merge the MRI and FTLBLDB csv files
mri_data = pd.read_csv(mri_data_path)
ftlblbd_data = pd.read_csv(ftlblbd_data_path)
# merged_data = pd.merge(ftlblbd_data, mri_data, on=["NACCID", "NACCVNUM", "NACCNMRI", "NACCMRSA", "NACCADC"], how='left')
merged_data = ftlblbd_data
# Convert the features and labels to the required format
merged_data_var_converted = features_conversion(merged_data)
merged_data_converted = merged_data_var_converted.apply(labels_conversion, axis=1)
merged_data_converted.to_csv('Preprocessed_NACC_DATA.csv', index=False)