#!/usr/bin/env python3
# -*- coding:  utf-8 -*-
"""
Created on Tue Apr 11 14: 59: 37 2023

@author:  rspirig
"""

def configdir2toml(configpath):
    from util.pathwalker import pathwalker
    configs = pathwalker(configpath, extension=['.config'])
    for config in configs:
        idlconfig2toml(config)


def idlconfig2toml(configfile='MBR5',
                   outfile='same',
                   preservecomments=True,
                   quiet=False):
    # use toml to run the string through
    # note that we preserve porential comments, but that loadtoml won't keep them
    import toml
    import os

    configfile = getconfigfile(configfile)
    assert configfile.endswith('config')

    config = readconfig(configfile)

    if config is False:
        if not quiet:
            print(os.path.exists(configfile))
            print(f'{configfile} could not be read')
        return
    elif isinstance(config, tuple):
        config, comments = config
    else:
        comments = {}

    if outfile == 'same':
        outfile = configfile.replace('.config', '.toml')

    # if we would not like to keep comments (that are at least somewhat helpful)
    # present we could run simply with the below 2 lines ...
    # with open(outfile, 'w') as fo:
    #     toml.dump(config, fo)

    with open(outfile, 'w') as fo:

        for mainsect in config.keys():
            hdr = toml.dumps({mainsect: {}})
            fo.write(f'{hdr}')
            for key, value in config[mainsect].items():
                _dict = toml.dumps({key: value})
                fo.write(f"{_dict}".strip('\n'))
                if mainsect in comments and key in comments[mainsect]:
                    comment = comments[mainsect][key].strip().lstrip('#').replace('#', '\n#')
                    if preservecomments:
                        fo.write(f"\n# {comment}")
                fo.write('\n')
            fo.write('\n')

    return outfile

def getconfigfile(file_or_name='MBR5',
                  ext='.config'):
    import os

    if not file_or_name.endswith(ext):
        file_or_name += ext

    if not os.path.exists(file_or_name):
        # file_or_name = os.path.dirname(__file__) + '/..' + f'/config/radar/{file_or_name}'
        # assuming its in the main repo
        file_or_name = os.path.dirname('./') + f'/config/radar/{file_or_name}'

    return file_or_name

def readconfig(file_or_name='MBR5',
               quiet=True):
    import os
    import toml

    file_or_name_config = getconfigfile(file_or_name)
    if os.path.exists(file_or_name_config):
        file_or_name = file_or_name_config
    else:
        if not quiet:
            print('No .config file could be found for {file_or_name}')
        file_or_name_toml = getconfigfile(file_or_name, ext='.toml')

        if os.path.exists(file_or_name_toml):
            config = toml.load(file_or_name_toml)
            # file_or_name = file_or_name_toml
            return config
        else:
            if not quiet:
                print('No .toml file could be found either')
            return False


    with open(file_or_name) as fo:
        lines = fo.readlines()

    lines = [line.strip().rstrip('#')
             for line in lines if line.strip().rstrip('#')]

    config, comments = {}, {}
    rc = False
    mainkey, subkey = '', ''

    for line in lines:
        line = line.strip().rstrip('#')

        if not line:
            continue

        if '#' in line:
            _line = line
            if _line.startswith('#'):
                rc = True
            else:
                rc = False

            if _line.startswith('#') and mainkey:
                if mainkey not in comments:
                    comments[mainkey] = {}

                if subkey and subkey not in comments[mainkey]:
                    comments[mainkey][subkey] = ''

                # if comments[mainkey][subkey]:
                #     pass
                # else:
                #     comments[mainkey][subkey] += '\n'

                comments[mainkey][subkey] += _line.strip()
                continue
            else:
                line = line.split('#')

                # remove empty spaces between
                line = [i for i in line if i]
                comment = ''.join(line[1:])

            if rc is False:
                line = line[0]
        else:
            comment = ''

        comment = comment.strip()

        if '=' in line:
            key, value = line.split('=')
            mainkey, subkey = key.split('.')
            mainkey, subkey, value = mainkey.strip(), subkey.strip(), value.strip()

        if value:
            if mainkey not in config:
                config[mainkey] = {}

            if subkey not in config[mainkey]:
                config[mainkey][subkey] = {}

            config[mainkey][subkey] = value

        if comment:
            if mainkey not in comments:
                comments[mainkey] = {}

            if subkey not in comments[mainkey]:
                comments[mainkey][subkey] = ''

            # if comments[mainkey][subkey].strip():
            #     pass
            # else:
            #     comments[mainkey][subkey] += '\n'

            comments[mainkey][subkey] += comment.strip()

    if 'spectraprocessing' in config and 'fftwin' in config['spectraprocessing']:
        config['spectraprocessing']['fftwin'] = config['spectraprocessing']['fftwin'].split(
            ':')

    return config, comments


def getradarconfig(radar='MBR5'):
    radars = {'MBR5': {
        'syspar': {
            # Thse 4 characters indicate the beginning of a data chunk in PDS
            # files. This is usefull for reading # corrupted data files.
            # Its a differentcharacter sequence for each cloud MIRA cloud radar.
            "magic4chs": "B5XC",

            # The format of the PDS files was changed greatly with the first
            # scanning MIRA system. This parameter
            "FzkPdsFmt": 1,
            # indicates the file type version. It is 1 for all newer systems
            # and 0 for the old DWD system and NASA.

            # "fzkscanner": 0                  # if set 90 deg is added in
            # mmclx to elv from the pds files to make it compatible with the
            # newer scanners.

            # 512 for systems with 2*500 range gates and 1024 for systems
            # with 2*1000 range gates.
            "mombufsize": 1024,

            # the center frequency range to which the magnetron is adjusted.
            # Since the new rule of the European
            "txfrequency": 35.15 * 10**9,
            # frequency authorities (ECC 166) the systems should be readjusted
            # to 35.15e9 Herz.

            "RecPulseWidths": -1,
            # "RecPulseWidths[ppar.pdr]": Pulse Width is ns. Attention": only the
            # pdr": 1 (~200) ns is tested. The larger settings are for the zspca
            #  files. pdr": 5 is used for 3*~200 ns
            # 100:200:300:400:500:600:700:800:900:1000 for DWD0,FZK,MPI, and NASA
            # 96:192:288:400:512:576:706:832:912:1040  between halo and DLR
            # 112:208:306:400:512:624:706:832:912:1040 for Devices beginning
            # with 1khg pdr from the pds file provides
            # the receiver gate width in ns. Therefore the last setting is only
            # necessary for the IFT radar bfore 09.2014.

            # "dsprangeoffset": 0.0
            # correct offset in range assignment caused by error in dsp
            # software. It should be -90 m for IFT and -30 for FMI and 0 else
            # set to -1 for radars where the falling targets have positive
            # Doppler shifts in the radial velocities in the pds files.

            "dwd_me": -1,
            # set to -1 for radars where the falling targets have positive
            # Doppler shifts in the spectra of the pds files.
            "dwd_me_spc": 1,
            # set to 1 for radars (mublaze) where the spectra in the pds files
            # are shifted by nfft/2 so that vel": 0 appears at nfft/2
            "pds_spectra_centered": 1,

            # 713031680. for mublaze radars and 1. for the others
            "pds_spectra_normalization": 713031680,

            # normalization needed with mode 12,..,26 in case of the new radarserver  MBR4
            # "rxspc_normalization": 2.69272e+09

            # normalization needed with mode 12,..,26 in case of the new radarserver    MBR3
            # "rxspc_normalization": 5.44

            # 30 for mublaze radars and 1. for the others (hopefully never important)
            "pds_noical_normalization": 30,

            # scaling factor used iq2spc (important should be adjusted for all radars)
            "iqpowerscalingfactor": 5.6,

            # 1 nur bei der noisecommessung vom 5.8.2014 MBR4
            # "noisecomgainmurx": 1

            # 1 for mublaze radars and and 0 for others
            "pds_pnoipcal_log": 1,

            # 1 for mublaze radar where the nois and cal gate appear at one rg lower
            "noise3": 1,

            # "dsprangeoffset": 0.0,
            # "hybridmode": 0,
            # stc-parameter in pparm is on sometimes though no stc mode is switched on
            "has_stc": 0,
            # Receiver bandwidth loss ~ 2 dB acording to Doviac & Zirnic 1978.
            # Set to 1.0 to maintain compatibility with wrong old calibration.
            # Default 1.0.
            "TxRecbandwidthMatchingLoss": 1.58,
            # Receiver bandwidth loss ~ 2 dB acording to Doviac & Zirnic 1978.
            # Set to 1.0 to maintain compatibility with wrong old calibration.
            # Default 1.0. Account here also for the dB discrapancy described
            # by Florian in the HALO calibration paper.

            # "TxRecbandwidthMatchingLoss": 2.0,
            # systems where the the time and azimuth stamps are not shifted by
            # one averaging period. 1 for uBlaze and for rx_client data, 0 else
            "no_stamp_shift": 1,

            # quality parameter of the ncdf4 compression
            "gzipq": 5,

            # "pcpinco": -30,
            # "pcpincx": -30,
            "pcpinco": 71,
            "pcpincx": 71,

            # For radars with fixed beam elevation
            # "fixedelv": 17.0
            # For radars with fixed beam azimuth
            # "fixedazi": 40.0

        },
        'mmclx': {
            # linear(-29.0) Polarization decoupling needed for saturation correction
            "PolDecoupl": 0.00126,
            # linear(-13.0) If average LDR is above this threshold the
            # clusterfilter for hydrometeors is increased
            # "StrongPlank": 0.035 ,
            # This is done in aboveStrongPlank/sinelv ranges above
            # "aboveStrongPlank": 50 ,
            # Peaks with fall velocities below plankVEL are never flagged as
            # plankton Active only in vertical elv
            "plankVEL": -7,
            "noscanner": 0,
            # activate next line for Eriswil/Payern
            # Typical difference in the temperature between the location of the
            # radar and the metar airport (negative if airport is lower)
            # "TempDifMetar7Mira36": -2.6  ,
            # fuer nebelmessungen
            # "LowesPlanktop": 700 ,
            # fuer nebelmessungem. I corrected this typo in newer
            # mmclx-versions (Nov 2016)
            # "LowestPlanktop": 700 ,
            "dRHO7dLDR": -0.028,
            # This value is used for saturation correction. Additionally if
            # SNR is above this value elevated LDR is assubed to be caused by
            # satturation and not by insects.
            "SNRsaturation": 900000.,
            "do_saturationcorr": 0,
            # First File With Out .00, default:1. Wenn die mmclx files eine
            # bestimmte Groesse erreicht haben, wird eine
            # naechste Ausgabedatei mit .01, .02,.. eroffnet.
            # Wenn "FirstfileWo00": 1  dann wird kein .00 an die erste  Datei
            # gehaengt. Das ist sinnvoll, wenn man selbst dafuer sorgen will,
            # dass die mmclx-Dateien micht zu gross werden.
            "FirstfileWo00": 1,
            # if 0 use noise receiver noise level for calibration level if 1
            # use internal refeerence source
            "do_corrbycal": 0,
        },
        'spectraprocessing': {
            # if 0 use noise receiver noise level for calibration level if
            # 1 use internal refeerence source
            "do_corrbycal": 0,




            "do_corrbycal": 0,
            # secondary spectral averaging by which can be performed by
            # " This should be set by th calling script
            # "nave": 1        ,
            # should be set to estimated power in case the thermistor is broken
            # "FixedValTpow": -1.0 ,
            # Eriswil
            # below this range in m the symmetric clutter filter is enabled
            "clutter_maxrange": 1050.,
            # below this height the symmetric clutter filter is enabled
            "clutter_maxhei": 1000.,
            # below this range in m the symmetric clutter filter is enabled
            # "clutter_maxrange": 500. ,
            # below this height the symmetric clutter filter is enabled
            # "clutter_maxhei": 300.,
            # obsolete max range gate number of clutter suppression
            "rgclutter": 900,
            # Noise removal is only done if both sholder are lower than the
            # head and the average of the sholders is smaller than the
            # head *linear(max_sholder_dbc)
            "max_sholder_dbc": -0.80,
            # "fftwin": 0.45:0.0048:0.00086:0.0006:0.00039:0.00028:0.00024:0.00016   # clutter footprint
            # "fftwin": 0.28:0.0032:.00064:0.0003:0.00019:0.00014:0.00012:0.00008:0:0:0:0:0.00004:0.00012:0.00004  # clutter footprint mit side lobes
            # "fftwin": .33:.004:.0004:.0001:.000015:.000003 # ohne side lobes schmaler  bis 25.10.2015
            # ohne side lobes schmaler
            "fftwin": [.32, .006, .0006, .00015, .000030, .000003,
                        .000001, .000001, .0000005, .0000005],
            # "fftwin": .96:.01:.0006:.00015:.000030:.000003 # scanning
            # "fftwin": .35:.0065:.00104:.00025:.000043:.000025:.000015:.000007 # ohne side lobes breiter
            # "fftwin": .31:.0042:.0010:0:0:0:0:0    # default
            "newgc": 0,
            "ngc_zmsr": 1,
            "GcLdrPeakDesc": 0.0126,
            "GcMaxRms": 0.092,
            "GcMaxVelo": 0.19,
            "GcLowestLdr": 0.0158,
            # Set to about -2 for vertically pointing mode
            "expected_vrange": 0.0,
            "avav_of0": 47,
            "navavt": 6,
            # This option only comes into effect if nave > 1 or tave > 0.
            # In this case if avepos": 1 the azi and elevations during the
            # averaging period defined by nave or tave are averaged
            # and then 720 is added which is the indicator for the software
            # that continues processing of the output files of spectraprocessing
            # that the time/azi/elv stamps are not shifted as it was the
            # case for radars before uBlaze
            "avepos": 0
            # "ldrcorr": 0.001,
        },

        'noisecom': {
            # "NoiseComFn_co":"~/specialdata/mbr5/170505_113009_noisecomCo.nc.sav",
            # "NoiseComFn_cx":"~/specialdata/mbr5/170505_105436_noisecomCx.nc.sav",
            # Zurich Geneva
            # "NoiseComFn_co":"~/specialdata/mbr5/190203_123418.noisecom35195_co.nc.sav",
            # "NoiseComFn_cx":"~/specialdata/mbr5/190203_130453.noisecom35195_cx.nc.sav",
            # ab 2022 leider erst 4.1.2022 13:00 aktiviert
            "NoiseComFn_co": "~/specialdata/mbr5/20211122/211122_174553.noisecom35140_co.nc.sav",
            "NoiseComFn_cx": "~/specialdata/mbr5/20211122/211123_025747.noisecom35140_cx.nc.sav",
            "KuhneFn_co": "~/specialdata/mbr5/20211122/211123_012512.kuhne35140_co.nc.sav",
            "KuhneFn_cx": "~/specialdata/mbr5/20211122/211122_161317.kuhne35140_cx.nc.sav",
        },
        'radarconstant': {
            # 1.2 m Antenne Zuerich Fog Campaign
            # linear(50.1)   # anpassen an Datenblatt
            # "AGain": 102329.,
            # 0.56 * !pi/180.  # anpassen an Datenblatt
            # "ABeamWidth": 0.009774,
            # single path waveguide length including the inside the feed
            # "WGlength": .55,
            # Antenna Diameter used for the near field correction
            # "AntennaDiameter4NearFieldCorr": 1.2,

            # 1.2 m antenne von Sats A117
            # linear(50.5)   # anpassen an Datenblatt
            "AGain": 112202.0,
            # 0.463 deg
            "ABeamWidth": 0.008081,
            # scanner + adapter (7 cm)
            "WGlength": 0.698,
            # Antenna Diameter used for the near field correction,
            "AntennaDiameter4NearFieldCorr": 1.2,

            # 1 m die cast antenna
            # linear(48.9)   # anpassen an Datenblatt
            # "AGain": 77625.,
            # 0.61 * !pi/180.  # anpassen an Datenblatt
            # "ABeamWidth": 0.0106,
            # single path waveguide length including the inside the feed
            # "WGlength": 0.75,
            # Antenna Diameter used for the near field correction
            # "AntennaDiameter4NearFieldCorr":"1.0,
        },

        "viewncdf": {

            # if northangle has not been set correctly by control client
            # then it can be set here to correct the orientation of the ppi
            # plots and headline of the rhi-plots.
            # Northangle is the meteorological angle to which the antenna
            # is pointing after moveabs x 0 y 0
            # noetig wenn northangel nich mit den control client gesetz wurde
            # "northangle":0.,
            # noetig wenn northangel nich mit den control client gesetz wurde
            # "northangle":297.8,
            # noetig wenn northangel nich mit den control client gesetz wurde
            # "northangle":301.0,
            # Parameter for MRM command: height where radiometer curve
            # should be plotted
            "mrm_hei": 11.0,
            # Parameter for MRM command: This parameter is added to the db
            # values of the radiometer curve
            # "mrm_off":-0.039,
            "saturate": 0,
            "findscan": 0,
            "min_txn_pow": 11.0,
        },
        'sun_position': {
            # needed for solar scans
            # at metek in mittlere Halle:
            # "alt": 0.013,
            # "lat": 53.742595,
            # "lon": 9.695466,
            # at metek Ausfahrt:
            # "alt": 0.012,
            # "lat": 53.742773,
            # "lon": 9.694812,
            # at Eriswil:
            "alt": 0.920,
            "lat": 47.07052,
            "lon": 7.87263,
        },
        "header2ncdf": {
            ##########
            # die Naechsten 3 Zeilen falls das nicht in der Zeile
            # "Description" von /icfg/header.ini steht
            # "Altitude" : "541 m",
            # "Latitude" : "48.147845 N",
            # "Longitude" : "11.573396 E",
            "institution": "METEK",
            "Copyright": "",
            "Copyright_Owner": "",
            # "System": "MIRA-35",
        }

    }}

    # TODO
    # hack to work with the NMR radar from tropos, requires better
    # knowdlege
    radars['NMRA'] = radars['MBR5'].copy()

    radars['NMRA']['syspar']["noise3"] = 0
    radars['NMRA']['syspar']["magic4chs"] = [
        radars['NMRA']['syspar']["magic4chs"]] + ['nmra']

    return radars[radar]

# general config


def getsysconfig(radar='MBR5'):
    #  contains system parameters which can not be found in the pds data as
    # e.g. the system frequency or the magic 4 characters which are in the
    # pds data but for extracting the data they should be known in advance.

    config = {'syspar':
              {"magic4chs": 'XXXX',
               # z.B. radarconst braucht syspar.iam aus dem magic4chs common
               # block um mit config2stru,radarconst,[<iam>.config] die
               # Radarconstante einstellen zu koennen.
               "iam": radar,
               # 1, if the radarserver saves the pds data in the format which
               # allows repeated headers DWD0 and NASA
               "FzkPdsFmt": 1,
               # number of channels, co cross, or only co
               "zchan": 2,
               # 100. if the power givn in the serverinfo structs is
               # given in centi Watts.
               "fzk100": 1.0,
               # 0: power is given as average power# 1 power is given as peak
               # power in W as in MBX1# 2: power in given as peak power in kW line in MBC3
               "powermetertype": 0,
               # If set 90 deg will be added to elvpos ELVPOS
               "fzkscanner": 0,
               # for vertically pointing systems this should be 1
               "noscanner": 0,
               # fixedelv in case of noscanner=1
               "fixedelv": 90.0,
               # fixedazi in case of noscanner=1
               "fixedazi": 0.0,
               # set to 0 for systems where the stc attm parameter needs to
               # under dsp msc needs to be set to 1 though the harware does
               # not have stc. (mpi2 hybrid receiver)
               "has_stc": 1,
               # 512 in case of all systems with max 500 range gates,
               # 1024 for the systems since 1khg.
               "mombufsize": 512,
               # center frequrncy
               "txfrequency": 35.5 * 10**9,
               # syspar.RecPulseWidths[ppar.pdr]=Pulse Width is ns
               "RecPulseWidths": [96, 192, 288, 400, 512, 576, 706, 832, 912, 1040],
               # correct offset in range assignment caused by error in dsp
               # software. It should be -90 m for IFT and -30 for FMI and 0 else
               "dsprangeoffset": 0.0,
               # set to -1 for radars where the falling targets have positive
               # Doppler shifts.
               "dwd_me": 1,
               # set to -1 for radars where the falling targets have
               # positive Doppler shifts.
               "dwd_me_spc": 1,
               # set to 1 for radars (mublaze) where the spectra in the pds
               # files are shifted by nfft/2 so that vel=0 appears at nfft/2
               "pds_spectra_centered": 0,
               # 713031680. for mublaze radars and 1. for the others
               "pds_spectra_normalization": 1,
               # normalization needed with mode 22,..,26 of xcrl radarserver
               "xcrl_spc_normalization": 2.69272 * 10**9,
               # normalization needed with mode 12,..,16 of rx_client
               "rx_spc_normalization": 5.44,
               # 30 for mublaze radars and 1. for the others (hopefully never
               # important). Because I cannot find why it should be 30, its not used any more.
               "pds_noical_normalization": 1,
               # scaling factor for the Npow1/2 and CPow1/2 from the srvinfo
               # structure in the pds files from the xcrl-processing (hopefully never important).
               "xcrl_noical_normalization": 2.0,
               # if set work around bug in DSP of IFT and FMI. In these
               # systems there are no noise and cal gates in the IQ data.
               "NoNoiCalGateInIQ": 0,
               # scale the spectra gained by iq-processing so that the adc
               # noise in these spectra is the same as for the spectra calculated by the DSP
               "iqpowerscalingfactor": 1.0,
               # 1 bei der vermurxten noisecom messung be mbr4, bei der
               # IF-affn2 (co-kanal) ploetzlich auf 5 stand.
               "noisecomgainmurx": 0,
               # 1 for mublaze radars and and 0 for others
               "pds_pnoipcal_log": 0,
               # 1 for mublaze radar where the nois and cal gate appear at one rg lower
               "noise3": 0,
               # difference in ns between PCP while the noisecom test and PCP
               # during operation with aged transmitter, co channel
               "pcpinco": 0,
               # difference in ns between PCP while the noisecom test and PCP
               # during operation with aged transmitter, cx channel
               "pcpincx": 0,
               # 1 for mublaze radars, and for rx_client data, 0 for radars
               # where the time and position stamps are from are shifted by one averaging periode
               "no_stamp_shift": 0,
               # 1 for hybrid mode radars
               "hybridmode": 0,
               # Receiver bandwidth loss (Probert Johnes 1964). Theoretically
               # it should be set to 1.5 = 1.8 dB. Due to not 100 percent
               # rectangular shape of the tx pulse rather 2 dB.
               # For compatibility with old data 1.0 is used for most mira3x
               # systems.  It is not in the radarconstant config structure
               # because it is already needed in header2ncdf.pro
               "TxRecbandwidthMatchingLoss": 1.,
               # loss due to pulse compression has to be defined for each
               # compression mode round(tx_pulsewitch/rx_pulsewidth).
               # 3 us,160 ns it was determined as 2.11
               "compression_loss19": 1.0,
               # quality parameter for the netcdf4 compression
               "gzipq": 4,
               # unit string for reflectivities. It may be set to the name
               # of the customer name if they want to keep known wrong values
               # of the radar constant to maintain continuity
               "Z_UnitString": 'Z',
               # Additional string for the long name of reflectivities. It may
               # be set to something like Offset if they want to keep known
               # wrong values of the radar constant to maintain continuity
               "Z_UnitDescription": '',
               "systemtype": 'MIRA36',
               "dumy":  '',
               }
              }
    # config2stru, syspar,'syspar', [whoami.iam+'.config']
    # FzkPdsFmt = syspar["FzkPdsFmt"]  # for compatibility with older programs
    config['syspar']['magic4chs'] = config['syspar']["magic4chs"][:4]
    # magic4chs = syspar["magic4chs"]

    return config


if __name__ == '__main__':
    pass
    # d = readconfig()
    # idlconfig2toml()
