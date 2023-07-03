#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:58:01 2023

@author: rspirig
"""

##########################################################################
# from https://numpy.org/doc/stable/reference/arrays.dtypes.html
# '?' boolean
# 'b' (signed) byte
# 'B' unsigned byte
# 'i' (signed) integer
# 'u' unsigned integer
# 'f' floating-point
# 'c' complex-floating point
# 'm' timedelta
# 'M' datetime
# 'O' (Python) objects
# 'S', 'a' zero-terminated bytes (not recommended)
# 'U'Unicode string
# 'V' raw data (void)
# codec = 'utf-8'
##########################################################################

def ppar2physhdr(pparframe, meta, config, lenppar):
    from scipy.constants import c
    import numpy as np
    # to be called on the ppar record (resp. dwll) is
    # the following transformation
    if pparframe['pol'][0] > 2:
        pol = 1
        tx_pulsewidth = pparframe['pdr'][0] * 10**-9
        pulse_width_i = pparframe['pol'][0]
        pparframe['pdr'][0] = pulse_width_i
    else:
        pol = pparframe['pol'][0]

        if pparframe['pdr'][0] <= 9:
            pulse_width_i = config["syspar"]["RecPulseWidths"][pparframe['pdr'][0]]
            pparframe['pdr'][0] = pulse_width_i
        else:
            pulse_width_i = pparframe['pdr'][0]

        tx_pulsewidth = pulse_width_i * 10**-9

    pulse_width = pulse_width_i * 10**-9
    delta_h = 0.5 * c * pulse_width

    mom_h_min = pparframe["ihp"][0] * delta_h + \
        config["syspar"]["dsprangeoffset"]
    # -2 is applied as chg includes the two noise
    # estimates which are appended after the higest range gate
    mom_zrg = pparframe["chg"][0] - 2
    mom_height = np.arange(mom_zrg) * delta_h + mom_h_min
    # RS: I'm not happy with this decision of 15k height as this
    # is a height we usually reach. they do contain the noise but to show
    # this it should be clearer than 15k... this is not necessarily much
    # better with 99999 ()
    # mom_height = np.append(mom_height, [15000.,15000.])
    mom_height = np.append(mom_height, [-1, -1])

    raw_h_min = (pparframe["raw_gate1"][0]) * delta_h + \
        config["syspar"]["dsprangeoffset"]
    raw_zrg = pparframe["raw_gate2"][0] - \
        pparframe["raw_gate1"][0] + 1 - config["syspar"]["noise3"]
    # print(raw_zrg)
    # raw_height = [findgen(raw_zrg) * delta_h + raw_h_min,15000.,15000.]

    if pparframe["prf"] < 10:
        prf = 2500. * (pparframe["prf"][0] + 1)
    else:
        prf = float(pparframe["prf"][0])

    if pparframe["sft"] >= 64:
        nfft = pparframe["sft"][0]
    else:
        nfft = 128 * 2 ** pparframe["sft"][0]

    ave = float(pparframe["avc"][0]) * nfft / prf
    overlapping = pparframe["pos"][0] == 2147483648

    len_srvi = -1

    physhdr = {
        # original name of the pds file
        "name": meta["filename"],
        # operator, should begin with 'fzk'
        "operator": meta["operator"],
        # may contain informations like Altitude given in /icfg/header.ini
        "description": meta["description"],
        # e.g. string from the gps receiver
        "location": meta["location"],
        # device
        "sn": 'fzk',
        # ranges
        'ranges': mom_height,
        # number of bytes of the ppar structure
        "len_ppar": lenppar,
        # number of bytes of the srvi structure
        "len_srvi": len_srvi,
        # Pulse repetition frequency
        "prf": prf,
        # puls diameter as choosen in the magnetron mode
        "pulse_width": pulse_width,
        # number of FFT points
        "nfft": nfft,
        # number of spectral averages, does not account for overlapping FFTs
        # like in the pds files
        "avc": pparframe["avc"][0],
        # number of spectral averages, does account for overlapping FFTs
        "nave": pparframe["avc"][0] * (overlapping + 1),
        # averaging time
        "ave": ave,
        # speed of light
        "c": c,
        # transmitting frequency. This is a constant and not from the BITE data!
        "xmt": config['syspar']['txfrequency'],
        # serverinfo.tpow/fzk100 = average tx power in Watts.
        "fzk100": config["syspar"]["fzk100"],
        # range spacing
        "delta_h": delta_h,
        # height of the lowes range gate of the moment data
        "mom_h_min": mom_h_min,
        # number of range gates of the moment data
        "mom_zrg": mom_zrg,
        # height of the lowes range gate of the raw (or spectrum) data
        "raw_h_min": raw_h_min,
        # number of range gates of the raw data
        "raw_zrg": raw_zrg,
        # flag -- oscillosgramm or spectra data available
        "do_raw": pparframe["raw"][0],
        # diameter of the range of velocities which can be allocated
        # without ambiguity  (Velocity Un-Ambigiuty Range)
        "vuar":  c * prf / (2.0*config['syspar']['txfrequency']),
        # nyquist velocity = vuar/2
        "vny": c * prf / (4.0*config['syspar']['txfrequency']),
        # wave length
        "la": c / config['syspar']['txfrequency'],
        # cross polarized data on/off
        "pol": pol,
        # 1 if attn or STC is switched on
        "att": pparframe["att"][0],
        # the header information, aka meta
        "meta": meta,
        # transmit pulsewidth
        'tx_pulsewidth': tx_pulsewidth,
        # FFT window type in xcrl-radarserver processing:
        # 0 – rectangle, 2 – hanning (cosine), 1 – hamming (raised cosine)
        # IOL functions basically bitcompares the nos with
        # 2**31+2**30 (3221225472) and then shifts it. can also shift is
        # instead and then compare to 3 which is far less .. concoluted
        # originally: ishift((pparframe["nos"]) & (2**31 + 2**30), -30),
        "FftWindowType": (pparframe["nos"] >> 30) & 3,
        # Phase Correction Position detected by AFA software in the DSP
        # IDL original: (ppar.pos and 2147483647L)/32L
        "pcp": (pparframe["pos"] & int('0b'+'1' * 31, base=2)) / 32,
        "ovl": overlapping,

    }

    return physhdr


def gzipread(fileobject, dtype, count=1):
    import numpy as np
    # getting readfunc(fileobject, dtype='a4', count=1)
    if isinstance(dtype, np.dtype):
        nbytes = dtype.itemsize
    else:
        dtype = np.dtype(dtype)
        nbytes = dtype.itemsize
    # print('**', fileobject.tell(), nbytes, dtype, count)
    _bytes = fileobject.read(nbytes*count)

    _rec = np.frombuffer(_bytes, dtype)#, count=count)
    _rec = np.array(_rec)
    return _rec


def srvi2phys(srvihdr, pparhdr, config):
    # srvihdr = [i.tolist() for i in srvihdr]
    # for older radar where this structure tag was used for some errorflag
    if srvihdr['microsec'][0] < 0 or srvihdr['microsec'][0] > 10**6:
        srvihdr['microsec'][0] = 0

    for key in ['npw1', 'npw2', 'cpw1', 'cpw2']:
        srvihdr[key] = srvihdr[key][0]

    if srvihdr['npw1'] < -10:
        for key in ['npw1', 'npw2', 'cpw1', 'cpw2']:
            srvihdr[key] = 10**(srvihdr[key]/10)

    if pparhdr['osc'] in [12, 16, 22, 26]:
        normfac = config['syspar']['xcrl_noical_normalization']
    else:
        normfac = config['syspar']['pds_noical_normalization']
        normfac *= config['syspar']['pds_spectra_normalization']

    for key in ['npw1', 'npw2', 'cpw1', 'cpw2']:
        srvihdr[key][0] = srvihdr[key][0] * normfac

    if srvihdr['npw1'][0] < -10**-9:
        # Pfusch der nötig ist solange wir zspc daten auswerten wollen,
        # bei denen statt in write_zspc beim aufruf von write_srvinfo
        # is_xcrl=0 nicht gesetzt war
        normfac = config['syspar']['xcrl_noical_normalization ']
        normfac /= config['syspar']['pds_noical_normalization']
        normfac /= config['syspar']['pds_spectra_normalization']
        for key in ['npw1', 'npw2', 'cpw1', 'cpw2']:
            srvihdr[key][0] *= normfac

    return srvihdr


def read_srvi_hdr(fileobject, readfunc):
    import numpy as np
    srviformat = ['u4', 'u4',
                  'f', 'f', 'f', 'f', 'f',
                  'u4', 'u4', 'u4', 'u4', 'u4',
                  'f', 'f', 'f', 'f', 'f',
                  'i4', 'i4',
                  'f', 'f',
                  ]
    srviformat = ['<' + i for i in srviformat]

    srviformat = [np.uint32] * 2 + [np.float32] * 5 + [np.uint32] * 5
    srviformat += [np.float32] * 5 + [np.int32] * 2 + [np.float32] * 2

    srvimeaning = ['framecount', 'time_t',
                   'tpow', 'npw1', 'npw2', 'cpw1', 'cpw2',
                   'ps_err', 'rc_err', 'tr_err', 'radar_status', 'grst',
                   'azipos', 'azivel', 'elvpos', 'elvvel', 'northangle',
                   'microsec', 'PD_DataQuality',
                   'LO_Frequency', 'DetuneFine', ]
    recformat = [(i, j) for i, j in zip(srvimeaning, srviformat)]
    srvi_rf = np.dtype(recformat)
    thissrvihdr = readfunc(fileobject, srvi_rf, count=1)

    return thissrvihdr


def read_srvi_frame(fileobject,
                    readfunc,
                    pparheader,
                    physhdr,
                    config,
                    srvihdr,  # filesize,
                    quiet=True):

    import numpy as np
    momentbuffer = config['syspar']['mombufsize']
    zchannels = config['syspar']['zchan']

    n_ranges = physhdr["raw_zrg"] + 2 + config['syspar']['noise3']

    frametypes = {'fftd': {'recformat': np.float32,
                           'count': physhdr['nfft'] * n_ranges * 2,
                           'shape': [n_ranges, 2, physhdr['nfft']],
                           },
                  # signal to noise ratio
                  'snrd': {'recformat': np.int32,
                           'count': momentbuffer * zchannels,
                           'shape': [momentbuffer, zchannels],
                           # technically, the reduction should be until
                           # raw_zrg / mom_zrg but like this we can keep
                           # the structure of
                           'reduce': [n_ranges, zchannels],
                           },
                  # hildebrand sekhon noise floor
                  'hsdv': {'recformat': np.float32,
                            'count': physhdr["raw_zrg"] * zchannels,
                            'shape': [physhdr["raw_zrg"], zchannels],
                           # 'count': n_ranges * zchannels,
                           # 'shape': [n_ranges, zchannels],
                           },

                  "ccox": {'recformat': '<c',
                           'count': physhdr['nfft'] * n_ranges,
                           'shape': (n_ranges, physhdr['nfft']),
                           },
                  # "xcrd": {'recformat': '<c',
                  #          'count': momentbuffer * zchannels,
                  #          'shape': (momentbuffer, zchannels),
                  #          },
                  }

    # lazy way to add more frametypes that are all the same anyway except
    # the header (and concent of course)
    for key in ['rmsd', 'veld', 'hned', 'xcrd', 'cltr', 'expd', ]:
        frametypes[key] = frametypes['snrd']

    # these are implemented in their own reading as they stem from zspc/zspca
    # files which are generally gzipped and require iterative reading
    for key in ['cocx', 'zspc', 'kmdi', 'km12', 'zspx', 'zspy']:
        frametypes[key] = frametypes['fftd']

    for key in ['cofa', ]:
        frametypes[key] = frametypes['hsdv']

    # special conditions for oscillationmode and momentbuffer 512
    # usually this would be float but in this case its going to be integer
    if pparheader['osc'] == 0 and momentbuffer == 512:
        for key in ['fftd', 'cocx', 'zspc', 'kmdi', 'km12', 'zspx', 'zspy']:
            frametypes[key]['recformat'] = np.int32
    elif pparheader['osc'] == 1:
        for key in ['fftd', 'cocx', 'zspc', 'kmdi', 'km12', 'zspx', 'zspy']:
            # switch all to integer, which means we need to read in
            # twice the "numbers" (bytes stays the same)
            frametypes[key]['recformat'] = np.int16
            frametypes[key]['shape'] = [2] + frametypes[key]['shape']
            frametypes[key]['count'] = 2 * frametypes[key]['count']
    elif pparheader['osc'] == 4:
        pass
        # do not do anything here as they are treated specially in the below
        # if conditions, mainly because their read in is interactive
        # the only thing is to set count to -1 to not report information
        # about missmatch in count and chunklen as these do not add up
        for key in ['fftd', 'cocx', 'zspc', 'kmdi', 'km12', 'zspx', 'zspy']:
        #     # switch all to integer, which means we need to read in
        #     # twice the "numbers" (bytes stays the same)
            frametypes[key]['count'] = -1

    # scaling functions for the single pieces of information
    frametypes['veld']['scaling'] = physhdr['la'] * physhdr['prf']
    frametypes['xcrd']['scaling'] = physhdr['la'] * physhdr['prf']
    frametypes['rmsd']['scaling'] = physhdr['la'] * physhdr['prf']


    subframes = {}

    # do this until we find the magic signature again
    # which breaks out the while loop anyway..
    thisframeheader = ''

    if isinstance(config['syspar']['magic4chs'], list):
        magiccharacters = config['syspar']['magic4chs']
    else:
        magiccharacters = [config['syspar']['magic4chs']]

    magiccharacters = [i.lower() for i in magiccharacters]
    while thisframeheader not in magiccharacters:
        # check which frameheadertype this is
        thisframeheader = readfunc(fileobject, dtype='a4', count=1)

        # since we never know when we are actually done with anything in
        # this silly pds format that can have frames absolutely random
        # in order according to the manual (wtf) we check
        if thisframeheader.size == 0:
            if not quiet:
                print(f'End of pds file reached at byte {fileobject.tell()}')
            break

        thisframeheader = thisframeheader[0].decode().lower()

        # account for 4 bytes per float or long int
        chunklen = readfunc(fileobject, dtype='i4', count=1)[0]
        chunklen //= 4

        # since we never know when we are actually done with anything in
        # this silly pds format that can have frames absolutely random
        # in order according to the manual (wtf) we check
        if thisframeheader in magiccharacters:

            if not quiet:
                print(f'End of srvi frame reached at byte {fileobject.tell()}')

            # move back the 8 bytes that we read in for the header so the
            # main programm and its while loop may continue
            fileobject.seek(fileobject.tell() - 8)
            break

        if chunklen == 0:
            if not quiet:
                print(f'Empty data for HEADER {thisframeheader}')
            continue
        else:
            if not quiet:
                print(f'HEADER {thisframeheader} of len {chunklen} bytes')

        if thisframeheader not in frametypes.keys():
            print(f'Unknown frameheader {thisframeheader},',
                  f'skipping the size {chunklen} forward to continue reading')

            # use of readfunction instead of seek because of gzipped files
            nonsense = readfunc(fileobject, dtype=np.uint32, count=chunklen)[0]

            continue

        rf, cnt, shp = (frametypes[thisframeheader]['recformat'],
                        frametypes[thisframeheader]['count'],
                        frametypes[thisframeheader]['shape'],
                        )

        reduce = frametypes[thisframeheader].get('reduce', False)
        scaling = frametypes[thisframeheader].get('scaling', False)

        if chunklen != cnt and cnt > 0:
            if not quiet:
                print(f'Issue with framelength! Was {chunklen} but expecting {cnt}')
        else:
            if not quiet:
                print(f'Reading {chunklen} bytes of type {rf}')

        # note only zspc and zspy could be tested at the moment
        if thisframeheader in ['zspx', 'zspy', 'zspc']:
            # special requirements to read, aka iterative as each piece
            # gives some information about the next one...


            # the general holder for all data

            buffer = np.zeros(shp) + np.nan
            cocxbuffer = np.zeros((physhdr['nfft'], n_ranges), dtype=np.complex64)
            # floats have been packed at integer to save storage.....
            # and were scaled by i.e. 2**16 (65536) as it is the max for unsigned ints
            # but instead someone chose 65530
            scaling_factor_unsignedint = 1/65530
            # this is given as 3.0520e-0 in the read1pds_dwell but I'm failry
            # certain that this should be the maxrange of a signed uint,
            # i.e. 2**15 (32768), the max for signed ints (1 bit for the sign)
            scaling_factor_signedint = 1/32765
            # either way, both scaling factors are like this in the IDL source
            # so we now have to accept this.......................
            # horrible horrible loop, never do this if avoidable.....
            # and blame metek for this mess.
            for i_range_gate in range(physhdr["raw_zrg"]):
                nrecs = readfunc(fileobject, np.int16, count=1)[0]

                for rec in range(nrecs):

                    # see how many records we have to read
                    index, nparts = readfunc(fileobject, np.int16, count=2)

                    # read those records
                    # this can/should be rewritten without loop
                    # _rf = f'<{nparts}u2,<f,' * zchannels
                    # _rf = _rf.rstrip(',')
                    _rf = []
                    for i in range(zchannels):
                        _rf += [(f'f{i*2+0}', np.uint16, (nparts)),
                               (f'f{i*2+1}', np.float32, )]
                    _rf = np.dtype(_rf)

                    spectrumpiece = readfunc(fileobject,
                                             _rf,
                                             # (f'<{nparts}u2,<f,' * zchannels).rstrip(','),
                                             count=1)[0]
                    scaling_max = np.asarray(spectrumpiece.item()[1::2])
                    spectrumpiece = np.asarray(spectrumpiece.item()[::2])
                    spectrumpiece = spectrumpiece.T * scaling_max * scaling_factor_unsignedint
                    buffer[i_range_gate, :, index:index+spectrumpiece.shape[0]] = spectrumpiece.T

                    if thisframeheader == 'zspx':
                        # first the magnitude
                        realparts = readfunc(fileobject,
                                             np.int16,
                                             count=nparts[1]+1)
                        scaling_max = realparts[-1]
                        realparts = realparts[:-1]
                        realparts *= scaling_max * scaling_factor_unsignedint

                        # then the phase
                        imparts = readfunc(fileobject,
                                           np.int16,
                                           count=nparts[1]+1)
                        scaling_max = imparts[-1]
                        imparts = imparts[:-1]
                        imparts *= scaling_max * scaling_factor_signedint

                        # refers to do cocx
                        #         if do_cocx then begin
                        #             readu,lun,spcpiece,max_
                        #             r_cocxbuf[ixa,irg]=1.5260186e-05 * max_ * spcpiece
                        #             pmspcpiece=intarr(zix,/nozero)
                        #             readu,lun,pmspcpiece,max_
                        #             ph_cocxbuf[ixa,irg]=3.0520e-05 * max_ * pmspcpiece ; the angles may be negative. Therefore they where packed by packpmflt2int.pro
                        #         endif
                        pass

                    elif thisframeheader == 'zspy':
                        # refers to do ccox
                        _rf = []
                        for i in range(zchannels):
                            _rf += [(f'f{i*2+0}', np.uint16, (nparts)),
                                   (f'f{i*2+1}', np.float32, )]
                        _dtype = np.dtype(_rf)
                        # _dtype = f'<{nparts}i2,<f,<{nparts}i2,<f'

                        spectrumpiece = readfunc(fileobject,
                                                 _dtype,
                                                 count=1)[0]
                        scaling_max = np.asarray(spectrumpiece.item()[1::2])
                        spectrumpiece = np.asarray(spectrumpiece.item()[::2])
                        spectrumpiece = spectrumpiece.T * scaling_max * scaling_factor_signedint
                        spectrumpiece = spectrumpiece[:,0] + spectrumpiece[:,1] * 1j
                        cocxbuffer[index:index+spectrumpiece.shape[0], i_range_gate] = spectrumpiece

            if 'cocx' not in subframes.keys():
                subframes['cocx'] = []

            subframes['cocx'].append(cocxbuffer.T)

            thissubframe = buffer

        else:

            thissubframe = readfunc(fileobject, rf, count=cnt)
            thissubframe = thissubframe.reshape(shp)

            # reduce dimensions to actual range gate
            if reduce is False:
                pass
            else:
                if len(reduce) != 2:
                    print('Wrong reduce shape!')
                    pass
                else:
                    thissubframe = thissubframe[:reduce[0], :reduce[1]]
            # scale values if given
            if scaling is False:
                pass
            else:
                thissubframe = thissubframe * scaling

        if thisframeheader not in subframes.keys():
            subframes[thisframeheader] = []

        subframes[thisframeheader].append(thissubframe)

    # skeleton to overwrite the fix the cocx frametype based on oscillation
    # mode that was defined the ppar header coming before
    if pparheader['osc'] in [2]:
        # for completeness wit the idl version where the two channels are
        # split up into r_cocx and ph_cocx, real part and phase
        if 'cocx' in subframes:
            # convert to complex and normalize
            subframes['r_cocx'] = [_frame[:, 0, :]
                                   for _frame in subframes['cocx']]
            subframes['ph_cocx'] = [_frame[:, 1, :]
                                    for _frame in subframes['cocx']]
        pass
    elif pparheader['osc'] in [3, 16, 24]:

        if 10 <= pparheader['osc'] <= 19:
            xspc_normalization = config["syspar"]['rx_spc_normalization']
        elif 20 <= pparheader['osc'] <= 29:
            xspc_normalization = config["syspar"]['xcrl_spc_normalization']
        else:
            xspc_normalization = 1

        if 'cocx' in subframes:
            # convert to complex and normalize
            subframes['cocx'] = [(_frame[:, 0, :] + 1j * _frame[:, 1, :])
                                 * xspc_normalization
                                 for _frame in subframes['cocx']]

    else:
        pass

    #################################
    ##### Postprocess Spectra #######
    #################################
    # apply fixes for the various FFTD/SDPD /ZSPC/ZSPX/ZSPY
    postprocess = [i for i in subframes.keys()
                   if i in ['fftd', 'sdpd', 'zspc', 'zspx', 'zspy']]

    for key in postprocess:

        if key not in subframes:
            # nothing to be done
            continue

        if pparheader['osc'] == 2:
            # spectra from iq2spc or pds output from other idl programs
            # nothing needs to be done really as we keep the channels
            # together in this reader.
            pass

        elif pparheader['osc'] == 4:
            # spectra from zspc
            # fake the noise and reference spectra so that spectraprocessing
            # can calculate npw and cpw from it. these are put into the 2
            # last range gates for each channel
            # print(srvihdr['npw1'], physhdr['nfft'])
            # print(subframes)
            for entryno, entry in enumerate(subframes[key]):
                # print('XXX', entry.shape)
                entry[-2, 0, :] = srvihdr['npw1'][0] / physhdr['nfft']
                entry[-1, 0, :] = srvihdr['cpw1'][0] / physhdr['nfft']

                if zchannels == 2:
                    entry[-2, 1, :] = srvihdr['npw2'][0] / physhdr['nfft']
                    entry[-1, 1, :] = srvihdr['cpw2'][0] / physhdr['nfft']

                subframes[key][entryno] = entry

        elif 'hned' in subframes.keys() or 'expd' in subframes.keys():
            # exp_ex = if EXPD in frame
            # hne_ex = if HNED in frame
            if 'expd' in subframes.keys():
                # we do not need indices because we have already reduced the
                # dimensions of the expd part and they do fit directly
                # and we also do not need a matrix multiplication here
                # but as of writing this function this branch is untested

                pds_spectra_normalization = config['syspar']['pds_spectra_normalization']

                subframes[key] = [entry * pds_spectra_normalization
                                  * 2. ** subframes['expd'][entryno][:, :, np.newaxis]
                                  for entryno, entry in enumerate(subframes[key])]

            elif 'hned' in subframes.keys():
                # spectra from uBlaze radarservers or rx_client
                # if abs(syspar.pds_spectra_normalization -1.0) <= 0.1:
                #     rx_client2dsp_scaling=1./(32768.**2)
                # else:
                #     rx_client2dsp_scaling=1.0 # (32768./1.852)**2

                if 10 <= pparheader['osc'] <= 19:
                    normfac = config['syspar']['rx_spc_normalization']
                elif 20 <= pparheader['osc'] <= 30:
                    normfac = config['syspar']['xcrl_spc_normalization']
                else:
                    normfac = config['syspar']['pds_spectra_normalization']

                subframes[key] = [entry * normfac for entry in subframes[key]]

            else:
                raise KeyError

        else:
            print(f'Postprocessing of spectra of kind {key} not possible',
                  'due to lack of scaling parameters (epxd/hned) in frame')

    # flatten lists if they are only 1 element long (which they usually are)
    for key, value in subframes.items():
        if len(value) == 1 and isinstance(value, list):
            subframes[key] = value[0]

    return subframes


def get_ppar_recformat():
    import numpy as np
    pparformat = ['i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4',
                  'f', 'f',
                  'i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4',
                  'i4', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4',
                  'f', 'f', 'f', 'f',
                  'i4', 'i4', 'i4', 'i4']
    pparformat = ['<' + i for i in pparformat]
    # faster than commasep string because of internal parsing by numpy
    pparformat = [np.int32] * 9 + [np.float32] * 2 + [np.int32] * 16
    pparformat += [np.float32] * 4 + [np.int32] * 4

    pparmeaning = ['prf', 'pdr', 'sft', 'avc', 'ihp', 'chg', 'pol', 'att', 'tx',
                   'adcgain0', 'adcgain1', 'wnd', 'pos', 'add', 'len', 'cal', 'nos',
                   'of0', 'of1', 'swt', 'sum', 'osc', 'tst', 'cor', 'ofs', 'HSBn',
                   'HSBa', 'calibrpower_m', 'calibrsnr_m', 'calibrpower_s',
                   'calibrsnr_s', 'raw_gate1', 'raw_gate2', 'raw', 'prc']
    recformat = [(i, j) for i, j in zip(pparmeaning, pparformat)]
    ppar_rf = np.dtype(recformat)
    return ppar_rf


def read_pds(file_or_list_of_files,
             # whether every n-th bytes should be searched in case
             # there is a missmatch
             bytesearchwindow=1,
             firstnframes=30,
             lastnframes=30,
             quiet=True,
             radarconfig='MBR5',
             *args,
             **kwargs,
             ):

    """
    Reads PDS files from a Metek Mira 35 and processes the data with the radar specific config.

    Parameters
    ----------
    file_or_list_of_files : str or list
        A single file path or a list of files to be read.
    bytesearchwindow : int, optional
        Specifies the interval of bytes to be searched for a match in case of
        a frame header mismatch. Defaults to 1, higher numbers means searching
        goes faster but at the risk of finding no additional frames after a
        corrupt header appeared
    firstnframes : int, optional
        Specifies the number of frames to process from the beginning of each file.
        Defaults to 30. Not applicable for gzipped PDS files
    lastnframes : int, optional
        Specifies the number of frames to process from the end of each file.
        lastnframes takes precedence over firstnframes, i.e. only of the two
        can be done at a time. The reason for lastnframes to be preferred is
        the idea of operational processing of PDS frames by spectraprocessing
        afterwards, i.e. a subset of the file can be read in. Due to the way
        seek works with compressed file (seek goes to the byte given, dis-
        regarding the compression) this is turned off when a zipped file is
        read.
        Defaults to 30.
    quiet : bool, optional
        Specifies whether to suppress printing progress messages. Defaults to True.
    radarconfig : str, dict, optional
        Either a config dict of a radar, the path to a config file of a radar
        or the name of a "known" radar (MBR5 and MBR7 at the moment)
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    An xarray Dataset with (time, range, fft, channels) of the individual
    srvi/pparframes

    Notes
    -----
    - This function reads PDS files and performs processing on the data similar
      to the processing of Metek (scaling, ranging etc) but no spectraprocessing.
    - It can read a single file or a list of files, returning an xarray dataset
    - The function can process a specified number of frames from either the
      beginning or the end of each file, but only for uncompressed PDS files
    - By default, it suppresses progress messages, but this behavior can be
      changed by setting 'quiet' to False.
    - The radarconfig parameter allows specifying the name of the radar,
      passing a path to a config file or a config dictionary
    - Numpy datatypes have been on purpose given a np.float and similar since
      the string parsing took up the bulk of processing when in compressed mode

    Example usage
    -------------
    >>> read_pds('data.pds')
    >>> read_pds(['data1.pds', 'data2.pds'], bytesearchwindow=2, lastnframes=50)

    See Also
    --------
    np.frombuffer : Load binary data from a bytes buffer.
    np.fromfile : Read binary data from a file into an array.

    """
    import os
    import gzip

    import datetime
    import numpy as np
    import pandas as pd
    import xarray as xr

    from radar.config import (getsysconfig, getradarconfig, readconfig)

    if bytesearchwindow <= 0:
        bytesearchwindow = 1

    if isinstance(file_or_list_of_files, str):
        file_or_list_of_files = [file_or_list_of_files]

    if firstnframes and lastnframes:
        if not quiet:
            print('One kind of reading has to be chosen, either first or last n frames')
            print('Defaulting to last frames')
        firstnframes = False

    # cross-conversion with gzipped pds files
    readfuncs = [np.fromfile, gzipread]
    openfuncs = [open, gzip.open]

    # config only contains syspar at this point
    config = getsysconfig()
    if (radarconfig, str) and not os.path.exists(radarconfig):
        radarconfig = getradarconfig(radar=radarconfig)
    elif (radarconfig, str) and os.path.exists(radarconfig):
        radarconfig = readconfig(radarconfig)
    else:
        # make sure its a dict like the config and that it holds
        # at least the syspar, we could also check for more though
        assert isinstance(radarconfig, dict), 'syspar' in radarconfig

    # update the general sysconfig with the specific config, only syspar is
    # relevant here for the processing
    if 'syspar' in radarconfig:
        for key, value in radarconfig['syspar'].items():
            config['syspar'][key] = value

    # config['syspar']['zchan']
    # return
    ppar_rf = get_ppar_recformat()
    # FILES HAVE TO BE READ SEQUENTIALLY
    for file in file_or_list_of_files:
        with open(file, 'rb') as fo:
            gzipchk = fo.read(2)

        # check if the first 2 bytes are the magic bytes
        # see gzip documentation for more
        gzipchk = gzipchk == b'\x1f\x8b'
        # first estimate of file size, will be overwritten
        # when file is open to better estimate the actual filesize also for
        # gzipped files
        filesize = os.stat(file).st_size

        if filesize < 1024:
            print(f'File {file} was truncated and has {filesize} size.',
                  'It should be at least (!) 1024')
            continue

        # select funtion to actually read file
        if file.endswith('.gz') or gzipchk:
            openfunc, readfunc = openfuncs[1], readfuncs[1]
            ftype = 'gzipped pds'
            if not quiet:
                print('Setting lastnframes and firstnframes to zero as',
                      'this is not supported with gzipped files')
            _lastnframes = False
            _firstnframes = False
        else:
            openfunc, readfunc = openfuncs[0], readfuncs[0]
            ftype = 'raw binary pds'
            _lastnframes = lastnframes
            _firstnframes = firstnframes

        if not quiet:
            print(f'Reading {file}, a {ftype} of size {filesize}')

        frames = []
        with openfunc(file, 'rb') as fo:

            filesize = fo.seek(0, 2)
            fo.seek(0)
            # common header
            filename = readfunc(fo, 'a32', count=1)
            time = readfunc(fo, 'a32', count=1)
            operator = readfunc(fo, 'a64', count=1)
            location = readfunc(fo, 'a128', count=1)
            description = readfunc(fo, 'a256', count=1)
            meta = {"filename": filename,
                    "time": time,
                    "operator": operator,
                    "location": location,
                    "description": description}

            for key, value in meta.items():
                meta[key] = value[0].split(b'\n')[0].decode()

            for entry in meta["description"].split(':'):
                key, value = entry.split('=')
                meta[key] = value

            if not quiet:
                print(f'Contained meta information is {meta}')

            preppar = fo.tell()
            pparhdr = readfunc(fo, ppar_rf, count=1)

            postppar = fo.tell()
            lenppar = postppar - preppar

            # move forward to end of systemheader
            fo.seek(postppar + (512 - lenppar))

            physhdr = ppar2physhdr(pparhdr, meta, config, lenppar)

            srviframecount, srvibytesize = 0, -1
            srvipositions, pparpositions = [], []
            lastframe, newpparhdr = False, False


            # THIS IS THE MAIN LOOP/CHUNK WHERE THE FRAMES ARE BEING READ
            while fo.tell() != filesize:
                # print(filesize, fo.tell())
                presig = fo.tell()
                signaturecheck = readfunc(fo, dtype='a4', count=1)

                if signaturecheck.size <= 0:
                    return
                signaturecheck = signaturecheck[0].decode()
                chunklen = readfunc(fo, dtype=np.int32, count=1)[0]

                thisframeheader = readfunc(fo, dtype='a4', count=1)[0].decode()

                if chunklen - 8 != lenppar and thisframeheader.lower() == signaturecheck.lower():
                    print('Header size missmatch, file corrupted after this entry')
                    print(f"Signature was {signaturecheck}, {thisframeheader}")
                    break

                chunklen = readfunc(fo, dtype=np.int32, count=1)

                if thisframeheader.lower() == 'ppar':
                    pparframecount =+ 1
                    pparpositions.append(fo.tell())

                    thisppar = readfunc(fo, ppar_rf, count=1)
                    thisphyshdr = ppar2physhdr(pparhdr, meta, config, lenppar)
                    for (key, dtype), ppvarval in zip(thisppar.dtype.descr, thisppar[0]):
                        # continue
                        if key in thisphyshdr:
                            # print(key, dtype,ppvarval, '******')
                            continue
                        else:
                            thisphyshdr[key] = thisppar[key][0]
                    if srvibytesize > 0:
                        newpparhdr = True

                    # return thisphyshdr
                elif thisframeheader.lower() == 'srvi':
                    # in theory, the order of srvi header and frames
                    # could be random as well.... so far this wasn't seen
                    # maybe ask for testcases? there is only one way to know
                    # whether the header isnt at the start since the srvi hdr
                    # has no signature per se is to check the other types
                    # or hope that the chunklen always adds up to be less
                    # than the other headers; typical srvi headers are
                    # less than 100 bytes

                    srviframecount =+ 1

                    # rough estimation when we need to stop skipping, only stop once
                    # print((filesize - fo.tell()) / srvibytesize <= )

                    if srvibytesize > 0 and (filesize - fo.tell()) / srvibytesize <= 1 and not lastframe:
                        lastframe = True
                        # now move the pointer back once for the proper ppar header

                        if pparpositions[-1] > srvipositions[-_lastnframes-1]:
                            validnframes = [i for i in srvipositions
                                            if i > pparpositions[-1]]

                            print('PPAR header changed too recently wrt to ',
                                  f'the last {_lastnframes} that were requested.',
                                  f'Try {len(validnframes)} as lastnframes instead')
                            return len(validnframes)
                        else:
                            fo.seek(srvipositions[-lastnframes-1])

                    if _lastnframes > 0 and srvibytesize > 0 and not lastframe and not newpparhdr:
                        # basically skip this srvi, but only this
                        # as we may need the ppar info that might be somewhere
                        # in between
                        srvipositions.append(fo.tell())
                        # print(srvibytesize)
                        # nonsense = readfunc(fo, dtype=f'{srvibytesize-25}a', count=1)[0]
                        # print(nonsense[:10])
                        fo.seek(fo.tell() + srvibytesize)
                        if not quiet:
                            print(f'Skipping {srvibytesize} bytes, searching',
                                  f'for the last {_lastnframes} frames')
                        continue

                    presrvi = fo.tell()
                    #############################
                    # first read in srvi header
                    #############################
                    thissrvihdr = read_srvi_hdr(fo, readfunc)
                    thissrvihdr = srvi2phys(thissrvihdr, thisphyshdr, config)
                    #############################
                    # then frames
                    #############################
                    thissrvi = read_srvi_frame(fo,
                                               readfunc,
                                               thisppar,
                                               thisphyshdr,
                                               config,
                                               thissrvihdr,
                                               quiet=quiet,
                                               )

                    postsrvi = fo.tell()

                    if _lastnframes > 0:
                        srvibytesize = postsrvi - presrvi

                    dwelltime = pd.to_datetime(thissrvihdr['time_t']
                                               + thissrvihdr['microsec']/10**9,
                                               unit='s')

                    nfft, ranges = thisphyshdr["nfft"], thisphyshdr['ranges']
                    # thisphyshdr.pop('ranges')
                    channels = ['co', 'cx']
                    nchannels, nranges = len(channels), len(ranges)

                    # move the noise gates away from the ranges, hence - 2
                    # ranges, nranges = ranges[:-2], nranges - 2

                    halfvel = thisphyshdr['vuar'] / 2
                    velocities = np.linspace(-halfvel,
                                             halfvel, thisphyshdr['nfft'])
                    # attach the data of the srvi header in case we need them later on
                    skipkey = ['microsec', 'time_t']
                    data_vars = {key: thissrvihdr[key]
                                 for (key, dtype), scrival in zip(thissrvihdr.dtype.descr, thissrvihdr[0])
                                 if key not in skipkey}

                    _data = xr.Dataset(coords={'time': dwelltime,
                                               # 'nfft': np.arange(nfft),
                                               'velocity': velocities,
                                               'range': ranges,
                                               'channel': channels,
                                               },
                                        attrs=thisphyshdr,
                                        # data_vars=data_vars,
                                       )

                    # if 'ranges' in _data.attrs:
                    #     _data.attrs.pop('ranges')

                    # add the nfft info just for good measure even though we
                    # do not really need it as we added the velocity that
                    # corresponds to the nfft
                    _data['nfft'] = (('velocity'), np.arange(nfft))

                    # attach the remaining data srvi frame
                    for key, value in thissrvi.items():
                        if value.shape == (nranges, nchannels):
                            # the usual moment buffers that do not have a nfft
                            # dimensions
                            _data[key] = (('time', 'range', 'channel', ),
                                          value[np.newaxis, :, :])
                        elif value.shape == (nranges-2, nchannels):
                            # the usual moment buffers that do not have a nfft
                            # dimensions but are also lacking the noise range gate
                            _value = np.zeros((nranges, nchannels),
                                              dtype=value.dtype)
                            _value[:-2, :] = value
                            value = _value

                            _data[key] = (('time', 'range', 'channel', ),
                                          value[np.newaxis, :, :])
                            pass
                        elif value.shape == (nranges, nchannels, nfft):
                            # things like fftd that have all infos
                            _data[key] = (('time', 'range', 'channel', 'velocity',),
                                          value[np.newaxis, :, :, :])
                        elif value.shape == (nranges, nfft, ):
                            # things like cocx that do not have a channel info
                            _data[key] = (('time', 'range', 'velocity',),
                                          value[np.newaxis, :, :])
                        else:
                            print(
                                f'No information for {key}',
                                f'of shape {value.shape}',
                                f'we were looking for something like',
                                f'number of range gate ({nranges}),',
                                f'number of channels ({nchannels}) or',
                                f'number of fft freqs ({nfft})')

                    _data = _data.sortby('range')

                    frames.append(_data)

                    # break
                    # return thissrvihdr
                elif thisframeheader.lower() == 'ftxt':
                    # free text from the radarserver....

                    ftxt = readfunc(fo, dtype='a'+str(chunklen), count=1)[0]
                    print('Read free text in {file}:')
                    print(ftxt)
                elif thisframeheader.lower() == 'term':
                    # terminated pds by radarserver or some other way
                    break
                else:
                    # continue to where we began this while loop
                    # where we checked for signature. Since we read 2x 4 chars
                    # and 2x a long (of each 4 bytes) we have proceeded 16 bytes
                    # hence we go back 15 bytes to ensure each byte is search
                    # ppar or srvi header
                    print('Searching')
                    if not quiet:
                        print('Seeking for more frames, ',
                              'this may take some time....')
                    presig = fo.seek(presig + bytesearchwindow)

                    break

                if _firstnframes is not False and len(frames) >= _firstnframes:
                    lastsrvisize = postsrvi - presrvi
                    bytesremaining = filesize - fo.tell()
                    if not quiet:
                        print(f'Reading terminated after {_firstnframes}.',
                              f'Roughly {bytesremaining/lastsrvisize:.1f} frames',
                              f'remain ({bytesremaining} bytes left)')
                    break

    nffts = [i['nfft'].size for i in frames]
    nffts = np.unique(nffts)
    data = {}
    for infft in nffts:
        _data = xr.concat([i for i in frames
                            if i['nfft'].size == infft],
                          dim='time')

        data[infft] = _data.sortby('time')

    # the first one is usually the default because the nfft hopefully did not
    # change but one never knows.
    if len(data.keys()) == 1:
        return data[infft]
    else:
        return data

if __name__ == '__main__':
    import socket
    from util.pathwalker import pathwalker
    if 'elrooto-desktop' in socket.gethostname():
        path = "/media/elrooto/data/polybox.ethz/cloudlab/data/mbr5/pds/"
    elif 'iacpc' in socket.gethostname():
        path = "/home/rspirig/clouds/polybox/cloudlab/data/mbr5/pds"

    files = pathwalker(path)
    file = files[-1]
    d = read_pds(file, quiet=True, nframes=-1,radar='NMRA', )

    file = files[-2]
    d = read_pds(file, quiet=True, nframes=-1,radar='NMRA', )

    # physhdr, thissrvihdr, thissrvi = read_pds(file, quiet=False)
    # file2 = files[0]
    # d = read_pds(file2,
    #              quiet=True,
    #              nframes=-1,)
