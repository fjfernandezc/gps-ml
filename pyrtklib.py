'''
Funciones de RTKLib para Python

Implementacion obtenida de:
https://github.com/tomojitakasu/RTKLIB/tree/rtklib_2.4.3
https://github.com/tomojitakasu/PocketSDR

+ algunas funciones y modificaciones agregadas
'''

#
#  Pocket SDR Python Library - RTKLIB Wrapper Functions
#
#  References:
#  [1] RTKLIB: An Open Source Program Package for GNSS Positioning
#      (https://github.com/tomojitakasu/RTKLIB)
#
#  Author:
#  T.TAKASU
#
#  History:
#  2021-12-24  1.0  new
#  2022-01-13  1.1  add API test_glostr()
#  2022-02-15  1.2  add API satno(), satsys(), satid2no(), obs2code(),
#                   code2obs(), epoch2time(), time2epoch(), gpst2time(),
#                   time2gpst(), gpst2utc(), utc2gpst(), timeadd(), timediff(),
#                   time2str(), timeget(), traceopen(), traceclose(),
#                   tracelevel(), satazel(), geodist(), ionmodel(), tropmodel(),
#                   get_tgd(), readrnx(), obsfree(), navfree(), obsget(),
#                   ephget(), gephget(), satpos(), ecef2pos(), pos2ecef(),
#                   ecef2enu(), enu2ecef()
#
import os, time, platform, math
from ctypes import *
import numpy as np

# load RTKLIB ([1]) ------------------------------------------------------------
env = platform.platform()
dir = os.getcwd()
if 'Windows' in env:
    librtk = cdll.LoadLibrary('lib/rtklib/win32/librtk.so')
elif 'Linux' in env:
    librtk = cdll.LoadLibrary('lib/rtklib/linux/librtk.so')
else:
    # printf('load librtk.so error for {:s}'.format(env))
    exit()

# get constant -----------------------------------------------------------------
def get_const_int(name):
    return librtk.get_const_int(c_char_p(name.encode()))

# constants --------------------------------------------------------------------
PI            = math.pi
D2R           = PI / 180.0
R2D           = 180.0 / PI
CLIGHT        = 299792458.0
OMGE          = 7.2921151467E-5
RE_WGS84      = 6378137.0
NFREQ         = get_const_int('NFREQ')
NEXOBS        = get_const_int('NEXOBS')
SNR_UNIT      = get_const_int('SNR_UNIT')
MAXSAT        = get_const_int('MAXSAT')
MAXSTA        = get_const_int('MAXSTA')
MAXANT        = get_const_int('MAXANT')
NTYPES        = NFREQ + NEXOBS

FREQ1         = 1.57542E9
FREQ2         = 1.22760E9

SYS_NONE      = 0x00    # navigation system
SYS_GPS       = 0x01
SYS_SBS       = 0x02
SYS_GLO       = 0x04
SYS_GAL       = 0x08
SYS_QZS       = 0x10
SYS_CMP       = 0x20
SYS_IRN       = 0x40
SYS_ALL       = 0xFF

IONOOPT_OFF   = 0       # ionosphere option
IONOOPT_BRDC  = 1
IONOOPT_SBAS  = 2
IONOOPT_IFLC  = 3
IONOOPT_EST   = 4
IONOOPT_TEC   = 5
IONOOPT_QZS   = 6
IONOOPT_STEC  = 8

TROPOPT_OFF   = 0       # troposphere option
TROPOPT_SAAS  = 1
TROPOPT_SBAS  = 2
TROPOPT_EST   = 3
TROPOPT_ESTG  = 4
TROPOPT_ZTD   = 5

EPHOPT_BRDC   = 0       # ephemeris option
EPHOPT_PREC   = 1
EPHOPT_SBAS   = 2
EPHOPT_SSRAPC = 3
EPHOPT_SSRCOM = 4

STR_SERIAL    = 1       # stream type
STR_FILE      = 2
STR_TCPSVR    = 3
STR_TCPCLI    = 4
STR_NTRIPSVR  = 5
STR_NTRIPCLI  = 6
STR_NTRIPCAS  = 9
STR_UDPSVR    = 10
STR_UDPCLI    = 11

STR_MODE_R    = 0x1     # stream reae/write mode
STR_MODE_W    = 0x2
STR_MODE_RW   = 0x3


# type definitions -------------------------------------------------------------
class GTIME(Structure): # gtime_t
    _fields_ = [
        ('time', c_int64 ), ('sec' , c_double)]

class OBSD(Structure): # obsd_t
    _fields_ = [
        ('time', GTIME   ), ('sat' , c_uint8 ), ('rcv' , c_uint8 ),
        ('SNR' , c_uint16 * NTYPES), ('LLI', c_uint8  * NTYPES),
        ('code', c_uint8  * NTYPES), ('L'  , c_double * NTYPES),
        ('P'   , c_double * NTYPES), ('D'  , c_float  * NTYPES)]

class EPH(Structure): # eph_t
    _fields_ = [
        ('sat' , c_int32 ), ('iode', c_int32 ), ('iodc', c_int32 ),
        ('sva' , c_int32 ), ('svh' , c_int32 ), ('week', c_int32 ),
        ('code', c_int32 ), ('flag', c_int32 ), ('toe' , GTIME   ),
        ('toc' , GTIME   ), ('ttr' , GTIME   ), ('A'   , c_double),
        ('e'   , c_double), ('i0'  , c_double), ('OMG0', c_double),
        ('omg' , c_double), ('M0'  , c_double), ('deln', c_double),
        ('OMGd', c_double), ('idot', c_double), ('crc' , c_double),
        ('crs' , c_double), ('cuc' , c_double), ('cus' , c_double),
        ('cic' , c_double), ('cis' , c_double), ('toes', c_double),
        ('fit' , c_double), ('f0'  , c_double), ('f1'  , c_double),
        ('f2'  , c_double), ('tgd' , c_double * 6),
        ('Adot', c_double), ('ndot', c_double)]

class GEPH(Structure): # geph_t
    _fields_ = [
        ('sat' , c_int32 ), ('iode' , c_int32 ), ('frq' , c_int32 ),
        ('svh' , c_int32 ), ('sva'  , c_int32 ), ('age' , c_int32 ),
        ('toe' , GTIME   ), ('tof'  , GTIME   ),
        ('pos', c_double * 3), ('vel', c_double * 3), ('acc', c_double * 3),
        ('taun', c_double), ('gamn' , c_double), ('dtaun', c_double)]

class SOL(Structure):
    _fields_ = [
        ('time', GTIME), ('rr', c_double * 6), ('qr', c_float * 6), ('qv', c_float * 6),
        ('dtr', c_double * 6), ('type', c_uint8), ('stat', c_uint8), ('ns', c_uint8),
        ('age', c_float), ('ratio', c_float), ('thres', c_float)]

class PCV(Structure):
    _fields_ = [
        ('sat', c_int8), ('type', c_char * MAXANT), ('code', c_char * MAXANT),
        ('ts', GTIME), ('te', GTIME), ('off', (c_double * 3) * NFREQ),
        ('var', (c_double * 19) * NFREQ)]


# global variable --------------------------------------------------------------
GTIME0 = GTIME(time=0, sec=0.0)

# satellite number -------------------------------------------------------------
def satno(sys, prn):
    return librtk.satno(c_int32(sys), c_int32(prn))

# satellite system -------------------------------------------------------------
def satsys(sat):
    prn = c_int32()
    sys = librtk.satsys(c_int32(sat), byref(prn))
    return sys, prn.value

# satellite ID to number -------------------------------------------------------
def satid2no(id):
    return librtk.satid2no(c_char_p(id.encode()))

# satellite number to ID -------------------------------------------------------
def satno2id(sat):
    buff = create_string_buffer(16)
    librtk.satno2id(c_int32(sat), buff)
    return buff.value.decode()

# obs type string to code ------------------------------------------------------
def obs2code(obs):
    librtk.obs2code.restype = c_uint8
    return librtk.obs2code(c_char_p(obs.encode()))

# obs code to type string ------------------------------------------------------
def code2obs(code):
    librtk.code2obs.restype = c_char_p
    return librtk.code2obs(c_uint8(code)).decode()

# epoch to time ----------------------------------------------------------------
def epoch2time(ep):
    epoch = np.zeros(6)
    epoch[:len(ep)] = ep
    p = epoch.ctypes.data_as(POINTER(c_double))
    librtk.epoch2time.restype = GTIME
    return librtk.epoch2time(p)

# time to epoch ----------------------------------------------------------------
def time2epoch(time):
    ep = np.zeros(6, dtype='double')
    p = ep.ctypes.data_as(POINTER(c_double))
    librtk.time2epoch.argtypes = [GTIME, c_void_p]
    librtk.time2epoch(time, p)
    return ep

# GPS week and tow to time -----------------------------------------------------
def gpst2time(week, sec):
    librtk.gpst2time.restype = GTIME
    return librtk.gpst2time(c_int32(week), c_double(sec))

# time to GPS week and tow -----------------------------------------------------
def time2gpst(time):
    week = c_int32()
    librtk.time2gpst.argtypes = [GTIME, POINTER(c_int32)]
    librtk.time2gpst.restype = c_double
    sec = librtk.time2gpst(time, byref(week))
    return week.value, sec

# GPS time to UTC --------------------------------------------------------------
def gpst2utc(time):
    librtk.gpst2utc.argtypes = [GTIME]
    librtk.gpst2utc.restype = GTIME
    return librtk.gpst2utc(time)

# UTC to GPS time --------------------------------------------------------------
def utc2gpst(time):
    librtk.utc2gpst.argtypes = [GTIME]
    librtk.utc2gpst.restype = GTIME
    return librtk.utc2gpst(time)

# add time ---------------------------------------------------------------------
def timeadd(time, sec):
    librtk.timeadd.argtypes = [GTIME, c_double]
    librtk.timeadd.restype = GTIME
    return librtk.timeadd(time, sec)

# time difference --------------------------------------------------------------
def timediff(time1, time2):
    librtk.timediff.argtypes = [GTIME, GTIME]
    librtk.timediff.restype = c_double
    return librtk.timediff(time1, time2)

# time to time string ----------------------------------------------------------
def time2str(time, n):
    buff = create_string_buffer(32 + n)
    librtk.time2str.argtypes = [GTIME, c_char_p, c_int32]
    librtk.time2str(time, buff, n)
    return buff.value.decode()

# get current time in UTC ------------------------------------------------------
def timeget():
    librtk.timeget.restype = GTIME
    return librtk.timeget()

# trace open -------------------------------------------------------------------
def traceopen(file):
    if 'Windows' in env:
        file = file.replace('/', '\\')
    librtk.traceopen(c_char_p(file.encode()))

# trace close ------------------------------------------------------------------
def traceclose():
    librtk.traceclose()

# trace level ------------------------------------------------------------------
def tracelevel(level):
    librtk.tracelevel(c_int32(level))

# satellite azimuth and elevation angle ----------------------------------------
def satazel(pos, e):
    if len(pos) < 3 or len(e) < 3:
        return 0.0, 0.0
    azel = np.zeros(2)
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(e, dtype='double').ctypes.data_as(POINTER(c_double))
    p3 = azel.ctypes.data_as(POINTER(c_double))
    librtk.satazel.restype = c_double
    el = librtk.satazel(p1, p2, p3)
    return azel

# geometric distance -----------------------------------------------------------
def geodist(rs, rr):
    if len(rs) < 3 or len(rr) < 3:
        return 0.0, np.zeros(3)
    e = np.zeros(3)
    p1 = np.array(rs, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(rr, dtype='double').ctypes.data_as(POINTER(c_double))
    p3 = e.ctypes.data_as(POINTER(c_double))
    librtk.geodist.restype = c_double
    r = librtk.geodist(p1, p2, p3)
    return r, e

# iono model with navigation data ----------------------------------------------
def ionmodel(time, nav, pos, azel):
    if len(pos) < 3 or len(azel) < 2 or not nav:
        return 0.0
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(azel, dtype='double').ctypes.data_as(POINTER(c_double))
    librtk.ionmodel_nav.argtypes = [GTIME, c_void_p, c_void_p, c_void_p]
    librtk.ionmodel_nav.restype = c_double
    return librtk.ionmodel_nav(time, nav, p1, p2)

# troposhere model -------------------------------------------------------------
def tropmodel(time, pos, azel, humi=0.7):

    # python datetime to GTIME
    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])

    if len(pos) < 3 or len(azel) < 2:
        return 0.0
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(azel, dtype='double').ctypes.data_as(POINTER(c_double))
    librtk.tropmodel.argtypes = [GTIME, c_void_p, c_void_p, c_double]
    librtk.tropmodel.restype = c_double
    return librtk.tropmodel(time, p1, p2, humi)

# get TGD in m -----------------------------------------------------------------
def get_tgd(sat, nav):
    librtk.navgettgd.restype = c_double
    return librtk.navgettgd(c_int32(sat), c_void_p(nav))

# extract unsigned bits --------------------------------------------------------
def getbitu(data, pos, len):
    if data.dtype != 'uint8':
        return 0
    librtk.getbitu.restype = c_uint32
    p = data.ctypes.data_as(POINTER(c_uint8))
    return librtk.getbitu(p, c_int(pos), c_int(len))

# extract signed bits ----------------------------------------------------------
def getbits(data, pos, len):
    if data.dtype != 'uint8':
        return 0
    librtk.getbits.restype = c_int32
    p = data.ctypes.data_as(POINTER(c_uint8))
    return librtk.getbits(p, c_int(pos), c_int(len))

# set unsigned bits ------------------------------------------------------------
def setbitu(data, pos, len, val):
    if data.dtype != 'uint8':
        return
    p = data.ctypes.data_as(POINTER(c_uint8))
    librtk.setbitu(p, c_int(pos), c_int(len), c_uint32(val))

# set signed bits --------------------------------------------------------------
def setbits(data, pos, len, val):
    if data.dtype != 'uint8':
        return
    p = data.ctypes.data_as(POINTER(c_uint8))
    librtk.setbits(p, c_int(pos), c_int(len), c_int32(val))

# CRC 16 -----------------------------------------------------------------------
def crc16(data, len):
    if data.dtype != 'uint8':
        return 0
    librtk.rtk_crc16.restype = c_uint32
    p = data.ctypes.data_as(POINTER(c_uint8))
    return librtk.rtk_crc16(p, c_int(len))

# CRC 24Q ----------------------------------------------------------------------
def crc24q(data, len):
    if data.dtype != 'uint8':
        return 0
    librtk.rtk_crc24q.restype = c_uint32
    p = data.ctypes.data_as(POINTER(c_uint8))
    return librtk.rtk_crc24q(p, c_int(len))

# CRC 32 -----------------------------------------------------------------------
def crc32(data, len):
    if data.dtype != 'uint8':
        return 0
    librtk.rtk_crc32.restype = c_uint32
    p = data.ctypes.data_as(POINTER(c_uint8))
    return librtk.rtk_crc32(p, c_int(len))

# test GLONASS string ----------------------------------------------------------
def test_glostr(data):
    if data.dtype != 'uint8':
        return 0
    librtk.test_glostr.restype = c_int32
    p = data.ctypes.data_as(POINTER(c_uint8))
    return librtk.test_glostr(p)

# open stream ------------------------------------------------------------------
def stropen(type, mode, path):
    librtk.strnew.restype = c_void_p;
    stream = librtk.strnew()
    if not stream:
        return None
    if not librtk.stropen(c_void_p(stream), c_int32(type), c_int32(mode),
               c_char_p(path.encode())):
        print('stropen error mode={:d} path={:s}'.format(mode, path))
        librtk.strfree(c_void_p(stream))
        return None
    return stream
        
# close stream -----------------------------------------------------------------
def strclose(stream):
    if str != None:
        librtk.strclose(c_void_p(stream))
        librtk.strfree(c_void_p(stream))

# write stream -----------------------------------------------------------------
def strwrite(stream, buff):
    if str == None or buff.dtype != 'uint8':
        return 0
    p = buff.ctypes.data_as(POINTER(c_uint8))
    return librtk.strwrite(c_void_p(stream), p, len(buff))

# read stream ------------------------------------------------------------------
def strread(stream, buff):
    if stream == None or buff.dtype != 'uint8':
        return 0
    p = buff.ctypes.data_as(POINTER(c_uint8))
    return librtk.strread(c_void_p(stream), p, len(buff))

# write line to stream ---------------------------------------------------------
def strwritel(stream, line):
    if stream == None:
        return 0
    buff = create_string_buffer(line)
    return librtk.strwrite(c_void_p(stream), buff, len(line))

# read line from stream --------------------------------------------------------
def strreadl(stream):
    if stream == None:
        return 0
    buff = create_string_buffer(1)
    line = ''
    while True:
        if librtk.strread(c_void_p(stream), buff, 1) == 0:
            time.sleep(0.01)
            continue
        line += buff.value.decode()
        if line[-1] == '\n':
            return line

# get stream status ------------------------------------------------------------
def strstat(stream):
    msg = create_string_buffer(256)
    stat = librtk.strstat(c_void_p(stream), msg)
    return stat, msg.value.decode()

# get stream summary -----------------------------------------------------------
def strsum(stream):
    inb, inr, outb, outr = c_int(), c_int(), c_int(), c_int()
    librtk.strsum(c_void_p(stream), byref(inb), byref(inr), byref(outb),
        byref(outr))
    return inb.value, inr.value, outb.value, outr.value

# read RINEX -------------------------------------------------------------------
def readrnx(file, rcv=1, ts=GTIME(), te=GTIME(), tint=0.0, opt=''):
    if 'Windows' in env:
        file = file.replace('/', '\\')
    librtk.obsnew.restype = c_void_p
    librtk.navnew.restype = c_void_p
    obs = librtk.obsnew()
    nav = librtk.navnew()
    librtk.readrnxt.argtypes = [c_char_p, c_int32, GTIME, GTIME, c_double,
        c_char_p, c_void_p, c_void_p, c_void_p]
    
    if not librtk.readrnxt(file.encode(), rcv, ts, te, tint, opt.encode(), obs,
        nav, None):
        return None, None
    return obs, nav

# free OBS data ----------------------------------------------------------------
def obsfree(obs):
    librtk.obsfree(c_void_p(obs))

# free NAV data ----------------------------------------------------------------
def navfree(nav):
    librtk.navfree(c_void_p(nav))

# get OBS data (generator function) --------------------------------------------
def obsget(obs):
    librtk.obsget.argtypes = [c_void_p, c_int32]
    librtk.obsget.restype = POINTER(OBSD)
    for i in range(10000000):
        data = librtk.obsget(obs, i)
        if data:
            yield data[0]
        else:
            break

# get ephemeris (generator function) -------------------------------------------
def ephget(nav):
    librtk.navgeteph.argtypes = [c_void_p, c_int32]
    librtk.navgeteph.restype = POINTER(EPH)
    for i in range(10000000):
        eph = librtk.navgeteph(nav, i)
        if eph:
            yield eph[0]
        else:
            break

# get GLONASS ephemeris (generator function) -----------------------------------
def gephget(nav):
    librtk.navgetgeph.argtypes = [c_void_p, c_int32]
    librtk.navgetgeph.restype = POINTER(GEPH)
    for i in range(10000000):
        geph = librtk.navgetgeph(nav, i)
        if geph:
            yield geph[0]
        else:
            break

# satellite position and clock by navigation data ------------------------------
def satpos(time, teph, sat, nav, ephopt=EPHOPT_PREC):

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    teph = epoch2time([teph.year, teph.month, teph.day, teph.hour, teph.minute, teph.second + teph.microsecond * 1e-6])
    rs = np.zeros(6, dtype='double')
    dts = np.zeros(2, dtype='double')
    var = c_double()
    svh = c_int32()
    librtk.satpos.argtypes = [GTIME, GTIME, c_int32, c_int32, c_void_p,
        c_void_p, c_void_p, POINTER(c_double), POINTER(c_int32)]
    p1 = rs.ctypes.data_as(POINTER(c_double))
    p2 = dts.ctypes.data_as(POINTER(c_double))
    if not librtk.satpos(time, teph, sat, ephopt, nav, p1, p2, byref(var),
        byref(svh)):
        return np.zeros(6), np.zeros(2), 0.0, 1
    return rs, dts, var.value, svh.value

# ECEF potition to latitude/longitude/height -----------------------------------
def ecef2pos(r):
    if len(r) < 3:
        return np.zeros(3)
    pos = np.zeros(3, dtype='double')
    p1 = np.array(r, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = pos.ctypes.data_as(POINTER(c_double))
    librtk.ecef2pos(p1, p2)
    return pos

# latitude/longitude/height to ECEF position -----------------------------------
def pos2ecef(pos):
    if len(pos) < 3:
        return np.zeros(3)
    r = np.zeros(3, dtype='double')
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = r.ctypes.data_as(POINTER(c_double))
    librtk.pos2ecef(p1, p2)
    return r

# ECEF position to ENU position ------------------------------------------------
def ecef2enu(pos, r):
    if len(pos) < 3 or len(r) < 3:
        return np.zeros(3)
    e = np.zeros(3, dtype='double')
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(r, dtype='double').ctypes.data_as(POINTER(c_double))
    p3 = e.ctypes.data_as(POINTER(c_double))
    librtk.ecef2enu(p1, p2, p3)
    return e

# ENU position to ECEF position ------------------------------------------------
def enu2ecef(pos, e):
    if len(pos) < 3 or len(e) < 3:
        return np.zeros(3)
    r = np.zeros(3, dtype='double')
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(e, dtype='double').ctypes.data_as(POINTER(c_double))
    p3 = r.ctypes.data_as(POINTER(c_double))
    librtk.enu2ecef(p1, p2, p3)
    return r

# read ANTEX -------------------------------------------------------------------
def readpcv(file, nav, time):
    if 'Windows' in env:
        file = file.replace('/', '\\')
    librtk.pcvnew.restype = c_void_p
    pcvs = librtk.pcvnew()
    librtk.readpcv.argtypes = [c_char_p, c_void_p]
    
    if not librtk.readpcv(file.encode(), pcvs):
        return None
    
    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    librtk.setsatpcv.argtypes = [GTIME, c_void_p, c_void_p]
    librtk.setsatpcv(time, nav, pcvs)

    return pcvs

# search antenna parameters ----------------------------------------------------
def searchpcv(sat, anttype, time, pcvs):

    librtk.searchpcv.restype = POINTER(PCV)
    librtk.searchpcv.argtypes = [c_int32, c_char_p, GTIME, c_void_p]

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    pcv = librtk.searchpcv(sat, anttype.encode(), time, pcvs)
    if not pcv:
        return None
    return pcv[0]

# receiver antenna model ------------------------------------------------------
def antmodel(pcv, delta, azel, opt):

    librtk.antmodel.restype = c_void_p
    librtk.antmodel.argtypes = [POINTER(PCV), POINTER(c_double), POINTER(c_double),
    c_int32, POINTER(c_double)]
    p1 = np.array(delta, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(azel , dtype='double').ctypes.data_as(POINTER(c_double))
    dant = np.zeros(NFREQ).ctypes.data_as(POINTER(c_double))
    librtk.antmodel(pcv, p1, p2, opt, dant)

    return np.array([dant[x] for x in range(NFREQ)])

# satellite antenna model -----------------------------------------------------
def antmodel_s(pcv, rs, rr):

    ru = rr - rs
    rz = -rs
    eu = (ru) / np.linalg.norm(ru)
    ez = (rz) / np.linalg.norm(rz)
    cosa = np.dot(eu, ez)
    if cosa < -1.0:
        cosa = -1.0
    elif cosa > 1.0:
        cosa = 1.0
    else:
        cosa = cosa
    nadir = np.arccos(cosa)
    
    librtk.antmodel_s.argtypes = [POINTER(PCV), c_double, POINTER(c_double)]
    librtk.antmodel_s.restype = c_void_p
    dant = np.zeros(NFREQ).ctypes.data_as(POINTER(c_double))
    librtk.antmodel_s(pcv, nadir, dant)

    return np.array([dant[x] for x in range(NFREQ)])

# troposhere mapping function -------------------------------------------------
def tropmapf(time, pos, azel):

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])

    if len(pos) < 3 or len(azel) < 2:
        return 0.0
    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(azel, dtype='double').ctypes.data_as(POINTER(c_double))
    mapfw = c_double()
    librtk.tropmapf.argtypes = [GTIME, c_void_p, c_void_p, POINTER(c_double)]
    librtk.tropmapf.restype = c_double
    mapfh = librtk.tropmapf(time, p1, p2, byref(mapfw))

    return mapfh, mapfw.value

# phase windup correction -----------------------------------------------------
def windupcorr(time, sat, rs, rr, prev_pwu):

    librtk.model_phw.argtypes = [GTIME, c_int, c_char_p, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    librtk.model_phw.restype = c_int

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    rs = rs.ctypes.data_as(POINTER(c_double))
    rr = rr.ctypes.data_as(POINTER(c_double))

    phw = c_double(prev_pwu)
    librtk.model_phw(time, sat, ''.encode(), 1, rs, rr, phw)

    return phw.value

# solid earth tides correction ------------------------------------------------
def tides(time, rr):
    
    librtk.tidedisp.argtypes = [GTIME, POINTER(c_double), c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    librtk.tidedisp.restype = c_void_p

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    tutc = gpst2utc(time)
    rr = rr.ctypes.data_as(POINTER(c_double))
    dr = np.zeros(3).ctypes.data_as(POINTER(c_double))
    librtk.tidedisp(tutc, rr, 1, None, None, dr)

    return dr

# satellite clock bias by navigation data ------------------------------
def ephclk(time, teph, sat, nav):

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    teph = epoch2time([teph.year, teph.month, teph.day, teph.hour, teph.minute, teph.second + teph.microsecond * 1e-6])
    dt = c_double()
    librtk.ephclk.argtypes = [GTIME, GTIME, c_int32, c_void_p, POINTER(c_double)]
    librtk.ephclk(time, teph, sat, nav, byref(dt))
    
    return dt.value

# read SP3 -------------------------------------------------------------------
def precehp(file, nav):
    if 'Windows' in env:
        file = file.replace('/', '\\')

    librtk.preceph.argtypes = [c_char_p, c_void_p]
    librtk.preceph(file.encode(), nav)

# read clk -------------------------------------------------------------------
def precclk(file, nav):
    if 'Windows' in env:
        file = file.replace('/', '\\')

    librtk.precclk.argtypes = [c_char_p, c_void_p]
    librtk.precclk(file.encode(), nav)

# read ionex -----------------------------------------------------------------
def readtec(file, nav):
    if 'Windows' in env:
        file = file.replace('/', '\\')

    librtk.readtec.argtypes = [c_char_p, c_void_p, c_int32]
    librtk.readtec(file.encode(), nav, 1)

# interpolate tec ------------------------------------------------------------
def iontec(time, nav, pos, azel):

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])

    if len(pos) < 3 or len(azel) < 2:
        return 0.0

    p1 = np.array(pos, dtype='double').ctypes.data_as(POINTER(c_double))
    p2 = np.array(azel, dtype='double').ctypes.data_as(POINTER(c_double))

    delay = c_double()
    var   = c_double()

    librtk.iontec.argtypes = [GTIME, c_void_p, c_void_p, c_void_p, c_int32, POINTER(c_double), POINTER(c_double)]
    librtk.iontec(time, nav, p1, p2, 1, delay, var)

    return delay.value, var.value

# convert utc to gmst --------------------------------------------------------
def utc2gmst(time):

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])

    librtk.utc2gmst.argtypes = [GTIME, c_double]
    librtk.utc2gmst.restype = c_double

    gmst = librtk.utc2gmst(time, 0.0)

    return gmst

# eclipse --------------------------------------------------------------------
def testeclipse(time, sat, nav, rs):

    time = epoch2time([time.year, time.month, time.day, time.hour, time.minute, time.second + time.microsecond * 1e-6])
    rs = np.array(rs, dtype='double').ctypes.data_as(POINTER(c_double))

    librtk.testeclipse_.argtypes = [GTIME, c_int, c_void_p, c_void_p]
    librtk.testeclipse_.restype = c_int

    return librtk.testeclipse_(time, sat, nav, rs)

# resolve integer ambiguities with LAMBDA ------------------------------------
def lambda_method(afloat, covfloat):

    librtk.lambda_.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    librtk.lambda_.restype = c_int

    n = afloat.shape[0]

    afloat = afloat.ctypes.data_as(POINTER(c_double))
    q = covfloat.reshape(-1, order='F').ctypes.data_as(POINTER(c_double))
    F = np.zeros(n * 3).ctypes.data_as(POINTER(c_double))
    s = np.zeros(3).ctypes.data_as(POINTER(c_double))

    info = librtk.lambda_(n, 3, afloat, q, F, s)

    F = np.ctypeslib.as_array(F, shape=(n * 3,))
    F = np.reshape(F, (n, 3), order='F')
    s = np.ctypeslib.as_array(s, shape=(3,))

    return F, s

# LAMBDA reduction -----------------------------------------------------------
def lambda_reduction(Q):

    librtk.lambda_reduction.argtypes = [c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
    librtk.lambda_reduction.restype = c_int

    n = Q.shape[0]

    Q = Q.reshape(-1, order='F').ctypes.data_as(POINTER(c_double))
    Z = np.eye(n * n).ctypes.data_as(POINTER(c_double))
    L = np.zeros(n * n).ctypes.data_as(POINTER(c_double))
    D = np.zeros(n * 1).ctypes.data_as(POINTER(c_double))

    info = librtk.lambda_reduction(n, Q, Z, L, D)

    Z = np.ctypeslib.as_array(Z, shape=(n * n,))
    Z = np.reshape(Z, (n, n), order='F')

    L = np.ctypeslib.as_array(L, shape=(n * n,))
    L = np.reshape(L, (n, n), order='F')

    D = np.ctypeslib.as_array(D, shape=(n * 1,))
    D = np.reshape(D, (n, 1), order='F')

    return Z, L, D

