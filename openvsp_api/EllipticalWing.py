import os
import openvsp as vsp
import numpy as np
import pandas as pd

#Set up Variables
halfSpan = 5.0 # half_span
rootChord = 1.5915 # root chord
n = 2.1 # super ellipse exp
sweepLoc = 0.25 # chord location of sweep
sweep = 0 # deg
growthRate = 1.4
numPoints = 22 # number of span cross sections + 1
AFname = 'e193.dat'

AoAStart = 8.284
AoAEnd = 8.284
AlphaNpts = 1
Xref = rootChord/4
Sref = 12.5

def genEllipticalWing (halfSpan,rootChord,n,sweepLoc,sweep,growthRate,numPoints,AFname, tc=0.05):
    vsp.VSPRenew()
    # == x is the spanwise direction ==
    x = []
    Cx = []
    step = halfSpan / (numPoints - 1);
    maxVal = (numPoints - 1) ** (1/growthRate) # This is to normalize the last value to 'p'
    for i in range(numPoints):
        x.append(((i**(1/growthRate))/maxVal)*halfSpan) # span location with growth bias
    
    #==== Add Wing ====
    wid = vsp.AddGeom( "WING", "" );
    vsp.SetGeomName( wid, "Wing_n"+str(int(n*10)));
    
    
    #===== Insert Extra Sections to total numPoints=====
    for i in range(numPoints-1):
        vsp.InsertXSec( wid, 1, vsp.XS_FILE_AIRFOIL)
        #string xsec_surf = GetXSecSurf(wid, 0 );
        #string xsec = GetXSec(xsec_surf, 1 );
        #ReadFileAirfoil(xsec, "e193.dat" );
        vsp.Update()
    
    #===== Cut The Original Section =====//
    vsp.CutXSec( wid, 1 )
    xsec_surf = vsp.GetXSecSurf(wid, 0 )
    vsp.ChangeXSecShape(xsec_surf, 0, vsp.XS_FILE_AIRFOIL) # Change root section to AF file
    xsec = vsp.GetXSec(xsec_surf, 0)
    vsp.ReadFileAirfoil(xsec, AFname)
    vsp.Update()
    
    #===== Change Driver =====//
    vsp.SetDriverGroup( wid, 1, 1, 5, 6)
    
    for i in range(1,numPoints):
        xsec = vsp.GetXSec(xsec_surf, i)
        vsp.ReadFileAirfoil(xsec, AFname)
        Croot = rootChord * ((1 - (abs(x[i-1]/halfSpan))** n)** (1/n)) # root chord of section
        xsecstr = "XSec_"+ str(i)
        vsp.SetParmVal( wid, "Root_Chord", xsecstr, Croot )
        vsp.SetParmVal( wid, "Sweep", xsecstr, sweep );
        vsp.SetParmVal( wid, "Sweep_Location", xsecstr, sweepLoc );
        vsp.SetParmVal( wid, "Span",  xsecstr, (x[i] - x[i-1]) );
        if i == numPoints-1:
            Ctip = rootChord * ((1 - (abs(x[i]/halfSpan))** n)** (1/n)) # tip chord
            vsp.SetParmVal( wid, "Tip_Chord",  xsecstr, Ctip+tc*rootChord )
        vsp.Update()
    S = vsp.GetParmVal( wid, "TotalArea", "WingGeom")
    print('Wing generated with', numPoints, 'points', 'Area = ',S)
    return S

def chordFinder(Sfix, n, rootChord = 1.5, tol = 0.0001):
    err = 1
    while abs(err) >0.0001:
        print(rootChord)
        S = genEllipticalWing (halfSpan,rootChord,n,sweepLoc,sweep,growthRate,numPoints,AFname)
        err = (Sfix - S)/Sfix
        rootChord = rootChord + rootChord * err
    return rootChord
    
def analyseVLM(AoAStart, AoAEnd,AlphaNpts,Xref,VLM = True, Sref=False):
    analysis_name = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(analysis_name)
    analysis_method = list(vsp.GetIntAnalysisInput(analysis_name, "AnalysisMethod" ))
    if VLM:
        analysis_method[0] = vsp.VORTEX_LATTICE
    else:
        analysis_method[0] = vsp.PANEL
    vsp.SetIntAnalysisInput(analysis_name, "AnalysisMethod", analysis_method)
    res_id = vsp.ExecAnalysis( analysis_name )
    analysis_name = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysis_name)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaStart", (AoAStart,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "AlphaEnd", (AoAEnd,), 0)
    vsp.SetIntAnalysisInput(analysis_name, "AlphaNpts", (AlphaNpts,), 0)
    vsp.SetIntAnalysisInput(analysis_name, "NCPU", (16,), 0)
    vsp.SetDoubleAnalysisInput(analysis_name, "Xcg", (Xref,), 0)
    if Sref:
        vsp.SetDoubleAnalysisInput(analysis_name, "Sref", (Sref,), 0)
    else:
        vsp.SetIntAnalysisInput(analysis_name, "RefFlag", (1,), 0)
    
    #vsp.SetIntAnalysisInput(analysis_name, "MachNpts", (1,), 0)
    vsp.Update()
    vsp.DeleteAllResults()
    res_id = vsp.ExecAnalysis(analysis_name)
    return res_id

def getResults():
    history_res = True
    res = {'CL':[],'CD':[],'L2D':[],'CMy':[],'AoA':[]}
    i=0
    while history_res:
        history_res = vsp.FindResultsID("VSPAERO_History",i)
        if history_res:
            res['CL'].append(vsp.GetDoubleResults(history_res, "CL", 0)[-1])
            res['CD'].append(vsp.GetDoubleResults(history_res, "CDtot", 0)[-1])
            res['L2D'].append(vsp.GetDoubleResults(history_res, "L/D", 0)[-1])
            res['CMy'].append(vsp.GetDoubleResults(history_res, "CMy", 0)[-1])
            res['AoA'].append(vsp.GetDoubleResults(history_res, "Alpha", 0)[-1])
        i+=1
    return res


path = os.getcwd()
fname = '/elliptical/blank.vsp3'
vsp.VSPRenew()
vsp.ReadVSPFile((path+fname))

CL6,CL8,CD6,CD8 = [],[],[],[]

S = 12.5
rootChord = 2

for i in np.arange(0.8,4.0,0.2):
    n = i
    vsp.VSPRenew()
    vsp.ReadVSPFile((path+fname))
    rootChord = chordFinder(S, n, rootChord)
    genEllipticalWing (halfSpan,rootChord,n,sweepLoc,sweep,growthRate,numPoints,AFname)
    analyseVLM(AoAStart, AoAEnd,AlphaNpts,Xref,VLM = False, Sref = Sref)
    res = getResults()
    #CL6.append(res['CL'][0])
    CL8.append(res['CL'][0])
    #CD6.append(res['CD'][0])
    CD8.append(res['CD'][0])
    fnamenew = '/elliptical/elliptical'+str(round(i,1))+'.vsp3'
    vsp.WriteVSPFile(path+fnamenew)


