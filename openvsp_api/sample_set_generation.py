from scipy.stats import qmc
import openvsp as vsp
import os

cwd = os.getcwd()
print(cwd)

#Create random distribution
sampler = qmc.LatinHypercube(d=7)
sample = sampler.random(n=100)
sampleLowerBounds = [17.1/2, 13.4, 0, 0, 0.05, 0, 0.25]
sampleUpperBounds = [38.7/2, 26.4, 1, 5, 0.2, 0.089, 0.7]
sample_scaled = qmc.scale(sample, sampleLowerBounds, sampleUpperBounds)
#Sample Parameters: Aspect Ratio, Span, Taper, Sweep, t/c, Camber, Camber Location
print(sample_scaled)

def create_wing(
        aspectRatio:float = 20.0,
        span:float = 20.0,
        taper:float = 0.5,
        sweep:float = 2.0,
        thickChord:float = 0.1,
        camber:float = 0.05,
        camberLoc:float = 0.5,
        outputFile:str = None
    ):
    stdout = vsp.cvar.cstdout
    errorMgr = vsp.ErrorMgrSingleton.getInstance()

    vsp.VSPCheckSetup()
    errorMgr.PopErrorAndPrint(stdout)

    vsp.VSPRenew()
    vsp.ClearVSPModel()
    #vsp.DeleteAllResults()

    #add wing component
    wing = vsp.AddGeom("WING", "")
    vsp.SetGeomName(wing, "Wing");

    vsp.SetDriverGroup(wing, 1, vsp.AR_WSECT_DRIVER, vsp.SPAN_WSECT_DRIVER, vsp.TAPER_WSECT_DRIVER)

    #define parameters
    vsp.SetParmVal(wing, "Aspect", "XSec_1", aspectRatio)
    vsp.SetParmVal(wing, "Span", "XSec_1", span)
    vsp.SetParmVal(wing, "Taper", "XSec_1", taper)
    vsp.SetParmVal(wing, "Sweep", "XSec_1", sweep)
    #vsp.InsertXSec(wing, 0, vsp.XS_FOUR_SERIES)
    vsp.SetParmVal(wing, "ThickChord", "XSecCurve_0", thickChord)
    vsp.SetParmVal(wing, "Camber", "XSecCurve_0", camber)
    vsp.SetParmVal(wing, "CamberLoc", "XSecCurve_0", camberLoc)

    vsp.Update()



    directory = os.path.dirname(outputFile)
    if not os.path.exists(directory):
        os.mkdir(directory)

    vsp.SetVSP3FileName(outputFile)
    vsp.WriteVSPFile(vsp.GetVSPFileName())


for i in range(len(sample_scaled)):
        outputFile = os.path.join(cwd, f"openvsp_api/sample_set/wing_geom_and_analysis_{i}", "wing_geom.vsp3")
        create_wing(
            aspectRatio = sample_scaled[i][0], 
            span = sample_scaled[i][1], 
            taper = sample_scaled[i][2], 
            sweep = sample_scaled[i][3],
            thickChord = sample_scaled[i][4], 
            camber = sample_scaled[i][5], 
            camberLoc = sample_scaled[i][6], 
            outputFile = outputFile
        )
        print(f"wing {i} created")
        vsp.VSPRenew()
        vsp.ReadVSPFile(outputFile)
        geoms = vsp.FindGeoms()
        wing = geoms[0]
        camberID = vsp.GetParm(wing, "Camber", "XSecCurve_0")
        camber = vsp.GetParmVal(camberID)
        print(f"Camber was set to be {sample_scaled[i][5]}. In the file it is {camber}")
