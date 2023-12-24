from scipy.stats import qmc
import openvsp as vsp

def create_random_input_params(num_samples:int) -> list[list[float]]:
    #Create random distribution
    sampler = qmc.LatinHypercube(d=7)
    sample = sampler.random(n=num_samples)
    sampleLowerBounds = [17.1/2, 13.4, 0, 0, 0.05, 0, 0.25]
    sampleUpperBounds = [38.7/2, 26.4, 1, 5, 0.2, 0.089, 0.7]
    sample_scaled = qmc.scale(sample, sampleLowerBounds, sampleUpperBounds)
    return sample_scaled

def create_wing(
        outputFile:str,
        aspectRatio:float = 20.0,
        span:float = 20.0,
        taper:float = 0.5,
        sweep:float = 2.0,
        thickChord:float = 0.1,
        camber:float = 0.05,
        camberLoc:float = 0.5
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
    vsp.SetVSP3FileName(outputFile)
    vsp.WriteVSPFile(vsp.GetVSPFileName())
