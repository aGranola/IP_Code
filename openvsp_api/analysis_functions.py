import openvsp as vsp

def get_Xref_and_Sref(vsp_input_file:str) -> tuple[float]:
    vsp.VSPRenew()
    vsp.ReadVSPFile(vsp_input_file)
    geoms = vsp.FindGeoms()
    wing = geoms[0]
    # calculate Xref
    c_id = vsp.GetParm(wing, "Root_Chord", "XSec_1")
    rootChord = vsp.GetParmVal(c_id)
    Xref = rootChord/4
    # calcuklate Sref
    areaID = vsp.GetParm(wing, "Area", "XSec_1")
    area = vsp.GetParmVal(areaID)
    Sref = area*2
    
    return Xref, Sref

def analyse_VLM(
        AoAStart, 
        AoAEnd,
        AlphaNpts,
        Xref,
        Sref
    ):
    # calculate geom
    analysisName = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(analysisName)
    analysisMethod = list(vsp.GetIntAnalysisInput(analysisName, "AnalysisMethod" ))
    analysisMethod[0] = vsp.VORTEX_LATTICE
    vsp.SetIntAnalysisInput(analysisName, "AnalysisMethod", analysisMethod)
    vsp.ExecAnalysis(analysisName)
    
    # calculate sweep
    analysisName = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(analysisName)
    vsp.SetDoubleAnalysisInput(analysisName, "AlphaStart", (AoAStart,), 0)
    vsp.SetDoubleAnalysisInput(analysisName, "AlphaEnd", (AoAEnd,), 0)
    vsp.SetIntAnalysisInput(analysisName, "AlphaNpts", (AlphaNpts,), 0)
    vsp.SetIntAnalysisInput(analysisName, "NCPU", (16,), 0)
    vsp.SetDoubleAnalysisInput(analysisName, "Xcg", (Xref,), 0)
    vsp.SetDoubleAnalysisInput(analysisName, "MachStart", (0.5,), 0)
    vsp.SetDoubleAnalysisInput(analysisName, "MachEnd", (0.5,), 0)
    vsp.SetIntAnalysisInput(analysisName, "MachNpts", (1,), 0)
    vsp.SetDoubleAnalysisInput(analysisName, "ReCref", (45e06,), 0 )
    vsp.SetIntAnalysisInput(analysisName, "ReCrefNpts", (1,), 0)
    vsp.SetDoubleAnalysisInput(analysisName, "Sref", (Sref,), 0)
    vsp.Update()
    vsp.DeleteAllResults()
    vsp.ExecAnalysis(analysisName)
