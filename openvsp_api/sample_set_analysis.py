import openvsp as vsp
import numpy as np
from cst_modeling.section import cst_foil, cst_curve, cst_foil_fit, foil_bump_modify, foil_increment
from matplotlib import pyplot as plt
#import PyQt5
import os
import pandas as pd
from scipy.stats import qmc


AoAStart = 5
AoAEnd = 5
AlphaNpts = 1



def analyse_VLM(
        AoAStart, 
        AoAEnd,
        AlphaNpts,
        Xref,
        Sref = False
    ):
    analysisName = "VSPAEROComputeGeometry"
    vsp.SetAnalysisInputDefaults(analysisName)
    analysisMethod = list(vsp.GetIntAnalysisInput(analysisName, "AnalysisMethod" ))
    analysisMethod[0] = vsp.VORTEX_LATTICE
    vsp.SetIntAnalysisInput(analysisName, "AnalysisMethod", analysisMethod)

    res_id = vsp.ExecAnalysis(analysisName)
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

    if Sref:
        vsp.SetDoubleAnalysisInput(analysisName, "Sref", (Sref,), 0)
    else:
        vsp.SetIntAnalysisInput(analysisName, "RefFlag", (1,), 0)
    
    
    vsp.Update()
    vsp.DeleteAllResults()
    res_id = vsp.ExecAnalysis(analysisName)
    return res_id

def sampleset_analysis(
        directory,
        AoAStart, 
        AoAEnd,
        AlphaNpts,
        Sref=False
    ):
    vsp.VSPRenew()
    vsp.ReadVSPFile(str(directory))
    geoms = vsp.FindGeoms()
    wing = geoms[0]
    c_id = vsp.GetParm(wing, "Root_Chord", "XSec_1")
    rootChord = vsp.GetParmVal(c_id)
    Xref = rootChord/4
    geoms = vsp.FindGeoms()
    areaID = vsp.GetParm(wing, "Area", "XSec_1")
    area = vsp.GetParmVal(areaID)
    print(area*2)
    analyse_VLM(AoAStart, AoAEnd, AlphaNpts, Xref, Sref = area*2)

def analysis_plots(
            xValues:list[float] = None,
            yValues:list[float] = None,
            pointLabels:list[str] = None,
            imageName:str = None,
            directory:str = None,
            plotTitle:str = "", 
            xLabel:str = "",
            yLabel:str = "",
    ):
    plt.scatter(xValues, yValues)
    plt.title(plotTitle)
    for label, x, y in zip(pointLabels, xValues, yValues):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(os.path.join(directory + imageName))
    #plt.show()

def create_data_for_plotting(
        inputVSPFile:str = None,
        inputVariable:str = None,
        inputXSecName:str= None,
        outputPolarFile:str = None,
        outputVariable:str = None
    ):
    vsp.VSPRenew()
    vsp.ReadVSPFile(str(inputVSPFile))
    geoms = vsp.FindGeoms()
    wing = geoms[0]
    id = vsp.GetParm(wing, inputVariable, inputXSecName)
    xValue = vsp.GetParmVal(id)

    with open(outputPolarFile, 'r') as file:
        lines = file.readlines()
    data_line = lines[1]

    # Split the data line into individual values
    values = data_line.split()

    # Find the index of the 'L/D' column in the header
    header_line = lines[0]
    header_values = header_line.split()
    index = header_values.index(outputVariable)

    # Extract the L/D value using the index
    yValue = float(values[index])

    return xValue, yValue
    

cwd = os.getcwd()
xValues = []
yValues = []
pointLabels = []
noSampleSets = 100

def multiple_sample_analysis(overwrite = False):
    for i in range(noSampleSets):
        inputFilepath = os.path.join(cwd, f"openvsp_api/sample_set/wing_geom_and_analysis_{i}/wing_geom.vsp3")
        outputFilepath = os.path.join(cwd, f"openvsp_api/sample_set/wing_geom_and_analysis_{i}/wing_geom_DegenGeom.polar")
        
        if overwrite or not os.path.exists(outputFilepath):
            print("Analysis started")
            sampleset_analysis(inputFilepath, AoAStart, AoAEnd, AlphaNpts)
            print(f"Sample Set {i} done")
        else:
            print(f"Analysis for Sample Set {i} already completed")

        xValue, yValue = create_data_for_plotting(
                                    inputVSPFile = inputFilepath,
                                    inputVariable = "Aspect",
                                    inputXSecName = "XSec_1",
                                    outputPolarFile = outputFilepath,
                                    outputVariable = "L/D"
                                    )
            
        xValues.append(xValue)
        yValues.append(yValue)
        pointLabels.append(str(i))

multiple_sample_analysis()

analysis_plots(
    xValues = xValues,
    yValues = yValues,
    pointLabels = pointLabels,
    imageName = "Aspect_Ratio_vs_LD",
    directory = os.path.join(cwd, f"openvsp_api/sample_set"),
    plotTitle = "Aspect Ratio vs the Lift Drag Ratio for the Generated Sample Set",
    xLabel = "Half Wing Aspect Ratio",
    yLabel = "Lift Drag Ratio"
)