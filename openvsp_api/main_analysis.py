from openvsp_api.generation_functions import create_random_input_params, create_wing
from openvsp_api.analysis_functions import get_Xref_and_Sref, analyse_VLM
from openvsp_api.plotting_functions import calculate_data_for_plotting, plot_analysis
import tempfile
import os

num_samples = 100
AoAStart = 5
AoAEnd = 5
AlphaNpts = 1
# create random input params
multiple_sample_params = create_random_input_params(num_samples)
# run analyses
xValues = []
yValues = []
pointLabels = []
outputParentDir = "sample_set"
for index, sample_params in enumerate(multiple_sample_params):
    print(f'Running analysis {index}')
    if outputParentDir:
        sample_output_dir = os.path.join(outputParentDir, f'sample_{i}')
    else:
        sample_output_dir = tempfile.TemporaryDirectory()
    vsp_file = os.path.join(sample_output_dir, 'wing_geom.vsp3')
    create_wing(
        aspectRatio = sample_params[0],
        span = sample_params[1],
        taper = sample_params[2],
        sweep = sample_params[3],
        thickChord = sample_params[4],
        camber = sample_params[5],
        camberLoc = sample_params[6],
        outputFile = vsp_file
    )
    Xref, Sref = get_Xref_and_Sref(vsp_file)
    analyse_VLM(AoAEnd,AlphaNpts,Xref,Sref)
    analysis_output_file = os.path.join(sample_output_dir, 'wing_geom_DegenGeom.polar')
    
    # get values for plotting
    xValue, yValue = calculate_data_for_plotting(
        inputVSPFile = vsp_file,
        inputVariable = "Aspect",
        inputXSecName = "XSec_1",
        outputPolarFile = analysis_output_file,
        outputVariable = "L/D"
    )
    xValues.append(xValue)
    yValues.append(yValue)
    pointLabels.append(str(index))
    # Clean up if a temporary directory was used
    if not outputParentDir:
        sample_output_dir.cleanup()

plot_analysis(
    xValues = xValues,
    yValues = yValues,
    pointLabels = pointLabels,
    output_file = os.path.join(os.cwd, 'Aspect_Ratio_vs_LD.png'),
    plotTitle = "Aspect Ratio vs the Lift Drag Ratio for the Generated Sample Set",
    xLabel = "Half Wing Aspect Ratio",
    yLabel = "Lift Drag Ratio"
)