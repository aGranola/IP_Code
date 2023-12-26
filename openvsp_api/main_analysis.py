from generation_functions import create_random_input_params, create_wing
from analysis_functions import get_Xref_and_Sref, analyse_VLM
from plotting_functions import calculate_data_for_plotting, plot_analysis
from utility_functions import get_data_from_vsp_file, get_data_from_vlm_output, calculate_rmse
from model_functions import split_data_for_model, create_neural_network, train_neural_network, plot_loss
import openvsp as vsp
import tempfile
import os
import numpy as np

num_samples = 15
AoAStart = 5
AoAEnd = 5
AlphaNpts = 1
numEpochs=500
# create random input params
multiple_sample_params = create_random_input_params(num_samples)
# plotting variables
xValues = []
yValues = []
pointLabels = []
outputParentDir = "example_data"

# ML variables
MLInputData = []
MLOutputData = []

for index, sample_params in enumerate(multiple_sample_params):
    vsp.VSPRenew()

   
    if outputParentDir:
        sample_output_dir = os.path.join(outputParentDir, f'sample_{index}')
        os.makedirs(sample_output_dir, exist_ok=True)
    else:
        sample_output_dir = tempfile.TemporaryDirectory()
    vsp_file = os.path.join(sample_output_dir, 'wing_geom.vsp3')
    # create vsp file if it does not exist
    if not os.path.exists(vsp_file):
        print(f'Creating wing {index}')
        create_wing(
            outputFile = vsp_file,
            aspectRatio = sample_params[0],
            span = sample_params[1],
            taper = sample_params[2],
            sweep = sample_params[3],
            thickChord = sample_params[4],
            camber = sample_params[5],
            camberLoc = sample_params[6]
        )
    Xref, Sref = get_Xref_and_Sref(vsp_file)
    
    vlm_analysis_output_file = os.path.join(sample_output_dir, 'wing_geom_DegenGeom.polar')
    if not os.path.exists(vlm_analysis_output_file):
        print(f'Running analysis {index}')
        analyse_VLM(AoAStart, AoAEnd, AlphaNpts, Xref, Sref)
    else:
         print(f'Skipping analysis {index}')
    
    # get values for plotting
    xValue, yValue = calculate_data_for_plotting(
        inputVSPFile = vsp_file,
        inputVariable = "Aspect",
        inputXSecName = "XSec_1",
        outputPolarFile = vlm_analysis_output_file,
        outputVariable = "L/D"
    )
    xValues.append(xValue)
    yValues.append(yValue)
    pointLabels.append(str(index))
    
    # create data for ML model
    sampleInputVariable = get_data_from_vlm_output(vlm_analysis_output_file, 'L/D')
    MLInputData.append(sampleInputVariable)
    sampleOutputVariables = []
    for xsec_variable in ['Root_Chord', 'Tip_Chord', 'Span', 'Sweep']:
        sampleOutputVariables.append(get_data_from_vsp_file(vsp_file, xsec_variable, 'XSec_1'))
    for xsec_curve_variable in ['ThickChord', 'Camber', 'CamberLoc']:
        sampleOutputVariables.append(get_data_from_vsp_file(vsp_file, xsec_curve_variable, 'XSecCurve_0'))
    MLOutputData.append(sampleOutputVariables)

    # Clean up if a temporary directory was used
    if not outputParentDir:
        sample_output_dir.cleanup()

plot_analysis(
    xValues = xValues,
    yValues = yValues,
    pointLabels = pointLabels,
    output_file = os.path.join(os.getcwd(), 'Aspect_Ratio_vs_LD.png'),
    plotTitle = "Aspect Ratio vs the Lift Drag Ratio for the Generated Sample Set",
    xLabel = "Half Wing Aspect Ratio",
    yLabel = "Lift Drag Ratio"
)

# ML training
MLInputDataNp = np.array(MLInputData)
MLOutputDataNp = np.array(MLOutputData)
trainInput, valInput, testInput, trainOutput, valOutput, testOutput = split_data_for_model(MLInputDataNp, MLOutputDataNp)

# Define output ranges
output_ranges = [(17.1/2, 38.7/2), (13.4, 26.4), (0, 1), (0, 5), (0.05, 0.2), (0, 0.089), (0.25, 0.7)]
model = create_neural_network(len(trainOutput[1]), output_ranges)

hist = train_neural_network(model, trainInput, trainOutput, valInput, valOutput, testInput, numEpochs)
plot_loss(hist)
# predict outputs
predictedOutput = model.predict(testInput)
# calculate L/D for testOutput values
calculatedLDs = []
for index, sample_params in enumerate(predictedOutput):
    vsp.VSPRenew()
    sample_params_list = list(sample_params)
    sample_params_list = [float(f) for f in sample_params_list]
    sample_output_dir = os.path.join(outputParentDir, f'model_test_sample_{index}')
    os.makedirs(sample_output_dir, exist_ok=True)
    vsp_file = os.path.join(sample_output_dir, 'wing_geom.vsp3')

    create_wing(
            outputFile = vsp_file,
            aspectRatio = sample_params_list[0],
            span = sample_params_list[1],
            taper = sample_params_list[2],
            sweep = sample_params_list[3],
            thickChord = sample_params_list[4],
            camber = sample_params_list[5],
            camberLoc = sample_params_list[6]
    )
    Xref, Sref = get_Xref_and_Sref(vsp_file)
    print(Sref)
    print(sample_params_list)
    
    vlm_analysis_output_file = os.path.join(sample_output_dir, 'wing_geom_DegenGeom.polar')
    analyse_VLM(AoAStart, AoAEnd, AlphaNpts, Xref, Sref)
    vlm_ld = get_data_from_vlm_output(vlm_analysis_output_file, 'L/D')
    calculatedLDs.append(vlm_ld)
    

print(list(testInput), calculatedLDs)
print(f'RMSE = {calculate_rmse(list(testInput), calculatedLDs)}')
