
from matplotlib import pyplot as plt
import openvsp as vsp
def calculate_data_for_plotting(
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

def plot_analysis(
            xValues:list[float] = None,
            yValues:list[float] = None,
            pointLabels:list[str] = None,
            output_file: str = None,
            plotTitle:str = '', 
            xLabel:str = '',
            yLabel:str = '',
    ):
    plt.scatter(xValues, yValues)
    plt.title(plotTitle)
    for label, x, y in zip(pointLabels, xValues, yValues):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.savefig(output_file)
