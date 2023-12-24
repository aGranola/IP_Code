
from matplotlib import pyplot as plt
import openvsp as vsp
from utility_functions import get_data_from_vsp_file, get_data_from_vlm_output

def calculate_data_for_plotting(
        inputVSPFile:str,
        inputVariable:str,
        inputXSecName:str,
        outputPolarFile:str,
        outputVariable:str = None
    ):
    xValue = get_data_from_vsp_file(
        inputVSPFile,
        inputVariable,
        inputXSecName,
    )

    yValue = get_data_from_vlm_output(
        outputPolarFile,
        outputVariable
    )

    return xValue, yValue

def plot_analysis(
            xValues:list[float],
            yValues:list[float],
            pointLabels:list[str],
            output_file: str,
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
