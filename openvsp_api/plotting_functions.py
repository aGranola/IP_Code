
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
    plt.show()


def plot_gan_loss(discriminator_losses, generator_losses, output_file:str = 'loss_curve.png'):
    x=discriminator_losses
    y=generator_losses
    x_label='Discriminator Loss'
    y_label='Generator Loss'
    x_axis_label = 'Epochs'
    y_axis_label = 'Loss'
    plot_loss(x, y, x_label, y_label, x_axis_label, y_axis_label, output_file)

    plt.legend()
    plt.savefig(output_file)

def plot_loss_from_hist(hist, output_file:str = 'loss_curve.png'):
    x = hist.history['loss']
    y = hist.history['val_loss']
    x_label='loss'
    y_label='val loss'
    x_axis_label = 'Epochs'
    y_axis_label = 'Loss'
    plot_loss(x, y, x_label, y_label, x_axis_label, y_axis_label, output_file)

def plot_loss(x:list[float],y:list[float], x_label:str, y_label:str, x_axis_label:str, y_axis_label:str, output_file:str):
    fig = plt.figure()
    plt.plot(x, color='blue', label=x_label)
    plt.plot(y, color='orange', label=y_label)
    fig.suptitle('Loss Curve', fontsize=20)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(loc="upper left")
    plt.savefig(output_file)
    plt.show()
    
    
def plot_gan(discriminator_losses, generator_losses, output_file:str = 'loss_curve.png'):
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_file)
    plt.show()