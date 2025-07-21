from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import imageio
import plot

def generateGIF(data, gifName, fps):
    clip = ImageSequenceClip(list(data), fps=fps)
    clip.write_gif(gifName, fps=fps)
    
def generateColoredGIF(data, gifName, duration, cmap=plot.getPrecCMap(), plotType=None):
    if cmap == 't':
        cmap = plot.getTempCMap()
    if plotType == 'c':
        plt.style.use('classic')
    for i in range(data.shape[0]):
        x = data[i]
        plt.imshow(x, cmap)
        plt.axis('off')
        plt.savefig('/FYP-DLSRWF/anim/' + str(i) + '.PNG', bbox_inches='tight',transparent=True, pad_inches=0)
    
    images = []
    filenames =[]
    for i in range(data.shape[0]):
        filenames.append(str(i) + '.PNG')
        
    with imageio.get_writer('/FYP-DLSRWF/anim/' + gifName, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread('/FYP-DLSRWF/anim/' + filename)
            writer.append_data(image)