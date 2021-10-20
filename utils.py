#------------------------------------------------------------
#   PLOT Conditional Entropies vs Ratios
#------------------------------------------------------------


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(9, 6))  
axs.plot(conditionalEntropies, winRatios, '.', label='winRatios')
axs.plot(conditionalEntropies, lossRatios, '.', label='Loss Ratios')
axs.set_title('Conditional Entropies vs Win Ratios')
axs.set_xlabel('H(X|X-1)')
plt.legend()
plt.show()

#---------------------------------------------------
# START JVM
#-----------------------------------------------------

from jpype import startJVM, getDefaultJVMPath, JArray, JDouble, JInt, JPackage, shutdownJVM, isJVMStarted
import numpy as np
import sys
# Our python data file readers are a bit of a hack, python users will do better on this:
sys.path.append("/home/jovyan/notebooks/jidt/demos/python")
import readIntsFile

# Add JIDT jar library to the path
jarLocation = "/home/jovyan/notebooks/jidt/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
if (not isJVMStarted()):
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

javaIntList = lambda l : JArray(JInt, 1)(l)
javaDoubleList = lambda l : JArray(JDouble, 1)(l)

# LOAD DATA FILE
#-------------------

import pandas as pd
#from pathlib import Path

filename = './data/SFI-heartRate_breathVol_bloodOx-extract.txt'
data = pd.read_csv(filename, delim_whitespace=True, comment='%', header=None, names = ['Heart rate', 'Chest volume', 'Blood oxygen'], index_col=None)
data.head()