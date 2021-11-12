 # Installation steps for ubuntu

 cd notebooks
 git clone https://github.com/jlizier/jidt.git

 sudo apt install default-jdk ant (already in tensyj)

 conda install -c conda-forge jpype1 (to do in container) - Done tensij2

 # TO DO IN THE CONTAINER - Done

 install jpype 
 create a volume for jidt and do the git clone and build automatically

 additional installations:
 (week 8 loading mnist experiments)
 conda install -c pytorch pytorch
 conda install -c pytorch torchvision
 conda install -c conda-forge pytorch-lightning
 /opt/conda/bin/python -m pip install -U torch-tb-profiler


 # running build scripts

 https://github.com/jlizier/jidt/wiki/AntScripts

 run "ant build" in the top-level directory of the distribution,


 # Autoanalyzer


 ## Launch autoanalyzer

 (base) jovyan@91c9341c2280:~/notebooks/jidt/demos/AutoAnalyser$ . ./launchAutoAnalyser.sh
Exception in thread "main" java.awt.HeadlessException: 
No X11 DISPLAY variable was set, but this program performed an operation which requires it.
        at java.desktop/java.awt.GraphicsEnvironment.checkHeadless(GraphicsEnvironment.java:208)
        at java.desktop/java.awt.Window.<init>(Window.java:548)
        at java.desktop/java.awt.Frame.<init>(Frame.java:423)
        at java.desktop/java.awt.Frame.<init>(Frame.java:388)
        at java.desktop/javax.swing.JFrame.<init>(JFrame.java:180)
        at infodynamics.demos.autoanalysis.AutoAnalyserLauncher.<init>(Unknown Source)
        at infodynamics.demos.autoanalysis.AutoAnalyserLauncher.main(Unknown Source)

## finding my container named volume files in File Explorer

\\wsl$\docker-desktop-data\version-pack-data\community\docker\volumes\notebooks\_data\CSYS5030

\\wsl$\docker-desktop-data\version-pack-data\community\docker\volumes\notebooks\_data\jidt

## launchAutoAnalyser from Windows 

After having copied the jar file from the File Explorer wsl into D:\src and navigating where my java.exe is : 

PS D:\OpenJDK\jdk-16.0.2\bin> .\java -jar D:\src\infodynamics.jar

# Running a python script

(base) jovyan@91c9341c2280:~/notebooks/jidt/demos/python$ /opt/conda/bin/python example1TeBinaryData.py
