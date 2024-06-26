# Code repository for the paper _Relevance of the Basset history term for Lagrangian particle dynamics_

This repository contains the Python scripts implemented for the development of the paper _Relevance of the Basset history term for Lagrangian particle dynamics_.

Here one can therefore find the scripts that generate the figures and tables in the following publication:

> _Urizarna-Carasa, L., Ruprecht, D., von Kameke, A., Padberg-Gehle, K., (2024), Relevance of the Basset history term for Lagrangian particle dynamics. To appear in Nonautonomous Dynamical Systems: theory, Methods, and Applications._

For questions, contact either [M. Sc. Julio Urizarna](https://www.linkedin.com/in/julio-urizarna/) or [Prof. Dr. Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht), or [Prof. Dr. Alexandra von Kameke](https://www.haw-hamburg.de/hochschule/beschaeftigte/detail/person/person/show/alexandra-von-kameke/) or [Prof. Dr. Kathrin Padberg-Gehle](https://www.leuphana.de/institute/imd/personen/kathrin-padberg-gehle.html).

## How to run the code

Wihtin this repository you will find 6 folders:

- 01-Trajectory_data : This folder contains the scripts needed to generate the trajectory data.
- 02-FTLE_data : This folder contains the scripts needed to generate the FTLE data.
- 03-FTLE_plots : Contains the plots and the necessary script to obtain the FTLE plots.
- 04-Clustering_plots : Contains the plots and script to obtain final position (clustering) particle plots
- 05-Diversion_data : Contains the script needed to obtain the difference between the with and without BHT solution
- 06-Checkups : Compares the IMEX (with BHT) and solve_ivp (no BHT) solver data with the Daitche (with BHT) and Leap-frog data (no BHT).

Folders 02, 03, 04, 05 and 06 are pretty straightforward, they contain a script and folders where the data and plots are saved. These folders are named after the value of the effective density ratio, R, and the velocity fields.

Folder 01-Trajectory_data works differently, since it contains several Python scripts.

- 00_2mm_Faraday_50Hz_40ms_1_6g.mat has all the field data (positions and velocities) used for the definition of the Faraday flow.
- a00_PMTERS_INPUT.py allows the User to define the parameters (densities, time scale, kinematic viscosity, particle radius, order of the methods used, etc.)
- a00_PMTERS_CONST.py calculates the constants of the MRE (R, S, alpha, etc.) with the parameters given in the previous script. *PLEASE DO NOT MODIFY!*,
- a01_SOLVER_ runs the given solver: DTCHE stands for Daitche's method, IMEX4 stands for the Finite difference methods that we described in [another paper](https://arxiv.org/abs/2403.13515), STOKS stands for the Leap-frog method applied on the MRE without BHT and STOKS_SVIVP stands for the solve_ivp solver applied again on the MRE without BHT.
- a03_FIELD0_ files define the velocity fields and their derivatives (everything needed for the MRE),
- a09_PRTCLE_ files hold the numerical or analytical solver (in case an analytical solution is available).

Generating the data is a long process and depends on the time-steps, the grid used and the field. It could range from less than an hour for the Double Gyre to 10 hours for the Faraday Flow. The times are considering the given grid and time-step run in parallel in a computer with 32 cores.

The toolbox currently depends on classical Python packages such as [Numpy](https://numpy.org/), [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/).

Additional libraries like [progressbar2](https://pypi.org/project/progressbar2/) may also require installation.

## Script naming convenction (in 01-Trajectory_data folder)

Each .py file is made up of a code with the following structure *z99_AAAAAA_BBBBB.py*. Prefix *z99* is linked to the code *AAAAAA*, but enables an appropiate sorting within the folder different from Alphabetical sorting. Code *AAAAAA* summarizes what the code does:

 - either defines parameters (PMTERS),
 - plots trajectories (PLOTTR),
 - defines the velocity fields (FIELD0) or
 - the particle classes associated to each solver (PRTCLE)

 The second part of the code, i.e. *BBBBB*, provides more specific information about the file, such as the type of solver or type of velocity field:

 - CONST stands for the fact that the file defines Constant parameters,
 - BICKL stands for Bickley Jet,
 - DATA1 stands for the Faraday flow, obtained from experimental Data,
 - 00000 stands for the abstract particle class,
 - DTCHE holds Daitche's 1st, 2nd, 3rd order schemes,
 - STOKS holds solvers appliied on the Maxey-Riley equation with History term,
 - IMEX4 holds the solver with either the 1st, 2nd, 3rd and 4th order IMEX.

## Acknowledgements

This repository is a result of the work carried out by 
[ M. Sc. Julio URIZARNA-CARASA](https://www.mat.tuhh.de/home/jurizarna_en), [Prof. Dr. Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht), from Hamburg University of Technology as well as [Prof. Dr. Alexandra von Kameke](https://www.haw-hamburg.de/hochschule/beschaeftigte/detail/person/person/show/alexandra-von-kameke/) from the Hamburg University of Applied Sciences and [Prof. Dr. Kathrin Padberg-Gehle](https://www.leuphana.de/institute/imd/personen/kathrin-padberg-gehle.html) from the Leuphana Universität Lüneburg.

<p align="center">
  <img src="./Logos/tuhh-logo.png" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./Logos/leuphana_logo.svg" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./Logos/HAW_Marke_weiss_RGB.svg" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;
</p>

This project is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - SFB 1615 - 503850735.

<p align="center">
  <img src="./Logos/tu_SMART_LOGO_02.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>
