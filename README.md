# SPDTM-Optimized-Differentiable-Caustic-Image-Projection-Surface-Solver
A Mitsuba based script that solves for Diamond Turning Machining optimized surface geometry to project a caustic image. 

Download Mitsuba example scenes to run. 

NOTE: This is mainly used for notetaking and library is mostly nonfunctional in terms of the scope. The Optimal transport math initializes geometry correctly at a reflected 45deg angle. 
This library is mainly used for analysis of the produced geometry among other notes but the differentiable optimization has yet to produce any desireable effects beyond the OT geometry.



If you are looking for reflective/refractive caustic surface generation in general check out this other repo:
https://github.com/dylanmsu/fast_caustic_design

The scripts contained in this library contain various notes and concepts on how to turn OT initialized geometry into finalized high contrast/ spatial frequency caustic images into a manufacturable surface. No generalized solution is formed.

The overall boilerplate code in this script in theory could be assembled into a functioning version of a optimization pipeline outlined in the paper below if the following was completed:
1) The OT initialization geometry needs to be converted into python instead of C++ and integrated into the differentiation pipeline
2) The differentiable optimization loss function needs to be developed to consistently produce high contrast caustics (Many various loss functions are described in the optimizer script)
3) Finally for manufacturability a loss function that describes swept surface tool clearance orthogonally along the tool radius as well as in the conical clearance face direction along the spiral or raster (etc.) toolpath direction need to be implemented into the surface generation before OT initialization.
4) If you do all that pls message me I would want to mess around with the tool. Manufacutring the finalized surface geometry to nanometer form accuracy is more accessible to me than the time needed to implement the rest of the code :(.

Paper describing end goal:
https://dl.acm.org/doi/10.1145/3732284

There are many other more functional implementations of these concepts but hopefully there might be some useful ideas/code in here. 

Example image showing base heightmap before and after filtering for DT surface profile.
<img width="1554" height="819" alt="image" src="https://github.com/user-attachments/assets/5990fb62-ffaf-463e-87cf-ff7baadf9b62" />

Nanocam surface profile after export into XYZ 3d Surface.
<img width="1421" height="922" alt="image" src="https://github.com/user-attachments/assets/ea86413b-4f50-43b0-896b-43d8876d8e95" />

Toolpath simulation.
<img width="1019" height="747" alt="image" src="https://github.com/user-attachments/assets/73242347-bddb-4e8d-aed3-017297dc3f70" />

Tool clearance violations before optimization.
<img width="1544" height="544" alt="image" src="https://github.com/user-attachments/assets/97325569-f177-4615-97e7-057f0c1b692d" />

Surface profile extremes shown in Ansys SPEOS before toolpath smoothing optimizations.
<img width="684" height="533" alt="image" src="https://github.com/user-attachments/assets/a2152af7-61b3-41ba-b0ba-bb6b4fc4431d" />

Tool clearance violations after optimization.
<img width="690" height="876" alt="image" src="https://github.com/user-attachments/assets/505c8014-078d-4f27-b042-9719613b59e3" />

Verified surface projection result in blender.
<img width="690" height="334" alt="image" src="https://github.com/user-attachments/assets/204d111c-3586-4a06-9b89-e5e93de25264" />
<img width="694" height="349" alt="image" src="https://github.com/user-attachments/assets/3342d09d-caa7-4fb3-ae2d-5955bd908593" />


As an overview this script is not finalized in its form of generating Diamond Turning machinable surfaces other than initializing the geometry to achieve the desired caustic image. The next steps would include initializing the starting geometry with an optimal transport problem and then during the differentiable optimization merit function include swept surface tool clearance loss functions.
