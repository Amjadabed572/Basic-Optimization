# Animation and Robotics - Assignment 1: <br> Optimization and visualization basics

## Introduction
Both animation and robotics heavily rely on optimization algorithms. In order to understand what happens inside the optimizer, and to be able to debug problems efficiently, we rely on *interactive* visualization techniques. "Interactive" means that it is possible to change parameters during runtime and see the change in result immediately, without having to stop the application, edit, and run again.
In this introductory assignment you will experiment with basic optimization and visualization techniques. The goal is to introduce you a different way of coding that is geared toward interactive applications. This will be important in future assignments.

## Instructions
Preliminary steps:
1. Make a Github account and get your Github student benefits at:
   
   `https://education.github.com/discount_requests/application`

   You will need to use your University email and have a scan of an up-to-date student card.
2. Install Python and VSCode. On Windows, you can open the Command Prompt and type

    `winget install Python.Python.3.12 --scope machine`
    `winget install Microsoft.VisualStudioCode --scope machine --override "/silent /mergetasks='!runcode,addcontextmenufiles,addcontextmenufolders,associatewithfiles,addtopath'"` 
3. Open VS Code and install the Python extension (`ctrl-shift-x` and search `python` and then `install`), and the Jupyter extension.
4. Install the Github Copilot extension and log in to it.

Setup steps:
1. Create a folder with no spaces and no non-english characters (Preferably in `C:\Courses\AnimationAndRobotics\Assignments\`) and clone the assignment repository with `git clone`:

    `git clone https://github.com/HaifaGraphicsCourses/animation-and-robotics-optimization-[your github id]`
    
    This will create a new folder that contains this repository.
2. Open the folder with VS Code.
3. Create a new Python environment (`ctrl-shift-p`, type `python env` and select `Python: Create Environment`). Follow the steps. VS Code should create a new folder called `.venv`.
4. Open a new terminal (`` ctrl-shift-` ``). If VS Code detected the Python environment correctly, the prompt should begin with `(.venv)`. If not, restart VS Code and try again. If it still doesn't make sure the default terminal is `cmd` or `bash` (use `CTRL-SHIFT-p` and then `Terminal: Select Default Profile` to change it) and start a new terminal. If it still fails, ask for help.
5. Install Vedo, a scientific visualization package for python, using `pip install vedo` in the terminal.
6. Open `Assignment1.py`. The file is divided into cells, where each cell is defined by `#%%`. Run the first cell, which contains the following code, by pressing `ctrl-enter`.
   ```python
    #%% Imports
    import vedo as vd
    import numpy as np
    from vedo.pyplot import plot
    from vedo import Latex

    vd.settings.default_backend= 'vtk'
    ```
    On the first run, VS Code will tell you that it needs to install the ipykernel runtime.
7.  Run the whole file, cell-by-cell, by pressing `shift-enter`. Running the last cell should result in a new window appearing, with a surface on it.
8.  Congrats! You can start working now!

## Tasks
You are required to write a report in a markdown format (`.md`). The report will be fetched and processed automatically, so it is important you follow these steps:
1. Create an .md file with the work `report` in its name (e.g. `report.md`) and put it in the root path of the repository.
2. Put your full name and ID number somewhere in the report.

The report should be divided into tasks. In each part, use images and videos, with extra text to complement them, to show what you did. The images and videos should be added to the repository. They should clearly and *concisely* demonstrate the tasks were performed correctly. Avoid adding too many images, or videos longer than 5 seconds, as this does not add value to the report and in fact can even make it worse. 

Before you begin, take a moment to interact with the GUI. Try to change the view with the mouse. As you move the mouse on the surface or under it, you will create a path. Take a note of the text flag and the text on the bottom left. The bottom right has a slider that changes the surface's transparency. Pressing `c` on the keyboard will reset the path. Try to press other buttons and see what happens. For more information, check the Vedo documentation.

### Task 1: Understand the code
The code is divided to several cells, each with its own purpose. We will use [Vedo](https://vedo.embl.es/) to create the GUI. The Imports cell contains the package imports. The Callbacks cell contains GUI callbacks. These functions are called in a response to a predefined user action, such as moving the mouse or pressing a button on the keyboard. The Optimization cell contains functions that will be required later for optimization. The Plotting cell contains calls to plotting functions.

To check your understanding, add the following to the code:
1. A callback for a mouse right button press. Choose what will happen and report on it. Be creative.
2. Visualize the function values on the path (the numpy array `Xi`) you made with the mouse with a graph. The graph should update every time a point is added to the path. Use `vedo.plot` and report what happened. Use `vedo.plot.clone2d` to fix it.
3. Add a way for the user to change the function using the GUI. There are many ways of doing that. Explain in the report what you did.

### Task 2: Gradient Descent
The code includes incomplete functions for optimization.
1. Change the code such that mouse motion does not create a path anymore. Do not remove the path code yet, as it will be used to plot the progression of the optimization.
2. Add a left mouse button click callback that, when clicked on the surface or the plane under it, clears the path and sets the initial guess for the optimization to the clicked point.
3. Add a button that runs a fixed number of gradient descent iterations and updates the path. Every click of the button will extend the path with new points. Allow the user to modify the number of steps and the step size. Do not implement back-tracking line search yet.

Demonstrate how gradient descent behaves for different step sizes. Try to experimentally find an optimal step size. Use the graph from task 1 to help with that. 

### Task 3: Newton's Method
1. Change the code to maintain *two* paths, one for gradient descent and one for Newton's method. Make the button for task 2.3 run the same number of Newton iterations and plot the two paths. Do not use line-search yet.
2. In the graph from task 1, show both gradient descent path and Newton's path.
3. One of the tests of a descent direction is the dot product with the gradient. For each Newton step, compute the gradient at that point, and the dot product with the Newton search direction. Show the dot product in the graph as well.
4. Show what happens when the initial point is near a minimum vs. saddle-point and maximum.

### Task 4: Evaluate
We regularly need to know which method performs better. To test this, we need to compare methods in terms of speed and convergence rate.
1. Complete the implementation of the line-search function.
2. When the Hessian is not positive-definite Newton's method might produce an ascent (instead of descent) direction. Implement the two methods we discussed in class to overcome the problem. Compare gradient descent, *vanilla* Newton's method, and the two modifications for several different initializations.
3. How many iterations are needed to converge to a minimum? Devise an automatic way to decide when to stop iterating and report on your approach.

### Task 5: Numerical derivatives
1. The gradients and Hessians in the code are computed *numerically* using finite differences. This is a slow but simple way to obtain them. The alternative is to compute them *analytically* by hand, using derivation rules. Write two new functions that compute the gradient and Hessian of the objective analytically and copy the code to your report.
2. Measure the time it takes to run the numerical vs. analytical computation.
3. The finite difference approximation relies on a finite ε. Compare the values of the finite difference gradients for different ε values with the analytical (true) value. 

## Submission
The last commit before the deadline is the one that will be examined. Later commits will be ignored.

## Grading
Grading will be based on the following criteria:
* Correctness of the results
* Quality of the report
* Creativity
* Effort
* Commit history

The feedback will appear on the feedback pullrequest of your repository.
