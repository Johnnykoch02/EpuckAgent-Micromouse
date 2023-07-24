# Micromouse Agent

This Python project is an implementation of a Micromouse agent. The agent is designed to traverse a maze-like grid with the goal of reaching the center. The agent uses sensors to observe its environment and update the map of the grid, it also uses algorithms to decide its next action based on the current state of the grid.

## Project Structure

The project is made up of a number of Python classes:

### Vector2

This class represents a 2D vector and is used for various mathematical operations throughout the project.

### Grid

This class represents the grid that the Micromouse agent navigates. It also defines classes for Nodes and Edges, which represent the cells and connections within the grid. The grid keeps track of the current position of the agent and any detected obstacles. It also includes functions for loading and saving the state of the grid to a file.

### TransitionModel

This class performs the calculations associated with transitioning from one grid cell to another. It contains a method `TransitionState()` which handles the transition of the agent from one grid cell to another. 

### Agent

The `Agent` class represents the Micromouse agent itself. It holds information about the agent's state, such as its current position and velocity, and contains methods to update its state based on its sensors. The agent uses a search-based technique to find its way to the center of the grid.

## Usage

After instantiating the `Agent` class and creating a `Grid` object, the agent is placed in the corner of the grid with `grid.set_current_cell_pos(15, 0)`. The destination cell is set to the center of the grid with `grid.set_destination_cell(120)`. The agent is then set to use this grid.

The main loop of the program involves calling the agent's `step()` function repeatedly, which updates the agent's state and moves it through the grid. The loop will continue until the agent's `Status()` function returns -1, indicating that the agent has reached its goal.

To ensure that progress is not lost if the program is interrupted, the state of the grid is saved to a file every 10 minutes using `Grid.save_to_file(agent.grid, 'GridState.json')`.

```

## Requirements

This project has the following dependencies:

- Python 3.7 or higher
- Webots simulation software (for `controller.Robot`, `controller.Camera`)
- numpy (`numpy`)
- OpenCV (`cv2`)
- Multiprocessing (`multiprocessing`)
- JSON (`json`)
- Threading (`threading`)
- Time (`time`)
- Math (`math`)
- Enum (`enum`)
- Collections (`collections`)

These libraries can be installed via pip, the Python package manager. The Webots simulation software must be installed separately. You can install the Python dependencies by running:

```bash
pip install numpy opencv-python
```

For Webots, follow the instructions in its [official documentation](https://cyberbotics.com/doc/guide/installation-procedure).

## Note

Ensure your Python environment matches the version specified and that all packages are installed successfully. If you encounter issues with the installed packages, consider using a Python virtual environment or check your Python version. This repository uses the Webots Robotics Simulator.