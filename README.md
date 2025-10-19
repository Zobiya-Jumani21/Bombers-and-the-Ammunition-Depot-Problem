Monte Carlo Bombing Simulation
Project Description

This project demonstrates a Monte Carlo Simulation — a type of static simulation used to estimate probabilities using random sampling.
The example models a squadron of bombers attempting to hit an ammunition depot. Each bomb’s landing point is determined by random variations in the horizontal and vertical directions, following a normal distribution.

 How It Works
The ammunition depot is represented as a polygon (either entered through coordinates or drawn manually).

Each bomb’s impact point (X, Y) is generated using:

X=Zx​×σx​,Y=Zy​×σy​
	​

 where Zx, Zy are random normal numbers.

The program checks if each bomb lands inside the depot — counting hits and misses.

The simulation results are displayed in a table and scatter plot with red (hits) and blue (misses).

 Features

Interactive GUI using CustomTkinter.

Two input options:

Type coordinates manually.

Draw the depot polygon directly.

Displays visual results, hit probability, and execution time.

🧠 Purpose

This project shows how Monte Carlo methods can simulate real-world uncertainty — in this case, the inaccuracy of bomb drops — and estimate the probability of successful hits.
