# Data Acquisition 
The scripts use the WieDiversIstMeinGarten (aka Birdiary) API. 
Documentation can be found at [Birdiary API Specification (1.0)](https://wiediversistmeingarten.org/doc).

First the stations are listed, then the movements of each station.
To make data quality assurance easier, only the movements with a human validation are saved for the next step.

Using the list of movements, each video is downloaded. 

Lastly, the videos are split into frames at a 2 FPS rate.